from collections.abc import Iterable

import torch


def resolve_module(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    """Resolve a dotted module path produced by model.named_modules()."""
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"Module not found: {module_name}")
    return modules[module_name]


def _tokenize(tokenizer, text: str, device: torch.device, max_length: int) -> dict[str, torch.Tensor]:
    """Use chat formatting when available; otherwise fall back to plain text tokens."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return {"input_ids": ids[:, -max_length:].to(device)}

    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def extract_refusal_direction(
    model: torch.nn.Module,
    tokenizer,
    prompt_pairs: Iterable[tuple[str, str]],
    module_name: str,
    max_length: int = 512,
) -> torch.Tensor:
    """Average harmful-prompt minus refusal-anchor activations at the selected module."""
    was_training = model.training
    model.eval()
    module = resolve_module(model, module_name)
    captured: dict[str, torch.Tensor] = {}

    def hook(_, __, output):
        # Some transformer blocks return tuples; only the activation tensor is needed.
        if isinstance(output, tuple):
            output = output[0]
        captured["activation"] = output.detach()

    handle = module.register_forward_hook(hook)
    diffs = []

    try:
        for harmful_prompt, refusal_anchor in prompt_pairs:
            captured.clear()
            model(**_tokenize(tokenizer, harmful_prompt, model.device, max_length))
            # Last-token activations provide a compact prompt-level direction estimate.
            harmful_act = captured["activation"][0, -1].float()

            captured.clear()
            model(**_tokenize(tokenizer, refusal_anchor, model.device, max_length))
            refusal_act = captured["activation"][0, -1].float()

            diffs.append(harmful_act - refusal_act)
    finally:
        handle.remove()
        if was_training:
            model.train()

    if not diffs:
        raise ValueError("At least one prompt/refusal pair is required.")

    direction = torch.stack(diffs).mean(dim=0)
    return direction / direction.norm().clamp_min(1e-8)
