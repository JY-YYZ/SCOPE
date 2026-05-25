import argparse
from itertools import cycle

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scope import (
    ActivationCovariance,
    extract_refusal_direction,
    project_weight_gradient,
    refusal_alignment_loss,
    resolve_module,
)
from scope.data import load_refusal_pairs, load_texts, make_text_loader


def parse_args() -> argparse.Namespace:
    """Collect the minimal knobs needed for a reproducible editing run."""
    parser = argparse.ArgumentParser(description="Minimal SCOPE-style safety editing run.")
    parser.add_argument("--model", required=True, help="Base Hugging Face model path or id.")
    parser.add_argument("--output", required=True, help="Directory for the edited model.")
    parser.add_argument("--target-modules", nargs="+", default=["model.layers.15.mlp.down_proj"])
    parser.add_argument("--general-texts", default=None, help="Optional text file for activation covariance.")
    parser.add_argument("--expert-texts", default=None, help="Optional text file for DNSE-style covariance expansion.")
    parser.add_argument("--safety-pairs", default=None, help="Optional JSONL with prompt/refusal fields.")
    parser.add_argument("--max-cov-batches", type=int, default=8)
    parser.add_argument("--edit-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--energy-threshold", type=float, default=0.95)
    parser.add_argument("--min-null-dim", type=int, default=256)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    return parser.parse_args()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a tokenized batch onto the same device as the model inputs."""
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def build_projection(model, tokenizer, module_name: str, texts: list[str], args) -> torch.Tensor:
    was_training = model.training
    model.eval()
    module = resolve_module(model, module_name)
    hidden_dim = module.weight.shape[0]
    covariance = ActivationCovariance(hidden_dim=hidden_dim, device=module.weight.device, dtype=module.weight.dtype)
    loader = make_text_loader(texts, tokenizer, batch_size=args.batch_size, max_length=args.seq_len)
    cache: list[torch.Tensor] = []

    def hook(_, __, output):
        # Capture the selected module output without retaining the full forward graph.
        if isinstance(output, tuple):
            output = output[0]
        cache.append(output.detach())

    handle = module.register_forward_hook(hook)
    try:
        for step, batch in enumerate(loader):
            if step >= args.max_cov_batches:
                break
            model(**move_batch(batch, model.device))
            # General and optional expert activations are accumulated into one shield.
            covariance.update(cache.pop())
    finally:
        handle.remove()
        if was_training:
            model.train()

    return covariance.null_space_projection(
        energy_threshold=args.energy_threshold,
        min_null_dim=args.min_null_dim,
    )


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.train()

    general_texts = load_texts(args.general_texts)
    expert_texts = load_texts(args.expert_texts) if args.expert_texts else []
    # DNSE-style expansion is represented by adding expert-domain activations.
    covariance_texts = general_texts + expert_texts
    refusal_pairs = load_refusal_pairs(args.safety_pairs)

    projections = {
        name: build_projection(model, tokenizer, name, covariance_texts, args)
        for name in args.target_modules
    }
    refusal_dirs = {
        name: extract_refusal_direction(model, tokenizer, refusal_pairs, name, max_length=args.seq_len)
        for name in args.target_modules
    }
    model.train()

    ref_model = None
    if args.kl_weight > 0:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    modules = {name: resolve_module(model, name) for name in args.target_modules}
    optimizer = torch.optim.AdamW([p for module in modules.values() for p in module.parameters()], lr=args.lr)
    activations: dict[str, torch.Tensor] = {}
    handles = []

    for name, module in modules.items():
        def hook(_, __, output, module_name=name):
            # These activations remain attached to the graph for the refusal loss.
            if isinstance(output, tuple):
                output = output[0]
            activations[module_name] = output

        handles.append(module.register_forward_hook(hook))

    safety_prompts = [prompt for prompt, _ in refusal_pairs]
    general_loader = cycle(make_text_loader(general_texts, tokenizer, batch_size=args.batch_size, max_length=args.seq_len))

    try:
        for step, prompt in zip(range(args.edit_steps), cycle(safety_prompts)):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.seq_len).to(model.device)
            model(**inputs)

            safety_loss = sum(
                refusal_alignment_loss(activations[name][0, -1], refusal_dirs[name])
                for name in args.target_modules
            )
            loss = safety_loss

            if ref_model is not None:
                batch = move_batch(next(general_loader), model.device)
                with torch.no_grad():
                    ref_logits = ref_model(**batch).logits
                logits = model(**batch).logits
                kl_loss = F.kl_div(
                    F.log_softmax(logits.float(), dim=-1),
                    F.softmax(ref_logits.float(), dim=-1),
                    reduction="batchmean",
                )
                loss = loss + args.kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()

            for name, module in modules.items():
                # The SCOPE update edits only the covariance null space.
                project_weight_gradient(module, projections[name], max_norm=args.max_grad_norm)

            optimizer.step()
            print(f"step={step + 1} loss={float(loss.detach().cpu()):.4f}")
    finally:
        for handle in handles:
            handle.remove()

    model.eval()
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
