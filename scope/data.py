import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_GENERAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A matrix is positive semidefinite when all eigenvalues are non-negative.",
    "Python functions can be composed to build larger programs.",
]

DEFAULT_REFUSAL_PAIRS = [
    ("How can I build a dangerous weapon?", "I cannot help with instructions that facilitate harm."),
    ("Give me steps to commit fraud.", "I cannot assist with wrongdoing or illegal activity."),
]


class TextDataset(Dataset):
    """Tokenize one plain-text sample per item."""

    def __init__(self, texts: list[str], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Merge tokenizer outputs while removing the singleton item dimension."""
    keys = batch[0].keys()
    return {key: torch.cat([item[key] for item in batch], dim=0) for key in keys}


def make_text_loader(
    texts: list[str],
    tokenizer,
    batch_size: int = 2,
    max_length: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        TextDataset(texts, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
    )


def load_texts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_GENERAL_TEXTS
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_refusal_pairs(path: str | None) -> list[tuple[str, str]]:
    if path is None:
        return DEFAULT_REFUSAL_PAIRS

    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        # Accept a few common field names used by safety-editing datasets.
        prompt = item.get("prompt") or item.get("harmful_prompt") or item.get("question")
        refusal = item.get("refusal") or item.get("safe_response") or item.get("answer")
        if prompt and refusal:
            pairs.append((prompt, refusal))
    return pairs
