"""Generate small offline datasets for LLaDA pre-training."""

import json
import os
import random
from pathlib import Path

PRETRAIN_SNIPPETS = [
    "LLaDA is a diffusion-based approach to language modeling without autoregressive factorization.",
    "The forward process randomly masks tokens with probability sampled per sequence.",
    "Training focuses on denoising masked tokens using a Transformer encoder backbone.",
    "This tiny corpus is purely synthetic and exists only to showcase offline training.",
    "Mask prediction encourages the model to recover clean text from noisy inputs.",
    "Diffusion-inspired language models can be trained from scratch with simple code.",
    "The reserved mask token identifier is fixed to 126336 as described in the guidelines.",
]


def _write_split(path: Path, num_samples: int) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(num_samples):
            text = random.choice(PRETRAIN_SNIPPETS)
            if random.random() < 0.25:
                text += " " + random.choice(PRETRAIN_SNIPPETS)
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def main(train_samples: int = 1024, eval_samples: int = 128) -> None:
    root = Path("data/pretrain")
    _write_split(root / "train.jsonl", train_samples)
    _write_split(root / "eval.jsonl", eval_samples)
    print(f"Wrote {train_samples} training samples to {root / 'train.jsonl'}")
    print(f"Wrote {eval_samples} eval samples to {root / 'eval.jsonl'}")


if __name__ == "__main__":
    main()
