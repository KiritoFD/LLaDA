#!/usr/bin/env python3
"""Run inference with the locally trained SimpleTransformer checkpoints.

This script reconstructs the tokenizer and model from a saved training checkpoint
and performs masked-diffusion style generation using the sampling routine from
``generate.py``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from generate import generate
from training.train_completely_offline import SimpleTokenizer, SimpleTransformer


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[SimpleTransformer, SimpleTokenizer, dict]:
    """Load model, tokenizer, and config from a saved training checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})

    tokenizer = SimpleTokenizer(vocab_size=saved_args.get("vocab_size", 20000))
    tokenizer.load_state(
        checkpoint["tokenizer_vocab"],
        checkpoint.get("tokenizer_special", tokenizer.special_tokens),
        checkpoint.get("mask_token_id", tokenizer.mask_token_id),
        checkpoint.get("embedding_vocab_size", len(tokenizer)),
    )

    model = SimpleTransformer(
        vocab_size=len(tokenizer),
        max_length=saved_args.get("max_length", 512),
        d_model=saved_args.get("d_model", 256),
        nhead=saved_args.get("nhead", 8),
        num_layers=saved_args.get("num_layers", 6),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    model.device = device  # used by the sampling helper

    return model, tokenizer, saved_args


def _encode_prompt(tokenizer: SimpleTokenizer, text: str, max_length: int) -> torch.Tensor:
    encoding = tokenizer.encode(text, max_length=max_length, padding=False)
    return encoding["input_ids"].unsqueeze(0)


def _decode_response(tokenizer: SimpleTokenizer, token_ids: Iterable[int]) -> str:
    filtered = [int(t) for t in token_ids if int(t) != tokenizer.mask_token_id]
    return tokenizer.decode(filtered, skip_special_tokens=True).strip()


def _prepare_generation_hyperparams(gen_length: int, block_length: int, steps: int) -> Tuple[int, int, int]:
    if gen_length <= 0:
        raise ValueError("gen_length must be positive")

    block_length = max(1, min(block_length, gen_length))
    if gen_length % block_length != 0:
        # Fallback to single block to satisfy sampling assertions
        block_length = gen_length

    num_blocks = max(1, gen_length // block_length)
    adjusted_steps = max(1, steps)
    if adjusted_steps % num_blocks != 0:
        adjusted_steps = num_blocks * math.ceil(adjusted_steps / num_blocks)

    return gen_length, block_length, adjusted_steps


def generate_from_prompt(
    model: SimpleTransformer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_length: int,
    gen_length: int,
    block_length: int,
    steps: int,
    temperature: float,
    cfg_scale: float,
) -> str:
    prompt_tensor = _encode_prompt(tokenizer, prompt, max_length)
    prompt_tensor = prompt_tensor.to(model.device)
    prompt_length = prompt_tensor.shape[1]

    if prompt_length >= max_length:
        raise ValueError(
            f"Prompt is too long for the model (prompt length {prompt_length} >= max_length {max_length})."
        )

    available_space = max_length - prompt_length
    if gen_length > available_space:
        gen_length = available_space

    gen_length, block_length, steps = _prepare_generation_hyperparams(gen_length, block_length, steps)

    with torch.no_grad():
        out = generate(
            model,
            prompt_tensor,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
            mask_id=tokenizer.mask_token_id,
        )

    response_tokens = out[0, prompt_length:].detach().cpu().tolist()
    return _decode_response(tokenizer, response_tokens)


def _iter_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"Prompt file {prompt_path} is empty")
        return prompts
    return [args.prompt]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline LLaDA inference with SimpleTransformer checkpoints")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs_offline/best_model.pt"),
        help="Path to the checkpoint produced by train_completely_offline.py",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="介绍一下LLaDA模型。",
        help="Prompt text when not using --prompt-file",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional file with one prompt per line",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=64,
        help="Number of tokens to generate per prompt",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Block length for semi-autoregressive sampling",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of sampling steps (rounded up to multiples of blocks)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (defaults to CUDA when available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, saved_args = _load_checkpoint(args.checkpoint, device)
    model_max_length = saved_args.get("max_length", model.max_length)

    prompts = _iter_prompts(args)
    for idx, prompt in enumerate(prompts, start=1):
        response = generate_from_prompt(
            model,
            tokenizer,
            prompt,
            max_length=model_max_length,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )
        print("=" * 80)
        print(f"Prompt {idx}: {prompt}")
        print("-" * 80)
        print(response if response else "<空响应>")
        print()


if __name__ == "__main__":
    main()
