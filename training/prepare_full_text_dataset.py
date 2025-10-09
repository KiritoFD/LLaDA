"""Convert a textbook-style plain text file into a JSONL dataset for pre-training.

Each paragraph (split by blank lines) becomes one or more JSON records with a
`text` field. Long paragraphs can optionally be re-chunked into smaller pieces.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, List

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?])")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(raw_text: str) -> List[str]:
    # Split on blank lines while keeping content order
    blocks = re.split(r"\n\s*\n", raw_text)
    paragraphs: List[str] = []
    for block in blocks:
        normalized = normalize_whitespace(block)
        if normalized:
            paragraphs.append(normalized)
    return paragraphs


def chunk_paragraph(text: str, max_chars: int) -> Iterator[str]:
    if len(text) <= max_chars:
        yield text
        return

    # Prefer sentence boundaries when chunking
    sentences = [normalize_whitespace(s) for s in SENTENCE_SPLIT_PATTERN.split(text) if normalize_whitespace(s)]
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + len(sentence) > max_chars:
            yield " ".join(current).strip()
            current = []
            current_len = 0
        current.append(sentence)
        current_len += len(sentence)

    if current:
        yield " ".join(current).strip()


def iter_chunks(paragraphs: Iterable[str], *, max_chars: int, min_chars: int) -> Iterator[str]:
    for paragraph in paragraphs:
        for chunk in chunk_paragraph(paragraph, max_chars=max_chars):
            if len(chunk) >= min_chars:
                yield chunk


def prepare_jsonl(source: Path, output: Path, *, max_chars: int, min_chars: int) -> int:
    raw_text = source.read_text(encoding="utf-8")
    paragraphs = split_paragraphs(raw_text)
    chunks = list(iter_chunks(paragraphs, max_chars=max_chars, min_chars=min_chars))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk}, ensure_ascii=False))
            f.write("\n")

    return len(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a plain text textbook into JSONL format")
    parser.add_argument("--source", type=Path, default=Path("xad.txt"), help="Path to the source text file")
    parser.add_argument("--output", type=Path, default=Path("data/xad_full_text.jsonl"), help="Destination JSONL file")
    parser.add_argument("--max_chars", type=int, default=1200, help="Maximum characters per chunk")
    parser.add_argument("--min_chars", type=int, default=40, help="Minimum characters per chunk to keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = prepare_jsonl(args.source, args.output, max_chars=args.max_chars, min_chars=args.min_chars)
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
