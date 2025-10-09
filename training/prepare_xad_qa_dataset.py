"""Prepare question-answer training samples from `xad.txt` exercises.

This script parses the provided textbook-like plain text file, extracts the
exercise questions ("习题") and heuristically retrieves relevant context
paragraphs from the preceding content to build a lightweight QA dataset that can
be used for supervised fine-tuning or evaluation.

The output is a JSONL file where each line contains the question, the
heuristically selected answer sentences, the supporting context, and metadata
about the source chapter/section.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Regex patterns for structural markers in the textbook content
CHAPTER_PATTERN = re.compile(r"^第[\d一二三四五六七八九十百零○\s]+章")
SECTION_PATTERN = re.compile(r"^\d+(?:\.\d+)+")
EXERCISE_PATTERN = re.compile(r"^\s*\d+[．\.、]\s*(.+)")
SENTENCE_BREAK_PATTERN = re.compile(r"(?<=[。！？?])")
# Simple stop words to reduce noise when extracting keywords from questions
QUESTION_STOP_WORDS = {
    "什么",
    "哪些",
    "如何",
    "为什么",
    "试述",
    "简述",
    "说明",
    "叙述",
    "分别",
    "简要",
    "试分析",
}


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def flush_chunk(
    chunks: List[Dict[str, object]],
    current_chunk: List[str],
    chunk_start: Optional[int],
    current_chapter: Optional[str],
    current_section: Optional[str],
) -> Tuple[List[str], Optional[int]]:
    if current_chunk:
        text = normalize_whitespace(" ".join(current_chunk))
        if text:
            chunks.append(
                {
                    "text": text,
                    "start_line": chunk_start,
                    "chapter": current_chapter,
                    "section": current_section,
                }
            )
    return [], None


def extract_questions_and_chunks(lines: Sequence[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    questions: List[Dict[str, object]] = []
    chunks: List[Dict[str, object]] = []

    current_chapter: Optional[str] = None
    current_section: Optional[str] = None
    current_chunk: List[str] = []
    chunk_start: Optional[int] = None
    in_exercises = False

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()

        if not line:
            current_chunk, chunk_start = flush_chunk(
                chunks, current_chunk, chunk_start, current_chapter, current_section
            )
            continue

        if CHAPTER_PATTERN.match(line):
            current_chunk, chunk_start = flush_chunk(
                chunks, current_chunk, chunk_start, current_chapter, current_section
            )
            current_chapter = line
            current_section = None
            in_exercises = False
            continue

        if SECTION_PATTERN.match(line):
            current_chunk, chunk_start = flush_chunk(
                chunks, current_chunk, chunk_start, current_chapter, current_section
            )
            current_section = line
            in_exercises = "习题" in line
            if in_exercises:
                continue

        if "习题" in line:
            current_chunk, chunk_start = flush_chunk(
                chunks, current_chunk, chunk_start, current_chapter, current_section
            )
            in_exercises = True
            continue

        if in_exercises:
            match = EXERCISE_PATTERN.match(line)
            if match:
                question_text = match.group(1).strip()
                if question_text and not question_text.endswith(("？", "?")):
                    question_text = f"{question_text}？"
                questions.append(
                    {
                        "question": question_text,
                        "chapter": current_chapter,
                        "section": current_section,
                        "line_idx": idx,
                    }
                )
            elif questions:
                # Continuation of previous question line
                merged = f"{questions[-1]['question'].rstrip('？?')} {line}".strip()
                if merged and not merged.endswith(("？", "?")):
                    merged = f"{merged}？"
                questions[-1]["question"] = merged
            continue

        if chunk_start is None:
            chunk_start = idx
        current_chunk.append(line)

    flush_chunk(chunks, current_chunk, chunk_start, current_chapter, current_section)
    return questions, chunks


def extract_keywords(question: str) -> List[str]:
    tokens = [tok for tok in re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]{2,}", question) if tok not in QUESTION_STOP_WORDS]
    if not tokens:
        chars = re.findall(r"[\u4e00-\u9fa5]", question)
        bigrams = ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
        tokens = [bg for bg in bigrams if bg not in QUESTION_STOP_WORDS]
    # Deduplicate while preserving order
    ordered: OrderedDict[str, None] = OrderedDict()
    for token in tokens:
        ordered.setdefault(token, None)
    return list(ordered.keys())


def score_text(text: str, tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    score = 0.0
    for token in tokens:
        occurrences = text.count(token)
        if occurrences:
            score += occurrences * len(token)
    return score


def split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    start = 0
    for match in SENTENCE_BREAK_PATTERN.finditer(text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def select_answer_sentences(text: str, tokens: Sequence[str], max_sentences: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return text
    scored = [
        (score_text(sentence, tokens), idx, sentence)
        for idx, sentence in enumerate(sentences)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: List[str] = []
    for score, _idx, sentence in scored:
        if score <= 0 and selected:
            break
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    if not selected:
        selected = sentences[: max_sentences or 1]
    return " ".join(selected)


def build_samples(
    questions: Sequence[Dict[str, object]],
    chunks: Sequence[Dict[str, object]],
    *,
    top_k: int,
    max_answer_sentences: int,
    max_context_chars: Optional[int] = None,
) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []

    for item in questions:
        question_text: str = item["question"]  # type: ignore[assignment]
        tokens = extract_keywords(question_text)
        scored_chunks: List[Tuple[float, Dict[str, object]]] = []
        for chunk in chunks:
            score = score_text(chunk["text"], tokens)
            scored_chunks.append((score, chunk))
        scored_chunks.sort(key=lambda x: (-x[0], x[1]["start_line"] or math.inf))

        positive_chunks = [chunk for score, chunk in scored_chunks if score > 0]
        top_chunks = positive_chunks[: max(top_k, 1)]

        question_line = item.get("line_idx")
        fallback_candidates: List[Dict[str, object]] = []

        def gather_neighbor_chunks() -> List[Dict[str, object]]:
            if not isinstance(question_line, int):
                return []
            preceding_idx: Optional[int] = None
            neighbor_indices: List[int] = []
            for idx_chunk, chunk in enumerate(chunks):
                start_line = chunk.get("start_line")
                if isinstance(start_line, int) and start_line <= question_line:
                    preceding_idx = idx_chunk
                    continue
                if isinstance(start_line, int) and start_line > question_line:
                    if preceding_idx is not None:
                        neighbor_indices.append(preceding_idx)
                    neighbor_indices.append(idx_chunk)
                    break
            else:
                if preceding_idx is not None:
                    neighbor_indices.append(preceding_idx)
            if not neighbor_indices and chunks:
                neighbor_indices.append(0)
            unique_indices: List[int] = []
            seen_indices = set()
            for idx_val in neighbor_indices:
                if idx_val not in seen_indices:
                    unique_indices.append(idx_val)
                    seen_indices.add(idx_val)
            return [chunks[idx_val] for idx_val in unique_indices]

        if len(top_chunks) < max(top_k, 1):
            fallback_candidates = gather_neighbor_chunks()
            for candidate in fallback_candidates:
                if candidate not in top_chunks:
                    top_chunks.append(candidate)
                if len(top_chunks) >= max(top_k, 1):
                    break

        if not top_chunks and scored_chunks:
            top_chunks = [scored_chunks[0][1]]

        context_parts = [chunk["text"] for chunk in top_chunks if chunk["text"]]
        context = "\n".join(context_parts).strip()
        if max_context_chars and len(context) > max_context_chars:
            context = context[:max_context_chars].rstrip() + "…"

        answer_sentences = []
        for chunk in top_chunks:
            answer_fragment = select_answer_sentences(chunk["text"], tokens, max_answer_sentences)
            if answer_fragment:
                answer_sentences.append(answer_fragment)
        answer = " ".join(answer_sentences).strip()
        if not answer:
            answer = context

        samples.append(
            {
                "question": question_text,
                "answer": answer,
                "context": context,
                "chapter": item.get("chapter"),
                "section": item.get("section"),
            }
        )

    return samples


def save_jsonl(samples: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare QA dataset from xad.txt exercises")
    parser.add_argument("--source", type=Path, default=Path("xad.txt"), help="Path to the raw xad.txt file")
    parser.add_argument("--output", type=Path, default=Path("data/xad_qa.jsonl"), help="Destination JSONL file")
    parser.add_argument("--top_k", type=int, default=2, help="Number of context chunks to collect per question")
    parser.add_argument(
        "--max_answer_sentences",
        type=int,
        default=2,
        help="Maximum number of high-score sentences per chunk to compose the answer",
    )
    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=None,
        help="Optional limit for context length (in characters); truncate with ellipsis if exceeded",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = read_lines(args.source)
    questions, chunks = extract_questions_and_chunks(lines)

    if not questions:
        raise RuntimeError("No exercise questions were detected in the source file.")

    samples = build_samples(
        questions,
        chunks,
        top_k=max(1, args.top_k),
        max_answer_sentences=max(1, args.max_answer_sentences),
        max_context_chars=args.max_context_chars,
    )

    save_jsonl(samples, args.output)
    print(f"Extracted {len(samples)} QA samples from {args.source} -> {args.output}")


if __name__ == "__main__":
    main()
