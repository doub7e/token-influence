#!/usr/bin/env python3
"""Download and convert DAPO-Math-17k to Archer training JSON format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Archer math training data from HF dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        help="Hugging Face dataset id.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--output",
        type=str,
        default="data/train/archer2.0-math-1.5b-train.json",
        help="Output json file in Archer format.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional truncation for quick tests.",
    )
    return parser.parse_args()


def _to_message_list(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        msgs: list[dict[str, str]] = []
        for item in prompt:
            if isinstance(item, dict) and "content" in item:
                role = str(item.get("role", "user"))
                content = str(item["content"])
                msgs.append({"role": role, "content": content})
        if msgs:
            return msgs
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return []


def _find_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("prompt", "question", "problem", "instruction", "query"):
        if key in row and row[key] is not None:
            msgs = _to_message_list(row[key])
            if msgs:
                return msgs
    return []


def _normalize_answer_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("answer", "answers", "ground_truth", "final_answer", "target", "solution"):
            if key in value:
                return _normalize_answer_value(value[key])
        return [json.dumps(value, ensure_ascii=False)]
    if isinstance(value, list):
        out: list[str] = []
        for x in value:
            out.extend(_normalize_answer_value(x))
        return [x for x in out if x]
    return [str(value).strip()]


def _find_answers(row: dict[str, Any]) -> list[str]:
    for key in (
        "answer",
        "answers",
        "ground_truth",
        "final_answer",
        "target",
        "solution",
        "reward_model",
    ):
        if key in row and row[key] is not None:
            answers = _normalize_answer_value(row[key])
            if answers:
                return answers
    return []


def convert_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    prompt = _find_prompt(row)
    answers = _find_answers(row)

    if not prompt:
        raise ValueError(f"Row {idx} has no usable prompt fields.")
    if not answers:
        raise ValueError(f"Row {idx} has no usable answer fields.")

    return {
        "data_source": "math",
        "prompt": prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answers},
        "extra_info": {"index": idx},
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    rows = ds.to_list()
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    converted: list[dict[str, Any]] = []
    skipped = 0
    for i, row in enumerate(rows):
        try:
            converted.append(convert_row(row, i))
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if skipped <= 5:
                print(f"[WARN] skip row {i}: {exc}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False)

    print(f"[OK] wrote {len(converted)} samples to {output_path}")
    print(f"[INFO] skipped {skipped} rows")
    if converted:
        print(f"[INFO] sample keys: {list(converted[0].keys())}")


if __name__ == "__main__":
    main()
