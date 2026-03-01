#!/usr/bin/env python3
"""Download MATH-500 from HuggingFace and convert to Archer2.0 evaluation format."""

import json
import os

from datasets import load_dataset


def main():
    output_path = os.path.join(os.path.dirname(__file__), "../../data/test/math500.json")

    print("Downloading HuggingFaceH4/MATH-500...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Loaded {len(ds)} problems")

    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "math500",
            "prompt": [
                {
                    "role": "user",
                    "content": row["problem"],
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": [row["answer"]],
            },
            "extra_info": {
                "index": i,
                "subject": row["subject"],
                "level": row["level"],
                "unique_id": row["unique_id"],
            },
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} problems to {output_path}")


if __name__ == "__main__":
    main()
