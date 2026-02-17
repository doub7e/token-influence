#!/usr/bin/env python3
"""Download a Hugging Face model snapshot to a local directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model snapshot from Hugging Face.")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--local-dir", type=str, required=True)
    parser.add_argument(
        "--required-pattern",
        type=str,
        default="*.safetensors",
        help="If any file matching this glob exists under local-dir, skip download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if any(out_dir.glob(args.required_pattern)):
        print(f"[INFO] model weights found at {out_dir} ({args.required_pattern}), skip download")
        return

    print(f"[INFO] downloading {args.repo_id} -> {out_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[OK] downloaded {args.repo_id} to {out_dir}")


if __name__ == "__main__":
    main()
