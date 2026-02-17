#!/usr/bin/env python3
"""Minimal GPU training smoke test for Archer image validation."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal training smoke test on GPU.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="archer-image-smoke")
    parser.add_argument("--wandb-run-name", type=str, default="archer-image-smoke")
    parser.add_argument(
        "--output-path",
        type=str,
        default="output/archer_image_smoke/model.pt",
        help="Checkpoint path relative to repository root or absolute path.",
    )
    return parser.parse_args()


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_wandb(args: argparse.Namespace):
    use_online = bool(os.environ.get("WANDB_API_KEY"))
    mode = os.environ.get("WANDB_MODE", "online" if use_online else "offline")
    try:
        import wandb  # pylint: disable=import-outside-toplevel

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=mode,
            config=vars(args),
        )
        return wandb, run, mode
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] wandb init failed, continue without wandb logging: {exc}")
        return None, None, "disabled"


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This smoke test requires a GPU.")

    device = torch.device("cuda")
    model = TinyMLP(args.input_dim, args.hidden_dim, args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    wandb_mod, wandb_run, wandb_mode = init_wandb(args)

    print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] torch: {torch.__version__}")
    print(f"[INFO] wandb mode: {wandb_mode}")
    print(f"[INFO] steps: {args.steps}, batch_size: {args.batch_size}")

    start = time.time()
    for step in range(1, args.steps + 1):
        inputs = torch.randn(args.batch_size, args.input_dim, device=device)
        targets = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        metrics = {
            "train/loss": float(loss.item()),
            "train/step": step,
            "train/elapsed_sec": time.time() - start,
        }
        if wandb_mod is not None:
            wandb_mod.log(metrics)
        if step % 5 == 0 or step == args.steps:
            print(f"[STEP {step:03d}] loss={metrics['train/loss']:.6f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "steps": args.steps,
            "seed": args.seed,
            "torch_version": torch.__version__,
        },
        output_path,
    )
    print(f"[OK] Smoke training finished. Checkpoint saved to: {output_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
