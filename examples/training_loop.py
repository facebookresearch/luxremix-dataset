#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Minimal training-loop demo using LuxRemixIterableDataset.

Iterates 10 batches over a directory of unpacked scenes, prints the shape
of each tensor in each batch, and exits. No model, no optimizer - this is
just a smoke test that the dataloader works end-to-end with multiple
worker processes.

Usage:
    python examples/training_loop.py /path/to/unpacked/scenes/

The argument should be a directory containing one folder per scene
(e.g. ``./luxremix/000000/``, ``./luxremix/000001/``, ...), as produced
by ``download.py --unpack``.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import LuxRemixIterableDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument(
        "--with-depth",
        action="store_true",
        help="Include the per-viewpoint depth.exr in each sample",
    )
    args = parser.parse_args()

    dataset = LuxRemixIterableDataset(
        args.scenes_dir,
        infinite=True,
        with_depth=args.with_depth,
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(
        f"Iterating {args.num_batches} batches "
        f"(batch_size={args.batch_size}, num_workers={args.num_workers})"
    )
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        if i == 0:
            print("\nBatch tensors (shape, dtype):")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k:18s} {tuple(v.shape)}  {v.dtype}")
                else:
                    print(f"  {k:18s} (list)  example={v[0]!r}")
            print()
        # In a real training loop you would: do perspective projection
        # (see tools/erp_to_perspective.py), apply tonemapping, run forward+loss,
        # backward, optimizer.step(). Here we just count batches.
        print(
            f"  batch {i + 1}/{args.num_batches}: "
            f"scene_ids={list(batch['scene_id'])}, "
            f"pass_ids={batch['pass_id'].tolist()}, "
            f"view_ids={batch['view_id'].tolist()}"
        )

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed / args.num_batches:.2f}s per batch)")


if __name__ == "__main__":
    main()
