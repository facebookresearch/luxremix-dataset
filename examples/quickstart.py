#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Print summary stats for one LuxRemix scene and write a vis_grid preview.

Usage:
    python examples/quickstart.py /path/to/unpacked/scene/000000/
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import Scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene_dir",
        type=Path,
        help="Path to an unpacked LuxRemix scene folder (e.g. ./luxremix/000000/)",
    )
    parser.add_argument(
        "--vis-grid",
        type=Path,
        default=None,
        help=(
            "Optional output path for a vis_grid.jpg overview. Requires "
            "tools/generate_vis_grid.py to be importable from the repo root."
        ),
    )
    args = parser.parse_args()

    scene = Scene(args.scene_dir)

    print(f"Scene id:      {scene.scene_id}")
    print(f"Viewpoints:    {scene.viewpoints}  ({len(scene.viewpoints)} view(s))")
    print(f"OLAT passes:   {scene.olat_passes}  ({len(scene.olat_passes)} pass(es))")
    print(f"Mix passes:    {scene.mix_passes}")
    if scene.bg_pass is not None:
        print(f"Background:    pass {scene.bg_pass}")
    else:
        print("Background:    MISSING")

    view = scene.viewpoints[0]
    print(f"\nLoading view {view:04d} as a smoke test...")
    olat = scene.read_rgb_olat(scene.olat_passes[0], view)
    mix = scene.read_rgb_mix(0, view)
    bg = scene.read_rgb_bg(view)
    depth = scene.read_depth(view)
    lgt_src = scene.read_lgt_src(view)
    print(
        f"  rgb_olat:    shape={olat.shape}  dtype={olat.dtype}  range=[{olat.min():.3f}, {olat.max():.3f}]"
    )
    print(
        f"  rgb_mix:     shape={mix.shape}  dtype={mix.dtype}  range=[{mix.min():.3f}, {mix.max():.3f}]"
    )
    print(f"  rgb_bg:      shape={bg.shape}  dtype={bg.dtype}")
    print(
        f"  depth:       shape={depth.shape}  dtype={depth.dtype}  range=[{depth.min():.3f}, {depth.max():.3f}]"
    )
    print(f"  lgt_src:     shape={lgt_src.shape}  dtype={lgt_src.dtype}")

    print(f"\nActive light for pass {scene.olat_passes[0]:04d}:")
    light = scene.active_light_meta(scene.olat_passes[0]) or {}
    for k in ("id", "type", "power", "color", "position"):
        if k in light:
            print(f"  {k:10s} {light[k]}")

    lighting = scene.meta(0).get("lighting", [])
    print(f"\nLights in this scene ({len(lighting)} total):")
    by_type = Counter(entry.get("type", "unknown") for entry in lighting)
    for t, n in sorted(by_type.items()):
        print(f"  {t:20s} {n}")

    if args.vis_grid is not None:
        import subprocess

        repo_root = Path(__file__).resolve().parent.parent
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "generate_vis_grid.py"),
            str(args.scene_dir),
            "--output",
            str(args.vis_grid),
        ]
        print(f"\nRendering vis grid: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Wrote {args.vis_grid}")


if __name__ == "__main__":
    main()
