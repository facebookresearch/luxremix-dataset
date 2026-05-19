#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Reproduce vis_grid.jpg from the released LuxRemix dataset images.

The original vis_grid was generated during rendering from raw Blender outputs
(before background subtraction). Each panel shows the scene lit by a single
light source plus the environment. This script reconstructs those images by
adding the OLAT HDR and background HDR, then tonemapping.

The grid layout matches the original: 2 columns, ordered as
  [background, OLAT_1, OLAT_2, ..., OLAT_N, mixed_reference]
using viewpoint 0001.

Dependencies:
    numpy, matplotlib, opencv-python (for EXR I/O)
    Optional: PyOpenColorIO (for exact Blender AgX tonemapping)

Usage:
    # From a local scene directory:
    python generate_vis_grid.py /path/to/scene/000000

    # Specify output path:
    python generate_vis_grid.py /path/to/scene/000000 -o vis_grid.jpg

    # Use a different viewpoint (default: 1):
    python generate_vis_grid.py /path/to/scene/000000 --viewpoint 2

    # Use approximate tonemapping (no PyOpenColorIO needed):
    python generate_vis_grid.py /path/to/scene/000000 --tonemap approx
"""

import argparse
import glob
import logging
import os
import re
import sys
from collections.abc import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def read_exr(path: str) -> np.ndarray:
    """Read an EXR file and return as float32 HxWxC array."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    log.debug(f"Read EXR {path} shape={img.shape} dtype={img.dtype}")
    # OpenCV reads as BGR, convert to RGB
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def read_ldr(path: str) -> np.ndarray:
    """Read a PNG/JPEG image and return as float32 HxWx3 in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def tonemap_agx_ocio(
    ocio_config: str | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create an AgX tonemapper using PyOpenColorIO (matches Blender exactly)."""
    import PyOpenColorIO as OCIO

    if ocio_config is None:
        ocio_config = os.path.join(
            os.path.dirname(__file__), "colormanagement", "config.ocio"
        )
        if not os.path.exists(ocio_config):
            raise FileNotFoundError(
                f"OCIO config not found at {ocio_config}. Provide --ocio-config or use --tonemap approx."
            )

    config = OCIO.Config.CreateFromFile(ocio_config)  # type: ignore[attr-defined]
    processor = config.getProcessor("Linear", "AgX Base sRGB").getDefaultCPUProcessor()

    def apply(img: np.ndarray) -> np.ndarray:
        out = img.copy().astype(np.float32)
        processor.applyRGB(out)
        return out

    return apply


def tonemap_approx(img: np.ndarray) -> np.ndarray:
    """Approximate AgX tonemapping: Reinhard + gamma.

    This is a reasonable approximation of Blender's AgX Base sRGB curve.
    Not pixel-exact but visually close.
    """
    max_point = 16.0
    x = np.clip(img, 0, None)
    # Extended Reinhard
    mapped = x * (1.0 + x / (max_point**2)) / (1.0 + x)
    # Gamma (AgX applies roughly this)
    gamma = 2.5
    mapped = np.power(np.clip(mapped, 0, 1), 1.0 / gamma)
    return mapped


def find_olat_passes(scene_dir: str, viewpoint: int) -> list[str]:
    """Find all OLAT pass IDs in a scene for a given viewpoint."""
    vp = f"{viewpoint:04d}"
    pattern = os.path.join(scene_dir, f"1*.{vp}.rgb_olat.exr")
    files = sorted(glob.glob(pattern))
    passes = []
    for f in files:
        basename = os.path.basename(f)
        match = re.match(r"(\d{4})\.\d{4}\.rgb_olat\.exr$", basename)
        if match:
            passes.append(match.group(1))
    return passes


def _read_light_types(scene_dir: str, olat_passes: list[str]) -> list[str]:
    """Read light type for each OLAT pass from metadata JSON files."""
    import json

    types: list[str] = []
    for pass_id in olat_passes:
        meta_path = os.path.join(scene_dir, f"{pass_id}.meta.json")
        if not os.path.exists(meta_path):
            return []  # metadata not available, skip labels
        with open(meta_path) as f:
            meta = json.load(f)
        # The active light has "update": true in the lighting list
        active = [lt for lt in meta.get("lighting", []) if lt.get("update")]
        if active:
            types.append(active[0].get("type", ""))
        else:
            types.append("")
    return types


def generate_vis_grid(
    scene_dir: str,
    viewpoint: int = 1,
    tonemap_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    output_path: str | None = None,
):
    """Generate a vis_grid.jpg matching the original rendering pipeline output.

    The grid shows, for a single viewpoint:
      1. Background (all lights off)
      2. Per-light renders (OLAT + background, tonemapped)
      3. Mixed reference (all lights on at reference power)
    """
    if tonemap_fn is None:
        tonemap_fn = tonemap_approx

    vp = f"{viewpoint:04d}"

    # 1. Background LDR (all lights off)
    bg_ldr_path = os.path.join(scene_dir, f"1000.{vp}.rgb_ldr_bg.png")
    bg_hdr_path = os.path.join(scene_dir, f"1000.{vp}.rgb_bg.exr")

    if not os.path.exists(bg_ldr_path):
        raise FileNotFoundError(f"Background LDR not found: {bg_ldr_path}")
    if not os.path.exists(bg_hdr_path):
        raise FileNotFoundError(f"Background HDR not found: {bg_hdr_path}")

    bg_ldr = read_ldr(bg_ldr_path)
    bg_hdr = read_exr(bg_hdr_path)

    # 2. OLAT passes — reconstruct raw render by adding back background
    olat_passes = find_olat_passes(scene_dir, viewpoint)
    if not olat_passes:
        raise FileNotFoundError(f"No OLAT EXR files found for viewpoint {vp}")

    olat_images = []
    for pass_id in olat_passes:
        olat_hdr_path = os.path.join(scene_dir, f"{pass_id}.{vp}.rgb_olat.exr")
        olat_hdr = read_exr(olat_hdr_path)
        # Reconstruct: scene with this single light on = olat + background
        combined_hdr = np.clip(olat_hdr + bg_hdr, 0, None)
        # Tonemap to LDR
        combined_ldr = tonemap_fn(combined_hdr)
        combined_ldr = np.clip(combined_ldr, 0, 1)
        olat_images.append(combined_ldr)

    # 3. Mixed reference LDR (all lights on)
    mix_ldr_path = os.path.join(scene_dir, f"0000.{vp}.rgb_ldr_mix.png")
    if not os.path.exists(mix_ldr_path):
        raise FileNotFoundError(f"Mixed LDR not found: {mix_ldr_path}")
    mix_ldr = read_ldr(mix_ldr_path)

    # Assemble in original order: background, OLAT_1..N, mixed
    vis = [bg_ldr] + olat_images + [mix_ldr]

    # Build labels for each panel
    labels = ["Background (all lights off)"]
    # Try to read light types from metadata
    light_types = _read_light_types(scene_dir, olat_passes)
    for i, pass_id in enumerate(olat_passes):
        light_type = light_types[i] if light_types else ""
        suffix = f" — {light_type}" if light_type else ""
        labels.append(f"OLAT {pass_id}{suffix}")
    labels.append("Mixed (all lights on)")

    # Create matplotlib grid (matching original layout)
    num_images = len(vis)
    ncols = 2
    nrows = (num_images + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = axes.flatten()
    for idx, img in enumerate(vis):
        axes[idx].imshow(img)
        axes[idx].set_title(labels[idx], fontsize=11, pad=4)
        axes[idx].axis("off")
    for idx in range(num_images, nrows * ncols):
        axes[idx].axis("off")
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(scene_dir, "vis_grid.jpg")
    plt.savefig(output_path)
    plt.close(fig)

    log.info(
        f"Saved {output_path} ({num_images} panels: 1 bg + {len(olat_images)} olat + 1 mix)"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce vis_grid.jpg from LuxRemix dataset scene files."
    )
    parser.add_argument(
        "scene_dir",
        help="Path to a scene directory (e.g., /path/to/000000/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: <scene_dir>/vis_grid.jpg)",
    )
    parser.add_argument(
        "--viewpoint",
        type=int,
        default=1,
        help="Viewpoint index (default: 1)",
    )
    parser.add_argument(
        "--tonemap",
        choices=["ocio", "approx"],
        default="ocio",
        help="Tonemapping method: 'ocio' for exact Blender AgX (requires PyOpenColorIO), "
        "'approx' for Reinhard+gamma approximation (default: ocio)",
    )
    parser.add_argument(
        "--ocio-config",
        default=None,
        help="Path to OCIO config.ocio (auto-detected if not specified)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=(
            logging.DEBUG
            if args.verbose
            else os.environ.get("LOGLEVEL", "INFO").upper()
        ),
    )

    if not os.path.isdir(args.scene_dir):
        log.error(f"{args.scene_dir} is not a directory")
        sys.exit(1)

    if args.tonemap == "ocio":
        try:
            tonemap_fn = tonemap_agx_ocio(args.ocio_config)
            log.debug("Using OCIO AgX tonemapping")
        except (ImportError, FileNotFoundError) as e:
            log.warning(f"{e}")
            log.warning("Falling back to approximate tonemapping.")
            tonemap_fn = tonemap_approx
    else:
        log.debug("Using approximate tonemapping")
        tonemap_fn = tonemap_approx

    generate_vis_grid(
        scene_dir=args.scene_dir,
        viewpoint=args.viewpoint,
        tonemap_fn=tonemap_fn,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
