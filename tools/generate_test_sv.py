#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Regenerate the test-sv (single-view) perspective test set from ERP panoramas.

Reads camera parameters from the released test-sv olat_meta.json files, then
projects the source ERP panoramas to 512x512 perspective views and generates
light mask variants and per-pass metadata.

Adapted from: LuxRemix_diffusion/test_mvl/ase_test_prepare.py

Usage:
    python generate_test_sv.py --erp-dir /path/to/erp_test_scenes \\
        --camera-dir /path/to/test-sv \\
        --output-dir ./test-sv
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import random

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
from tqdm import tqdm

from erp_to_perspective import (
    envmap_vec,
    get_cam_matrix,
    get_pers_image,
    latlong_to_cubemap_torch,
)

log = logging.getLogger(__name__)


def _mask_seed(*parts) -> int:
    # sha256-based seed so two separate Python processes produce identical
    # values; Python's built-in hash() is randomized per-process.
    key = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_panorama_image(img_path, normalize=True):
    """Read a panorama image (supports .exr, .png, .jpg)."""
    if img_path.endswith(".exr"):
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize and not img_path.endswith(".exr"):
        img = img.astype(np.float32) / 255.0
    return img


def save_image(img, save_path):
    """Save an image (supports .exr, .png, .jpg)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith(".exr"):
        img_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_save)
    elif save_path.endswith(".png"):
        img_save = (img * 255).astype(np.uint8)
        img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_save)
    elif save_path.endswith(".jpg"):
        img_save = (img * 255).astype(np.uint8)
        img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_save, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ---------------------------------------------------------------------------
# Light mask generation
# ---------------------------------------------------------------------------


def get_colormap(num_items, cmap_name="turbo"):
    """Get a colormap for the given number of items."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name, num_items)
    cmap = np.array([cmap(i)[:3] for i in range(cmap.N)])
    return cmap


def get_mask_by_color(c_mask, color, diff_thres=0.12):
    """Extract a binary mask by matching against a reference color."""
    color_diff = np.abs(c_mask - color).sum(-1)
    mask = color_diff < diff_thres
    return mask


def pix_to_erp_angle(pix_pos, erp_resolution):
    """Convert a pixel position to ERP azimuth/elevation angles."""
    y, x = pix_pos
    height, width = erp_resolution
    azim = (x / width) * 2 * np.pi - np.pi
    elev = (y / height) * np.pi - np.pi / 2
    return azim, elev


def angle_to_dir(azim, elev):
    """Convert azimuth and elevation to a unit direction vector."""
    sin_elev, cos_elev = np.sin(elev), np.cos(elev)
    sin_azim, cos_azim = np.sin(azim), np.cos(azim)
    return np.array([sin_elev * sin_azim, cos_elev, -sin_elev * cos_azim])


def create_convex_mask(binary_mask):
    """Create a convex-hull mask from a binary mask."""
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    if binary_mask[:, 0].sum() > 0 and binary_mask[:, -1].sum() > 0:
        contours = None
    else:
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    if not contours:
        return np.zeros_like(binary_mask) > 0
    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    convex_mask = np.zeros_like(binary_mask)
    cv2.fillConvexPoly(convex_mask, hull, 255)
    return convex_mask > 0


def create_all_lgt_rgb_masks(
    lgt_src_mask,
    lgt_obj_mask,
    inp_lighting,
    out_lighting,
    erp_resolution,
    updated_light_only=True,
    seed=None,
    strategy_override=None,
):
    """Create all variants of light RGB masks: mixed_obj, src, obj, convex.

    Args:
        lgt_src_mask: Light source mask [H, W, 3].
        lgt_obj_mask: Light object mask [H, W, 3] or None.
        inp_lighting: Input lighting info from base metadata.
        out_lighting: Output lighting info from OLAT metadata.
        erp_resolution: (height, width) of ERP image.
        updated_light_only: If True, only mask lights with ``update=True``.
        seed: If provided, seed a local RNG for deterministic mixed_obj masks.
        strategy_override: If provided, a dict like
            ``{"strategy": "lgt_obj_convex"}`` or
            ``{"strategy": "lgt_src_cone", "cos_threshold": 0.998}``
            to use instead of random selection for the mixed_obj mask.

    Returns:
        dict with keys ``mixed_obj``, ``src``, ``obj``, ``convex`` -> RGB masks.
    """
    rng = random.Random(seed)
    num_lights = len(out_lighting)
    lgt_mask_cmap = get_colormap(num_lights)

    erp_vec = envmap_vec(erp_resolution).flip(1)
    erp_vec = torch.roll(erp_vec, erp_vec.shape[1] // 2, dims=1)

    lgt_rgb_masks = {
        "mixed_obj": np.zeros((*erp_resolution, 3), dtype=np.float32),
        "src": np.zeros((*erp_resolution, 3), dtype=np.float32),
        "obj": np.zeros((*erp_resolution, 3), dtype=np.float32),
        "convex": np.zeros((*erp_resolution, 3), dtype=np.float32),
    }

    for i in range(num_lights):
        light_info = out_lighting[i]

        if updated_light_only and not light_info.get("update", False):
            continue

        lgt_src_mask_i = get_mask_by_color(lgt_src_mask, lgt_mask_cmap[i])

        if lgt_obj_mask is not None:
            lgt_obj_mask_i = get_mask_by_color(lgt_obj_mask, lgt_mask_cmap[i])
        else:
            lgt_obj_mask_i = lgt_src_mask_i

        lgt_vis_i = lgt_src_mask_i.any()
        if not lgt_vis_i and lgt_obj_mask is not None:
            lgt_vis_i = lgt_obj_mask_i.any()

        if not lgt_vis_i:
            continue

        if lgt_src_mask_i.any():
            lgt_center_i = np.array(lgt_src_mask_i.nonzero()).mean(1)
        else:
            lgt_center_i = np.array(lgt_obj_mask_i.nonzero()).mean(1)

        lgt_color = np.array(light_info.get("color", [1.0, 1.0, 1.0]))
        if "chinese_lampion" in light_info["id"]:
            lgt_color = (
                0.8 * np.array([0.9743002, 0.38891026, 0.00775103]) + 0.2 * lgt_color
            )

        lgt_azim, lgt_elev = pix_to_erp_angle(lgt_center_i, erp_resolution)
        lgt_center_vec = torch.from_numpy(
            angle_to_dir(lgt_azim, lgt_elev + np.pi / 2)
        ).float()
        lgt_cos_map = erp_vec @ lgt_center_vec
        lgt_cos_mini_cone_mask_i = (lgt_cos_map > np.cos(np.deg2rad(2))).numpy()

        # 1. src mask
        lgt_src_mask_final = lgt_src_mask_i.copy()
        if lgt_src_mask_final.sum() > 0:
            lgt_rgb_masks["src"][lgt_src_mask_final] = lgt_color

        # 2. obj mask
        lgt_obj_mask_final = lgt_obj_mask_i.copy()
        if "LampFactory" in inp_lighting[i]["id"]:
            lgt_obj_mask_final = lgt_src_mask_i | lgt_obj_mask_i
        lgt_cos_mask_i = lgt_cos_map > np.cos(np.deg2rad(15))
        lgt_obj_mask_final = lgt_obj_mask_final * lgt_cos_mask_i.numpy()
        if lgt_obj_mask_final.sum() < lgt_cos_mini_cone_mask_i.sum():
            lgt_obj_mask_final = lgt_cos_mini_cone_mask_i
        if lgt_obj_mask_final.sum() > 0:
            lgt_rgb_masks["obj"][lgt_obj_mask_final] = lgt_color

        # 3. convex mask
        lgt_convex_mask_final = create_convex_mask(lgt_obj_mask_final)
        lgt_convex_mask_final = lgt_convex_mask_final * lgt_cos_mask_i.numpy()
        if lgt_convex_mask_final.sum() < lgt_cos_mini_cone_mask_i.sum():
            lgt_convex_mask_final = lgt_cos_mini_cone_mask_i
        if lgt_convex_mask_final.sum() > 0:
            lgt_rgb_masks["convex"][lgt_convex_mask_final] = lgt_color

        # 4. mixed_obj mask (random variant selection or override)
        if strategy_override is not None:
            light_mask_type = strategy_override["strategy"]
        else:
            light_mask_type = rng.choices(
                ["lgt_src", "lgt_src_cone", "lgt_obj", "lgt_obj_convex"],
                weights=[0.25, 0.25, 0.1, 0.4],
            )[0]

        lgt_src_mask_cone = lgt_src_mask_i.copy()
        lgt_obj_mask_mixed = lgt_obj_mask_i.copy()

        if light_mask_type.startswith("lgt_obj"):
            if "LampFactory" in inp_lighting[i]["id"]:
                lgt_obj_mask_mixed = lgt_src_mask_i | lgt_obj_mask_i
            lgt_cos_mask_i = lgt_cos_map > np.cos(np.deg2rad(15))
            lgt_obj_mask_mixed = lgt_obj_mask_mixed * lgt_cos_mask_i.numpy()
            if light_mask_type == "lgt_obj_convex":
                lgt_obj_mask_mixed = create_convex_mask(lgt_obj_mask_mixed)
                lgt_obj_mask_mixed = lgt_obj_mask_mixed * lgt_cos_mask_i.numpy()

        if light_mask_type == "lgt_src_cone":
            if strategy_override is not None and "cos_threshold" in strategy_override:
                cos_threshold = strategy_override["cos_threshold"]
                lgt_cos_mask_i = lgt_cos_map > cos_threshold
            else:
                lgt_cos_mask_i = lgt_cos_map > np.cos(np.deg2rad(rng.uniform(2, 3.5)))
            if lgt_cos_mask_i.sum() > lgt_src_mask_cone.sum():
                lgt_src_mask_cone = lgt_cos_mask_i.numpy()

        lgt_mask_mixed = (
            lgt_src_mask_cone
            if light_mask_type.startswith("lgt_src")
            else lgt_obj_mask_mixed
        )
        if lgt_mask_mixed.sum() < lgt_cos_mini_cone_mask_i.sum():
            lgt_mask_mixed = lgt_cos_mini_cone_mask_i

        if lgt_mask_mixed.sum() > 0:
            lgt_rgb_masks["mixed_obj"][lgt_mask_mixed] = lgt_color

    return lgt_rgb_masks


# ---------------------------------------------------------------------------
# Panorama pass discovery
# ---------------------------------------------------------------------------


def find_all_panorama_passes(scene_path, view_id):
    """Find all panorama passes for a given scene and view.

    Returns:
        dict: pass_id -> list of (pano_path, pass_type, ext) tuples.
    """
    pass_files = {}
    pattern = os.path.join(scene_path, f"*.{view_id}.*")
    all_files = glob.glob(pattern)

    for file_path in all_files:
        basename = os.path.basename(file_path)
        parts = basename.split(".")

        if len(parts) < 4:
            continue

        pass_id = parts[0]
        pass_type = ".".join(parts[2:-1])
        ext = parts[-1]

        if ext.lower() not in ["exr", "png", "jpg"]:
            continue

        # Skip auxiliary passes (handled separately)
        if pass_type in [
            "lgt_src",
            "lgt_obj",
            "depth",
            "diffcol",
            "normal",
            "transind",
            "window",
        ]:
            continue

        if pass_id not in pass_files:
            pass_files[pass_id] = []

        pass_files[pass_id].append((file_path, pass_type, ext))

    return pass_files


# ---------------------------------------------------------------------------
# Camera parameter reading
# ---------------------------------------------------------------------------


def read_camera_params(scene_dir):
    """Read camera parameters from the released test-sv metadata.

    Checks olat_meta.json files first, then falls back to transforms.json.

    Args:
        scene_dir: Path to a test-sv scene directory (e.g., .../014649.0003).

    Returns:
        dict with ``azimuth`` (rad), ``elevation`` (rad), ``fov`` (deg),
        or None if no camera metadata found.
    """
    # Try olat_meta.json files (primary source)
    olat_metas = sorted(glob.glob(os.path.join(scene_dir, "*.olat_meta.json")))
    if olat_metas:
        with open(olat_metas[0], "r") as f:
            meta = json.load(f)
        return {
            "azimuth": meta["azimuth"],
            "elevation": meta["elevation"],
            "fov": meta["fov"],
        }

    # Fall back to transforms.json
    transforms_path = os.path.join(scene_dir, "transforms.json")
    if os.path.exists(transforms_path):
        with open(transforms_path, "r") as f:
            transforms = json.load(f)
        frames = transforms.get("frames", [])
        if frames:
            frame = frames[0]
            fov = frame.get("fov", 60.0)
            azimuth = frame.get("azimuth", None)
            elevation = frame.get("elevation", None)
            if azimuth is not None and elevation is not None:
                return {"azimuth": azimuth, "elevation": elevation, "fov": fov}

    return None


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------


def process_single_panorama(pano_path, c2w, fov, cube_resolution, pers_resolution):
    """Process a single ERP panorama to a perspective view."""
    pano_img = read_panorama_image(pano_path, normalize=True)
    pano_tensor = torch.from_numpy(pano_img).float()
    cubemap = latlong_to_cubemap_torch(pano_tensor, [cube_resolution, cube_resolution])
    pers_img = get_pers_image(c2w, cubemap, pers_resolution, fov)
    return pers_img.numpy()


def process_scene_view(
    scene_path,
    view_id,
    camera_params,
    output_dir,
    cube_resolution=512,
    pers_resolution=(512, 512),
):
    """Process all passes for a given scene and view.

    Args:
        scene_path: Path to the ERP scene directory (e.g., /erp/014649).
        view_id: View ID string (e.g., '0003').
        camera_params: Dict with 'azimuth' (rad), 'elevation' (rad), 'fov' (deg).
        output_dir: Root output directory.
        cube_resolution: Resolution for cubemap conversion.
        pers_resolution: (height, width) for perspective projection.
    """
    scene_id = os.path.basename(scene_path)

    pass_files = find_all_panorama_passes(scene_path, view_id)
    if not pass_files:
        log.warning(f"No panorama passes found for scene {scene_id}, view {view_id}")
        return

    # Load auxiliary masks from base pass
    base_pass_id = "0000"
    lgt_src_path = os.path.join(scene_path, f"{base_pass_id}.{view_id}.lgt_src.png")
    lgt_obj_path = os.path.join(scene_path, f"{base_pass_id}.{view_id}.lgt_obj.png")

    lgt_src_mask = None
    lgt_obj_mask = None
    if os.path.exists(lgt_src_path):
        lgt_src_mask = read_panorama_image(lgt_src_path, normalize=True)
    if os.path.exists(lgt_obj_path):
        lgt_obj_mask = read_panorama_image(lgt_obj_path, normalize=True)

    erp_resolution = (
        lgt_src_mask.shape[:2] if lgt_src_mask is not None else (1024, 2048)
    )

    # Camera setup
    azimuth = camera_params["azimuth"]
    elevation = camera_params["elevation"]
    fov = np.deg2rad(camera_params["fov"])

    w2c = get_cam_matrix(azimuth, elevation, radius=1)
    c2w = torch.linalg.inv(w2c)

    # Output directory: {scene_id}.{view_id}
    scene_view_dir = f"{scene_id}.{view_id}"
    scene_output_dir = os.path.join(output_dir, scene_view_dir)
    os.makedirs(scene_output_dir, exist_ok=True)

    # Load base metadata for inp_lighting (needed for light masks)
    base_meta_path = os.path.join(scene_path, f"{base_pass_id}.meta.json")
    inp_lighting = []
    if os.path.exists(base_meta_path):
        with open(base_meta_path, "r") as f:
            base_meta = json.load(f)
        inp_lighting = base_meta.get("lighting", [])

    lgt_rgb_generated = set()

    for pass_id, files in pass_files.items():
        is_olat = pass_id.startswith("1")

        for pano_path, pass_type, ext in files:
            # Output: {pass_id}.{pass_type}.{ext} (no view_id)
            new_basename = f"{pass_id}.{pass_type}.{ext}"
            output_path = os.path.join(scene_output_dir, new_basename)

            try:
                pers_img_np = process_single_panorama(
                    pano_path, c2w, fov, cube_resolution, pers_resolution
                )
                save_image(pers_img_np, output_path)
            except Exception as e:
                log.error(f"Error processing {pano_path}: {e}")
                continue

        # Generate light masks for OLAT passes
        if is_olat and pass_id not in lgt_rgb_generated:
            meta_path = os.path.join(scene_path, f"{pass_id}.meta.json")
            if os.path.exists(meta_path) and lgt_src_mask is not None:
                with open(meta_path, "r") as f:
                    pass_meta = json.load(f)

                out_lighting = pass_meta.get("lighting", [])

                # Seed based on scene + pass for deterministic mixed_obj masks
                mask_seed = _mask_seed(scene_id, view_id, pass_id)

                # Look up strategy override from mapping
                sv_key = f"{scene_id}.{view_id}"
                strat = None
                if _strategy_map_sv is not None and sv_key in _strategy_map_sv:
                    strat = _strategy_map_sv[sv_key].get(pass_id)

                lgt_rgb_masks = create_all_lgt_rgb_masks(
                    lgt_src_mask,
                    lgt_obj_mask,
                    inp_lighting,
                    out_lighting,
                    erp_resolution,
                    updated_light_only=True,
                    seed=mask_seed,
                    strategy_override=strat,
                )

                mask_suffixes = {
                    "mixed_obj": "lgt_rgb_olat",
                    "src": "lgt_rgb_src_olat",
                    "obj": "lgt_rgb_obj_olat",
                    "convex": "lgt_rgb_convex_olat",
                }

                for mask_key, suffix in mask_suffixes.items():
                    lgt_rgb_mask = lgt_rgb_masks[mask_key]
                    lgt_rgb_tensor = torch.from_numpy(lgt_rgb_mask).float()
                    lgt_rgb_cubemap = latlong_to_cubemap_torch(
                        lgt_rgb_tensor,
                        [cube_resolution, cube_resolution],
                        mode="nearest",
                    )
                    lgt_rgb_pers = get_pers_image(
                        c2w, lgt_rgb_cubemap, pers_resolution, fov, mode="nearest"
                    )
                    lgt_rgb_pers_np = lgt_rgb_pers.numpy()
                    lgt_rgb_basename = f"{pass_id}.{suffix}.png"
                    lgt_rgb_output_path = os.path.join(
                        scene_output_dir, lgt_rgb_basename
                    )
                    save_image(lgt_rgb_pers_np, lgt_rgb_output_path)

                # Per-pass metadata
                olat_metadata = {
                    "pass_id": pass_id,
                    "azimuth": float(azimuth),
                    "elevation": float(elevation),
                    "fov": float(np.rad2deg(fov)),
                    "lights": [],
                }

                for i, light_info in enumerate(out_lighting):
                    if light_info.get("update", False):
                        olat_metadata["lights"].append(
                            {
                                "light_index": i,
                                "lgt_type": light_info.get("type", "Unknown"),
                                "lgt_power": float(light_info.get("power", 0.0)),
                                "lgt_color": [
                                    float(c)
                                    for c in light_info.get("color", [1.0, 1.0, 1.0])
                                ],
                            }
                        )

                meta_json_path = os.path.join(
                    scene_output_dir, f"{pass_id}.olat_meta.json"
                )
                with open(meta_json_path, "w") as f:
                    json.dump(olat_metadata, f, indent=2)

                lgt_rgb_generated.add(pass_id)


# Module-level strategy map (loaded at startup, used by process_scene_view)
_strategy_map_sv = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate test-sv perspective test set from ERP panoramas"
    )
    parser.add_argument(
        "--erp-dir",
        type=str,
        required=True,
        help="Root directory containing ERP test scenes (one sub-dir per scene ID)",
    )
    parser.add_argument(
        "--camera-dir",
        type=str,
        default=None,
        help="Released test-sv directory with olat_meta.json camera parameters",
    )
    parser.add_argument(
        "--camera-params",
        type=str,
        default=None,
        help="JSON file with pre-sampled camera parameters (alternative to --camera-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for regenerated test-sv",
    )
    parser.add_argument(
        "--cube-resolution",
        type=int,
        default=512,
        help="Cubemap face resolution (default: 512)",
    )
    parser.add_argument(
        "--pers-size",
        type=int,
        default=512,
        help="Perspective image size (default: 512, produces 512x512)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    level = (
        logging.DEBUG if args.verbose else os.environ.get("LOGLEVEL", "INFO").upper()
    )
    logging.basicConfig(
        level=level, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    if not args.camera_dir and not args.camera_params:
        parser.error("one of --camera-dir or --camera-params is required")

    pers_resolution = (args.pers_size, args.pers_size)

    # Load strategy mapping for deterministic mixed_obj masks
    global _strategy_map_sv
    strategy_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "mask_strategies_sv.json"
    )
    if os.path.exists(strategy_path):
        with open(strategy_path, "r") as f:
            _strategy_map_sv = json.load(f)
        log.info(f"Loaded mask strategy mapping ({len(_strategy_map_sv)} scenes)")

    # Build list of (scene_id, view_id, camera_params)
    scene_jobs = []

    if args.camera_params:
        # Read pre-sampled camera parameters from JSON
        with open(args.camera_params, "r") as f:
            all_params = json.load(f)
        for scene_id, params in sorted(all_params.items()):
            sv = params["sv"]
            camera_params = {
                "azimuth": sv["azimuth"],
                "elevation": sv["elevation"],
                "fov": sv["fov"],
            }
            view_id = sv["view_id"]
            scene_jobs.append((scene_id, view_id, camera_params))
        log.info(
            f"Loaded camera params for {len(scene_jobs)} scenes from {args.camera_params}"
        )
    else:
        # Discover scene directories in camera-dir (format: {scene_id}.{viewpoint})
        scene_dirs = sorted(
            d
            for d in os.listdir(args.camera_dir)
            if os.path.isdir(os.path.join(args.camera_dir, d)) and "." in d
        )
        if not scene_dirs:
            log.error(f"No scene directories found in {args.camera_dir}")
            return
        for scene_view_name in scene_dirs:
            parts = scene_view_name.rsplit(".", 1)
            if len(parts) != 2:
                log.warning(
                    f"Skipping directory with unexpected name: {scene_view_name}"
                )
                continue
            scene_id, view_id = parts
            camera_dir_path = os.path.join(args.camera_dir, scene_view_name)
            camera_params = read_camera_params(camera_dir_path)
            if camera_params is None:
                log.warning(
                    f"No camera parameters found for {scene_view_name}, skipping"
                )
                continue
            scene_jobs.append((scene_id, view_id, camera_params))
        log.info(f"Found {len(scene_jobs)} scene directories in {args.camera_dir}")

    for scene_id, view_id, camera_params in tqdm(scene_jobs, desc="Processing scenes"):
        # Find corresponding ERP scene
        erp_scene_path = os.path.join(args.erp_dir, scene_id)
        if not os.path.isdir(erp_scene_path):
            log.warning(f"ERP scene not found: {erp_scene_path}, skipping")
            continue

        log.debug(
            f"Processing {scene_id}.{view_id} (az={camera_params['azimuth']:.3f}, "
            f"el={camera_params['elevation']:.3f}, fov={camera_params['fov']:.1f}°)"
        )

        process_scene_view(
            erp_scene_path,
            view_id,
            camera_params,
            args.output_dir,
            args.cube_resolution,
            pers_resolution,
        )

    log.info(f"Done. Output written to {args.output_dir}")


if __name__ == "__main__":
    main()
