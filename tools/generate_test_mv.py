#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Regenerate the test-mv (multi-view) perspective test set from ERP panoramas.

Reads camera parameters from the released test-mv transforms.json and
per-frame meta.json files, then projects the source ERP panoramas to 512x512
perspective views for 32 viewpoints per scene and generates depth maps,
light mask variants, per-frame metadata, and transforms.json.

Adapted from: LuxRemix_diffusion/test_mvl/ase_test_prepare_mv.py

Usage:
    python generate_test_mv.py --erp-dir /path/to/erp_test_scenes \\
        --camera-dir /path/to/test-mv \\
        --output-dir ./test-mv
"""

import argparse
import glob
import hashlib
import json
import logging
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch
from tqdm import tqdm

from erp_to_perspective import (
    get_cam_matrix,
    get_pers_image,
    latlong_to_cubemap_torch,
    rotate_x,
    rotate_y,
)

# Reuse I/O and light-mask helpers from generate_test_sv
from generate_test_sv import (
    create_all_lgt_rgb_masks,
    find_all_panorama_passes,
    read_panorama_image,
    save_image,
)

log = logging.getLogger(__name__)


def _mask_seed(*parts) -> int:
    # sha256-based seed so two separate Python processes produce identical
    # values; Python's built-in hash() is randomized per-process.
    key = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


# Module-level strategy map (loaded at startup, used by process_mv_frame)
_strategy_map_mv = None


# ---------------------------------------------------------------------------
# Camera parameter reading
# ---------------------------------------------------------------------------


def read_mv_camera_params(scene_camera_dir):
    """Read per-frame camera parameters from a released test-mv scene.

    Tries per-frame ``{mv_id:04d}.meta.json`` first (has azimuth, elevation,
    fov, pano_view_id).  Falls back to ``transforms.json`` + the convention
    that every 8 consecutive frames share an ERP viewpoint.

    Args:
        scene_camera_dir: Path to a test-mv scene directory (e.g., .../014649).

    Returns:
        list of dicts, one per frame, each with:
          ``mv_id``, ``pano_view_id`` (1-indexed str, e.g. "0001"),
          ``azimuth`` (rad), ``elevation`` (rad), ``fov`` (deg).
        Returns ``None`` if no camera metadata is found.
    """
    frames_out = []

    # --- Try per-frame meta.json ---
    meta_files = sorted(glob.glob(os.path.join(scene_camera_dir, "????.meta.json")))
    if meta_files:
        for mf in meta_files:
            with open(mf, "r") as f:
                meta = json.load(f)
            pano_view_id = meta.get("pano_view_id")
            if isinstance(pano_view_id, int):
                pano_view_id = f"{pano_view_id:04d}"
            elif isinstance(pano_view_id, str) and len(pano_view_id) < 4:
                pano_view_id = pano_view_id.zfill(4)
            frames_out.append(
                {
                    "mv_id": meta["mv_id"],
                    "pano_view_id": pano_view_id,
                    "azimuth": meta["azimuth"],
                    "elevation": meta["elevation"],
                    "fov": meta["fov"],
                }
            )
        frames_out.sort(key=lambda x: x["mv_id"])
        return frames_out

    # --- Fall back to transforms.json ---
    transforms_path = os.path.join(scene_camera_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        return None

    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    tf_frames = transforms.get("frames", [])
    if not tf_frames:
        return None

    for idx, frame in enumerate(tf_frames):
        fov = frame.get("fov", 60.0)
        azimuth = frame.get("azimuth", None)
        elevation = frame.get("elevation", None)

        # Convention: every 8 consecutive frames share an ERP viewpoint
        pano_view_idx = idx // 8  # 0-indexed
        pano_view_id = f"{pano_view_idx + 1:04d}"  # 1-indexed, e.g. "0001"

        if azimuth is not None and elevation is not None:
            frames_out.append(
                {
                    "mv_id": idx,
                    "pano_view_id": pano_view_id,
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "fov": fov,
                }
            )
        else:
            # Try to recover from olat_meta.json for this view
            olat_metas = sorted(
                glob.glob(os.path.join(scene_camera_dir, f"{idx:04d}.*.olat_meta.json"))
            )
            if olat_metas:
                with open(olat_metas[0], "r") as f:
                    ometa = json.load(f)
                frames_out.append(
                    {
                        "mv_id": idx,
                        "pano_view_id": pano_view_id,
                        "azimuth": ometa["azimuth"],
                        "elevation": ometa["elevation"],
                        "fov": ometa.get("fov", fov),
                    }
                )
            else:
                log.warning(
                    f"Cannot determine azimuth/elevation for frame {idx}, skipping"
                )

    return frames_out if frames_out else None


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------


def process_mv_frame(
    frame_params,
    erp_scene_path,
    scene_id,
    output_dir,
    cube_resolution=512,
    pers_resolution=(512, 512),
    pano_cache=None,
):
    """Process a single multi-view frame.

    Args:
        frame_params: Dict with mv_id, pano_view_id, azimuth, elevation, fov.
        erp_scene_path: Path to the ERP scene directory.
        scene_id: Scene identifier string.
        output_dir: Root output directory.
        cube_resolution: Cubemap face resolution.
        pers_resolution: (height, width) for perspective projection.
        pano_cache: Optional dict to cache loaded panorama images.

    Returns:
        dict with frame metadata for transforms.json, or None on failure.
    """
    mv_id = frame_params["mv_id"]
    pano_view_id = frame_params["pano_view_id"]
    azimuth = frame_params["azimuth"]
    elevation = frame_params["elevation"]
    fov = np.deg2rad(frame_params["fov"])

    # Local c2w for panorama sampling
    w2c = get_cam_matrix(azimuth, elevation, radius=1)
    c2w = torch.linalg.inv(w2c)

    # Load base metadata (camera poses + lighting)
    base_meta_path = os.path.join(erp_scene_path, "0000.meta.json")
    base_meta = None
    if os.path.exists(base_meta_path):
        with open(base_meta_path, "r") as f:
            base_meta = json.load(f)

    # Compute global c2w using panorama's camera pose
    global_c2w = c2w.numpy()
    if base_meta is not None:
        camera_poses = base_meta.get("camera_pose", [])
        pano_view_idx = int(pano_view_id) - 1  # 1-indexed -> 0-indexed
        if 0 <= pano_view_idx < len(camera_poses):
            camera_pose = camera_poses[pano_view_idx]
            if (
                "transform_matrix" in camera_pose
                and len(camera_pose["transform_matrix"]) > 0
            ):
                pano_c2w = np.array(camera_pose["transform_matrix"][0])
                azim_rot = rotate_y(-azimuth + np.pi / 2).numpy()
                elev_rot = rotate_x(-elevation).numpy()
                anchor_c2w = pano_c2w @ azim_rot
                global_c2w = anchor_c2w @ elev_rot

    # Output directory
    scene_output_dir = os.path.join(output_dir, scene_id)
    os.makedirs(scene_output_dir, exist_ok=True)

    # Discover panorama passes for this viewpoint
    pass_files = find_all_panorama_passes(erp_scene_path, pano_view_id)
    if not pass_files:
        log.warning(f"No panorama passes for scene {scene_id}, view {pano_view_id}")
        return None

    # Load auxiliary masks
    base_pass_id = "0000"
    lgt_src_path = os.path.join(
        erp_scene_path, f"{base_pass_id}.{pano_view_id}.lgt_src.png"
    )
    lgt_obj_path = os.path.join(
        erp_scene_path, f"{base_pass_id}.{pano_view_id}.lgt_obj.png"
    )

    lgt_src_mask = _cached_read(lgt_src_path, pano_cache, normalize=True)
    lgt_obj_mask = _cached_read(lgt_obj_path, pano_cache, normalize=True)

    erp_resolution = (
        lgt_src_mask.shape[:2] if lgt_src_mask is not None else (1024, 2048)
    )

    inp_lighting = base_meta.get("lighting", []) if base_meta is not None else []

    lgt_rgb_generated = set()
    output_files = []

    for pass_id, files in pass_files.items():
        is_olat = pass_id.startswith("1")

        for pano_path, pass_type, ext in files:
            output_basename = f"{mv_id:04d}.{pass_id}.{pass_type}.{ext}"
            output_path = os.path.join(scene_output_dir, output_basename)

            try:
                pano_img = _cached_read(pano_path, pano_cache, normalize=True)
                if pano_img is None:
                    continue
                pano_tensor = torch.from_numpy(pano_img).float()
                cubemap = latlong_to_cubemap_torch(
                    pano_tensor, [cube_resolution, cube_resolution]
                )
                pers_img = get_pers_image(c2w, cubemap, pers_resolution, fov)
                save_image(pers_img.numpy(), output_path)
                output_files.append(output_basename)
            except Exception as e:
                log.error(f"Error processing {pano_path}: {e}")
                continue

        # Process depth map
        depth_path = os.path.join(erp_scene_path, f"{pass_id}.{pano_view_id}.depth.exr")
        if os.path.exists(depth_path):
            try:
                depth_img = _cached_read(depth_path, pano_cache, normalize=False)
                if depth_img is not None:
                    depth_tensor = torch.from_numpy(depth_img).float()
                    depth_cubemap = latlong_to_cubemap_torch(
                        depth_tensor, [cube_resolution, cube_resolution]
                    )
                    depth_pers = get_pers_image(
                        c2w, depth_cubemap, pers_resolution, fov
                    )
                    depth_basename = f"{mv_id:04d}.{pass_id}.depth.exr"
                    depth_output_path = os.path.join(scene_output_dir, depth_basename)
                    save_image(depth_pers.numpy(), depth_output_path)
                    output_files.append(depth_basename)
            except Exception as e:
                log.error(f"Error processing depth {depth_path}: {e}")

        # Light masks for OLAT passes
        if is_olat and pass_id not in lgt_rgb_generated:
            meta_path = os.path.join(erp_scene_path, f"{pass_id}.meta.json")
            if os.path.exists(meta_path) and lgt_src_mask is not None:
                with open(meta_path, "r") as f:
                    pass_meta = json.load(f)

                out_lighting = pass_meta.get("lighting", [])

                # Seed based on scene + view + pass for deterministic mixed_obj masks
                mask_seed = _mask_seed(scene_id, pano_view_id, pass_id)

                # Look up strategy override from mapping
                strat = None
                if _strategy_map_mv is not None and scene_id in _strategy_map_mv:
                    mv_key = f"{mv_id:04d}.{pass_id}"
                    strat = _strategy_map_mv[scene_id].get(mv_key)

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
                    lgt_rgb_basename = f"{mv_id:04d}.{pass_id}.{suffix}.png"
                    lgt_rgb_output_path = os.path.join(
                        scene_output_dir, lgt_rgb_basename
                    )
                    save_image(lgt_rgb_pers.numpy(), lgt_rgb_output_path)
                    output_files.append(lgt_rgb_basename)

                # Per-view per-pass metadata
                olat_metadata = {
                    "pass_id": pass_id,
                    "mv_id": mv_id,
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

                meta_json_basename = f"{mv_id:04d}.{pass_id}.olat_meta.json"
                meta_json_path = os.path.join(scene_output_dir, meta_json_basename)
                with open(meta_json_path, "w") as f:
                    json.dump(olat_metadata, f, indent=2)

                output_files.append(meta_json_basename)
                lgt_rgb_generated.add(pass_id)

    # Per-frame metadata
    frame_meta = {
        "scene_id": scene_id,
        "mv_id": mv_id,
        "pano_view_id": pano_view_id,
        "azimuth": float(azimuth),
        "elevation": float(elevation),
        "fov": float(np.rad2deg(fov)),
        "camera_matrix": {"c2w": global_c2w.tolist()},
        "resolution": {"height": pers_resolution[0], "width": pers_resolution[1]},
    }

    frame_meta_path = os.path.join(scene_output_dir, f"{mv_id:04d}.meta.json")
    with open(frame_meta_path, "w") as f:
        json.dump(frame_meta, f, indent=2)

    # Data for transforms.json
    focal_y = 0.5 * pers_resolution[0] / np.tan(0.5 * fov)
    focal_x = focal_y  # square pixels
    return {
        "file_path": f"{mv_id:04d}.0000.rgb_ldr_mix.png",
        "transform_matrix": global_c2w.tolist(),
        "fl_x": float(focal_x),
        "fl_y": float(focal_y),
        "fov": float(np.rad2deg(fov)),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cached_read(path, cache, normalize=True):
    """Read an image with optional caching.  Returns None if not found."""
    if not os.path.exists(path):
        return None
    if cache is not None and path in cache:
        return cache[path]
    img = read_panorama_image(path, normalize=normalize)
    if cache is not None:
        cache[path] = img
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate test-mv perspective test set from ERP panoramas"
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
        help="Released test-mv directory with transforms.json camera parameters",
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
        help="Output directory for regenerated test-mv",
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

    if not args.camera_dir and not args.camera_params:
        parser.error("one of --camera-dir or --camera-params is required")

    level = (
        logging.DEBUG if args.verbose else os.environ.get("LOGLEVEL", "INFO").upper()
    )
    logging.basicConfig(
        level=level, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    pers_resolution = (args.pers_size, args.pers_size)

    # Load strategy mapping for deterministic mixed_obj masks
    global _strategy_map_mv
    strategy_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "mask_strategies_mv.json"
    )
    if os.path.exists(strategy_path):
        with open(strategy_path, "r") as f:
            _strategy_map_mv = json.load(f)
        log.info(f"Loaded mask strategy mapping ({len(_strategy_map_mv)} scenes)")

    # Build list of (scene_id, frame_params_list)
    scene_jobs = []

    if args.camera_params:
        # Read pre-sampled camera parameters from JSON
        with open(args.camera_params, "r") as f:
            all_params = json.load(f)
        for scene_id, params in sorted(all_params.items()):
            frame_params_list = params["mv"]
            scene_jobs.append((scene_id, frame_params_list))
        log.info(
            f"Loaded camera params for {len(scene_jobs)} scenes from {args.camera_params}"
        )
    else:
        # Discover scene directories in camera-dir
        scene_dirs = sorted(
            d
            for d in os.listdir(args.camera_dir)
            if os.path.isdir(os.path.join(args.camera_dir, d))
        )
        if not scene_dirs:
            log.error(f"No scene directories found in {args.camera_dir}")
            return
        for scene_id in scene_dirs:
            scene_camera_dir = os.path.join(args.camera_dir, scene_id)
            frame_params_list = read_mv_camera_params(scene_camera_dir)
            if not frame_params_list:
                log.warning(f"No camera parameters for scene {scene_id}, skipping")
                continue
            scene_jobs.append((scene_id, frame_params_list))
        log.info(f"Found {len(scene_jobs)} scene directories in {args.camera_dir}")

    for scene_id, frame_params_list in tqdm(scene_jobs, desc="Processing scenes"):
        # Find corresponding ERP scene
        erp_scene_path = os.path.join(args.erp_dir, scene_id)
        if not os.path.isdir(erp_scene_path):
            log.warning(f"ERP scene not found: {erp_scene_path}, skipping")
            continue

        log.debug(f"Processing scene {scene_id} ({len(frame_params_list)} frames)")

        # Per-scene panorama cache
        pano_cache = {}
        transforms_frames = []

        for frame_params in frame_params_list:
            tf_frame = process_mv_frame(
                frame_params,
                erp_scene_path,
                scene_id,
                args.output_dir,
                args.cube_resolution,
                pers_resolution,
                pano_cache,
            )
            if tf_frame is not None:
                transforms_frames.append(tf_frame)

        pano_cache.clear()

        # Write transforms.json
        if transforms_frames:
            scene_output_dir = os.path.join(args.output_dir, scene_id)
            os.makedirs(scene_output_dir, exist_ok=True)
            transforms = {
                "camera_model": "PINHOLE",
                "h": 512,
                "w": 512,
                "cx": 256,
                "cy": 256,
                "frames": transforms_frames,
            }
            transforms_path = os.path.join(scene_output_dir, "transforms.json")
            with open(transforms_path, "w") as f:
                json.dump(transforms, f, indent=2)

    log.info(f"Done. Output written to {args.output_dir}")


if __name__ == "__main__":
    main()
