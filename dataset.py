# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Reference PyTorch IterableDataset for the LuxRemix dataset.

This module provides two pieces:

  * ``Scene`` - a pure-Python reader for one unpacked LuxRemix scene
    folder. No PyTorch dependency. Use this when you want to iterate
    files manually or build your own pipeline.

  * ``LuxRemixIterableDataset`` - a ``torch.utils.data.IterableDataset``
    that yields per-sample dicts (HDR OLAT target, HDR mix input, light
    mask + power + color metadata, optional depth). Suitable as a
    drop-in for training loops that do their own perspective projection
    and tonemapping. See ``examples/training_loop.py`` for a starter.

Both pieces are **robust to variable viewpoint counts**: most scenes have
4 viewpoints, but 114 scenes have 3 and one scene has 1. The reader
infers the view set per scene from filenames rather than hard-coding 4
views. See ``docs/DATACARD.md`` for context.

The internal training loader used by the LuxRemix paper
(``LuxRemix_diffusion/src/data/mvl_ase_pano_dataset.py``, 1,745 lines)
is significantly more elaborate - it implements multi-view sampling,
OLAT recomposition with per-light color jitter, multi-level buffer
queues, task-conditioning masks, and a number of dataset variants. This
public reference covers the common single-frame OLAT-in / mix-out
workflow that most consumers will want to start from. Extend as needed.
"""

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Iterator

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

# Filename grammar — see FORMAT.md for the spec.
SCENE_ID_RE = re.compile(r"^\d{6}$")
IMAGE_NAME_RE = re.compile(
    r"^(?P<pass>\d{4})\.(?P<view>\d{4})\.(?P<type>[a-z_]+\.(?:exr|png|jpg))$"
)
META_NAME_RE = re.compile(r"^(?P<pass>\d{4})\.meta\.json$")

# Pass-id boundaries.
BG_PASS_ID = 1000
MIX_PASS_IDS = range(0, 4)  # 0000..0003 inclusive
OLAT_PASS_MIN = 1001

# OLAT-pass scaling levels used in the LuxRemix training loop.
DEFAULT_OLAT_SCALES = (0.1, 0.36, 1.0)


# --------------------------------------------------------------------------- #
# Scene reader                                                                #
# --------------------------------------------------------------------------- #


class Scene:
    """Reader for one unpacked LuxRemix scene folder.

    Indexes the folder contents on construction, then exposes:
      * ``viewpoints`` - the view ids actually present (1..4)
      * ``olat_passes`` - the OLAT pass ids (1001+)
      * ``mix_passes`` - the mixed-light pass ids (0000..0003)
      * ``meta(pass_id)`` - parsed per-pass meta.json
      * ``read_*`` - typed image loaders that return ``np.ndarray``s

    All image loads use OpenCV with ``IMREAD_ANYCOLOR | IMREAD_ANYDEPTH``
    for EXR files; consumers should expect half-float EXRs (``rgb_mix``,
    ``rgb_bg``) to come back as ``np.float32`` (cv2 auto-promotes).
    """

    def __init__(self, scene_dir: str | Path):
        self.dir = Path(scene_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"scene dir not found: {self.dir}")

        # (pass_id, view_id, type) -> filename
        self._files: dict[tuple[int, int, str], str] = {}
        self._meta_files: dict[int, str] = {}
        for name in sorted(os.listdir(self.dir)):
            m = IMAGE_NAME_RE.match(name)
            if m:
                self._files[
                    (int(m.group("pass")), int(m.group("view")), m.group("type"))
                ] = name
                continue
            m = META_NAME_RE.match(name)
            if m:
                self._meta_files[int(m.group("pass"))] = name

        self.scene_id = self.dir.name

        all_views: set[int] = {vid for _, vid, _ in self._files}
        self.viewpoints: list[int] = sorted(all_views)

        all_passes: set[int] = {pid for pid, _, _ in self._files} | set(
            self._meta_files
        )
        self.mix_passes: list[int] = sorted(p for p in all_passes if p in MIX_PASS_IDS)
        self.bg_pass: int | None = BG_PASS_ID if BG_PASS_ID in all_passes else None
        self.olat_passes: list[int] = sorted(
            p for p in all_passes if p >= OLAT_PASS_MIN
        )

    # ----- low-level lookup -----

    def _file(self, pass_id: int, view_id: int, type_: str) -> Path:
        try:
            return self.dir / self._files[(pass_id, view_id, type_)]
        except KeyError:
            raise FileNotFoundError(
                f"scene {self.scene_id}: no file for pass={pass_id:04d} "
                f"view={view_id:04d} type={type_}"
            )

    def has(self, pass_id: int, view_id: int, type_: str) -> bool:
        return (pass_id, view_id, type_) in self._files

    def meta(self, pass_id: int) -> dict:
        if pass_id not in self._meta_files:
            raise FileNotFoundError(
                f"scene {self.scene_id}: no meta.json for pass={pass_id:04d}"
            )
        with open(self.dir / self._meta_files[pass_id]) as f:
            return json.load(f)

    # ----- typed loaders -----

    @staticmethod
    def _read_exr(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise OSError(f"cv2.imread returned None for {path}")
        return img

    @staticmethod
    def _read_png(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise OSError(f"cv2.imread returned None for {path}")
        return img

    def read_rgb_olat(self, pass_id: int, view_id: int) -> np.ndarray:
        """HDR OLAT contribution (3-channel float32, BG-subtracted)."""
        return self._read_exr(self._file(pass_id, view_id, "rgb_olat.exr"))

    def read_rgb_mix(self, pass_id: int, view_id: int) -> np.ndarray:
        """HDR mixed-light render (4-channel half-float; cv2 returns float32)."""
        return self._read_exr(self._file(pass_id, view_id, "rgb_mix.exr"))

    def read_rgb_bg(self, view_id: int) -> np.ndarray:
        """HDR background pass (all lights OFF)."""
        if self.bg_pass is None:
            raise FileNotFoundError(f"scene {self.scene_id}: no background pass")
        return self._read_exr(self._file(self.bg_pass, view_id, "rgb_bg.exr"))

    def read_depth(self, view_id: int) -> np.ndarray:
        """Per-viewpoint ray depth (1-channel float32, from pass 0000)."""
        return self._read_exr(self._file(0, view_id, "depth.exr"))

    def read_lgt_src(self, view_id: int) -> np.ndarray:
        """Per-light source-emitter mask (color-coded RGB, pass 0000)."""
        return self._read_png(self._file(0, view_id, "lgt_src.png"))

    def read_lgt_obj(self, view_id: int) -> np.ndarray:
        """Per-light fixture mask (color-coded RGB, pass 0000)."""
        return self._read_png(self._file(0, view_id, "lgt_obj.png"))

    def read_normal(self, view_id: int) -> np.ndarray:
        """Surface normals (RGB, pass 0000)."""
        return self._read_png(self._file(0, view_id, "normal.png"))

    def active_light_meta(self, pass_id: int) -> dict | None:
        """Return the lighting[] entry whose ``update`` is True for this pass.

        OLAT passes activate exactly one light; mix passes activate none and
        return None.
        """
        m = self.meta(pass_id)
        for entry in m.get("lighting", []):
            if entry.get("update"):
                return entry
        return None


# --------------------------------------------------------------------------- #
# PyTorch IterableDataset                                                     #
# --------------------------------------------------------------------------- #


def _to_chw_tensor(arr: np.ndarray) -> torch.Tensor:
    """HxWxC (or HxW) numpy → contiguous CxHxW torch tensor.

    Returns a contiguous tensor so downstream ``pin_memory`` / first GPU op
    doesn't pay a hidden copy from a non-contiguous transpose view. Dtype
    is preserved.
    """
    if arr.ndim == 2:
        arr = arr[None, ...]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(arr))


def discover_scenes(scenes_dir: str | Path) -> list[Path]:
    """Find every subfolder of ``scenes_dir`` that looks like a LuxRemix scene."""
    root = Path(scenes_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"scenes_dir does not exist: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir() and SCENE_ID_RE.match(p.name))


class LuxRemixIterableDataset(IterableDataset):
    """Yields per-sample dicts suitable for OLAT-in / mix-out training.

    Each sample is one (scene, viewpoint, OLAT pass) tuple. By default
    the dataset iterates scenes in random order, picks a random
    viewpoint from each scene's *actual* view set, and a random OLAT
    pass. With ``infinite=True`` it loops forever; otherwise it makes a
    single pass over all scenes.

    Each yielded dict has:

      ``scene_id``       (str)            scene folder name, e.g. "000000"
      ``pass_id``        (int)            OLAT pass id (1001+)
      ``view_id``        (int)            viewpoint id
      ``rgb_olat``       (3, H, W) float  OLAT HDR (scaled by sampled level)
      ``rgb_mix``        (4, H, W) float  reference mixed-light HDR (pass 0000)
      ``rgb_bg``         (4, H, W) float  background HDR (pass 1000)
      ``lgt_src_mask``   (3, H, W) uint8  per-light source-emitter mask
      ``lgt_obj_mask``   (3, H, W) uint8  per-light fixture mask
      ``depth``          (1, H, W) float  ray depth (only if ``with_depth=True``)
      ``light_power``    (scalar) float   sampled scale * meta power
      ``light_color``    (3,) float       light color (RGB, [0, 1])
      ``light_type``     (str)            "Ceiling", "Floor", "Desk", ...

    All tensors are returned as ``torch.Tensor``. Multi-worker iteration is
    supported: each worker gets a disjoint slice of the scene list.
    """

    def __init__(
        self,
        scenes_dir: str | Path,
        *,
        infinite: bool = True,
        olat_scales: tuple[float, ...] = DEFAULT_OLAT_SCALES,
        with_depth: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        self.scenes = discover_scenes(scenes_dir)
        if not self.scenes:
            raise RuntimeError(f"no scenes found under {scenes_dir}")
        self.infinite = infinite
        self.olat_scales = tuple(olat_scales)
        self.with_depth = with_depth
        self.base_seed = seed
        # Per-worker Scene cache so we only listdir + filename-parse each
        # scene once across the worker's whole lifetime.
        self._scene_cache: dict[Path, Scene] = {}

    # ----- worker-aware slicing -----

    def _worker_scenes(self) -> list[Path]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return list(self.scenes)
        return self.scenes[info.id :: info.num_workers]

    def _make_rng(self) -> random.Random:
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        if self.base_seed is None:
            return random.Random()
        return random.Random(self.base_seed * 1_000_003 + worker_id)

    # ----- main iteration -----

    def __iter__(self) -> Iterator[dict]:
        rng = self._make_rng()
        scenes = self._worker_scenes()

        while True:
            order = list(scenes)
            rng.shuffle(order)
            for scene_dir in order:
                sample = self._try_load(scene_dir, rng)
                if sample is not None:
                    yield sample
            if not self.infinite:
                return

    def _get_scene(self, scene_dir: Path) -> Scene | None:
        cached = self._scene_cache.get(scene_dir)
        if cached is not None:
            return cached
        try:
            scene = Scene(scene_dir)
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"skipping {scene_dir}: {e}")
            return None
        self._scene_cache[scene_dir] = scene
        return scene

    def _try_load(self, scene_dir: Path, rng: random.Random) -> dict | None:
        scene = self._get_scene(scene_dir)
        if scene is None:
            return None
        if not scene.olat_passes or not scene.viewpoints or scene.bg_pass is None:
            logger.warning(f"skipping {scene_dir}: incomplete pass set")
            return None

        view = rng.choice(scene.viewpoints)
        olat_pass = rng.choice(scene.olat_passes)
        try:
            # Pass 0000 is the full-power reference mix; always present.
            mix = scene.read_rgb_mix(0, view)
            olat = scene.read_rgb_olat(olat_pass, view)
            bg = scene.read_rgb_bg(view)
            lgt_src = scene.read_lgt_src(view)
            lgt_obj = scene.read_lgt_obj(view)
            depth = scene.read_depth(view) if self.with_depth else None
            light = scene.active_light_meta(olat_pass) or {}
        except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
            logger.warning(f"skipping {scene_dir}: read failed: {e}")
            return None

        scale = rng.choice(self.olat_scales)
        # olat is already float32 from cv2 EXR auto-promotion; scalar multiply
        # by a Python float returns float32 — no extra astype copies.
        olat_scaled = olat * np.float32(scale)

        sample: dict = {
            "scene_id": scene.scene_id,
            "pass_id": olat_pass,
            "view_id": view,
            "rgb_olat": _to_chw_tensor(olat_scaled),
            "rgb_mix": _to_chw_tensor(mix),
            "rgb_bg": _to_chw_tensor(bg),
            "lgt_src_mask": _to_chw_tensor(lgt_src),
            "lgt_obj_mask": _to_chw_tensor(lgt_obj),
            "light_power": torch.tensor(
                float(light.get("power", 1.0)) * scale, dtype=torch.float32
            ),
            "light_color": torch.tensor(
                light.get("color", [1.0, 1.0, 1.0]), dtype=torch.float32
            ),
            "light_type": str(light.get("type", "unknown")),
        }
        if depth is not None:
            sample["depth"] = _to_chw_tensor(depth)
        return sample
