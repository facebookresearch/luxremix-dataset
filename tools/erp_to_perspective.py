# Copyright (c) Meta Platforms, Inc. and affiliates.
"""ERP to Perspective Projection Utilities

Self-contained module for projecting equirectangular panorama (ERP) images
to pinhole perspective views via intermediate cubemap representation.

Extracted from:
  - LuxRemix_diffusion/src/data/rendering_utils.py
  - LuxRemix_diffusion/test_mvl/ase_test_prepare.py

Dependencies: torch, numpy (no nvdiffrast, no cv2).
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


# ---------------------------------------------------------------------------
# Cubemap operations
# ---------------------------------------------------------------------------


def cube_to_dir(s, x, y):
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    else:  # s == 5
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap_torch(latlong_map, res, mode="bilinear"):
    """Convert a lat-long (equirectangular) map to a cubemap.

    Args:
        latlong_map: [H, W, C] or [B, H, W, C] tensor.
        res: [face_h, face_w] resolution of each cubemap face.
        mode: Interpolation mode ("bilinear" or "nearest").

    Returns:
        Cubemap tensor [6, res[0], res[1], C] or [B, 6, res[0], res[1], C].
    """
    ndim = latlong_map.ndim
    batch_size = 1 if ndim == 3 else latlong_map.shape[0]
    if ndim == 3:
        latlong_map = latlong_map.unsqueeze(0)
    device = latlong_map.device
    cubemap = torch.zeros(
        batch_size,
        6,
        res[0],
        res[1],
        latlong_map.shape[-1],
        dtype=torch.float32,
        device=device,
    )

    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(
                -1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device
            ),
            torch.linspace(
                -1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device
            ),
            indexing="ij",
        )
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi

        tu = tu * 2 - 1
        tv = tv * 2 - 1

        grid = torch.cat((tu, tv), dim=-1)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        texture = latlong_map.permute(0, 3, 1, 2)

        sampled = F.grid_sample(
            texture,
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=False,
        )

        cubemap[:, s] = sampled.permute(0, 2, 3, 1)

    if ndim == 3:
        cubemap = cubemap.squeeze(0)

    return cubemap


def cubemap_sample_torch(cubemap, dirs, mode="bilinear"):
    """Sample from a cubemap given direction vectors.

    Args:
        cubemap: [6, H, W, C] or [B, 6, H, W, C] tensor.
        dirs: [N, 3] direction vectors.
        mode: Interpolation mode ("bilinear" or "nearest").

    Returns:
        Sampled values [N, C] or [B, N, C].
    """
    device = cubemap.device
    N = dirs.shape[0]
    C = cubemap.shape[-1]
    h, w = cubemap.shape[-3:-1]
    ndim = cubemap.ndim
    batch_size = cubemap.shape[0] if ndim == 5 else 1
    if ndim == 4:
        cubemap = cubemap.unsqueeze(0)

    dirs = safe_normalize(dirs)

    abs_dirs = torch.abs(dirs)
    max_axis = torch.argmax(abs_dirs, dim=-1)
    max_vals = abs_dirs[torch.arange(N, device=device), max_axis]

    sign = torch.sign(dirs[torch.arange(N, device=device), max_axis])

    result = torch.zeros(batch_size, N, C, device=device)

    for s in range(6):
        face_idx = torch.where(
            max_axis == 0,
            torch.where(sign > 0, 0, 1),
            torch.where(
                max_axis == 1,
                torch.where(sign > 0, 2, 3),
                torch.where(sign > 0, 4, 5),
            ),
        )
        mask = face_idx == s

        if not mask.any():
            continue

        dirs_s = dirs[mask]
        max_vals_s = max_vals[mask]

        u = torch.zeros_like(max_vals_s)
        v = torch.zeros_like(max_vals_s)

        if s == 0:  # +X
            u = -dirs_s[..., 2] / max_vals_s
            v = -dirs_s[..., 1] / max_vals_s
        elif s == 1:  # -X
            u = dirs_s[..., 2] / max_vals_s
            v = -dirs_s[..., 1] / max_vals_s
        elif s == 2:  # +Y
            u = dirs_s[..., 0] / max_vals_s
            v = dirs_s[..., 2] / max_vals_s
        elif s == 3:  # -Y
            u = dirs_s[..., 0] / max_vals_s
            v = -dirs_s[..., 2] / max_vals_s
        elif s == 4:  # +Z
            u = dirs_s[..., 0] / max_vals_s
            v = -dirs_s[..., 1] / max_vals_s
        elif s == 5:  # -Z
            u = -dirs_s[..., 0] / max_vals_s
            v = -dirs_s[..., 1] / max_vals_s

        u = (u + 1) * 0.5
        v = (v + 1) * 0.5

        u = u * 2 - 1
        v = v * 2 - 1
        grid = torch.stack((u, v), dim=-1)

        grid = grid.view(-1, 1, 1, 2)

        texture = cubemap[:, s].permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(0)
        texture = texture.expand(grid.shape[0], -1, -1, -1)

        sampled = F.grid_sample(
            texture,
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=False,
        )

        result[:, mask] = sampled.view(-1, batch_size, C).permute(1, 0, 2)

    if ndim == 4:
        result = result.squeeze(0)

    return result


# ---------------------------------------------------------------------------
# Camera operations
# ---------------------------------------------------------------------------


def cam_intrinsics(fov, width, height, device=None):
    """Build a 3x3 pinhole intrinsic matrix.  *fov* is along the height axis."""
    focal = 0.5 * height / np.tan(0.5 * fov)
    intrinsics = torch.tensor(
        [[focal, 0, 0.5 * width], [0, focal, 0.5 * height], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    return intrinsics


def get_cam_matrix(phi, theta, t=None, radius=1, device=None):
    """Compute a world-to-camera 4x4 matrix for a camera on a sphere.

    Args:
        phi: Azimuth angle (radians).
        theta: Elevation angle (radians).
        t: Optional translation offset for look-at target.
        radius: Distance from origin.
    """
    z = np.sin(theta)
    r = np.cos(theta)
    pos = (
        torch.tensor(
            [r * np.cos(phi), z, r * np.sin(phi)], dtype=torch.float32, device=device
        )
        * radius
    )
    look_at = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    if t is not None:
        look_at += torch.tensor(t, dtype=torch.float32, device=device)
    w = safe_normalize(pos - look_at)
    u = safe_normalize(torch.cross(up, w, dim=-1))
    v = safe_normalize(torch.cross(w, u, dim=-1))
    translate = torch.tensor(
        [[1, 0, 0, -pos[0]], [0, 1, 0, -pos[1]], [0, 0, 1, -pos[2]], [0, 0, 0, 1]],
        dtype=pos.dtype,
        device=pos.device,
    )
    rotate = torch.tensor(
        [
            [u[0], u[1], u[2], 0],
            [v[0], v[1], v[2], 0],
            [w[0], w[1], w[2], 0],
            [0, 0, 0, 1],
        ],
        dtype=pos.dtype,
        device=pos.device,
    )
    world_to_cam = rotate @ translate
    return world_to_cam


def uv_mesh(width, height, device=None):
    """Create a [H, W, 3] mesh of homogeneous pixel coordinates."""
    uv = (
        torch.stack(
            torch.meshgrid(
                torch.arange(width) + 0.5, torch.arange(height) + 0.5, indexing="xy"
            ),
            dim=-1,
        )
        .float()
        .to(device)
    )
    uv = torch.cat([uv, torch.ones((height, width, 1), device=device)], dim=-1)
    return uv


def rotate_x(a, device=None):
    """4x4 rotation matrix around X axis by angle *a* (radians)."""
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def rotate_y(a, device=None):
    """4x4 rotation matrix around Y axis by angle *a* (radians)."""
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


# ---------------------------------------------------------------------------
# Environment map operations
# ---------------------------------------------------------------------------


def latlong_vec(res, device=None):
    """Direction vectors for a lat-long map of the given resolution.

    Returns:
        [H, W, 3] tensor of unit direction vectors.
    """
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    dir_vec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)
    return dir_vec


def envmap_vec(res, device=None):
    """Environment map direction vectors (flipped lat-long).

    Returns:
        [H, W, 3] tensor of unit direction vectors.
    """
    return -latlong_vec(res, device).flip(0).flip(1)


# ---------------------------------------------------------------------------
# Perspective projection
# ---------------------------------------------------------------------------


def get_pers_image(c2w, cubemap, pers_resolution, pers_fov=1.05, mode="bilinear"):
    """Project a panorama cubemap to a perspective view.

    Args:
        c2w: [4, 4] camera-to-world matrix.
        cubemap: [6, H, W, C] or [B, 6, H, W, C] cubemap tensor.
        pers_resolution: (height, width) of the output perspective image.
        pers_fov: Field of view in radians (along height axis).
        mode: Interpolation mode ("bilinear" or "nearest").

    Returns:
        Perspective image tensor [height, width, C].
    """
    ch_dim = cubemap.shape[-1]

    intrinsic = cam_intrinsics(pers_fov, pers_resolution[1], pers_resolution[0])
    pers_uv = uv_mesh(pers_resolution[1], pers_resolution[0])

    pos_cam = pers_uv @ torch.linalg.inv(intrinsic).T
    ray_dir = safe_normalize(pos_cam @ c2w[:3, :3].T)
    nrm_pers = -ray_dir.flip(1).contiguous()

    pers_proj = cubemap_sample_torch(
        cubemap, nrm_pers.reshape(-1, 3), mode=mode
    ).reshape(*nrm_pers.shape[:2], ch_dim)

    return pers_proj
