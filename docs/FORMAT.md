# LuxRemix file format

Each scene in the dataset is one folder named with a six-digit zero-padded
scene id (e.g. `000000/`). All files for a scene live directly inside that
folder.

## Shard layout

The dataset is distributed as tar archives, each containing **10
consecutive scenes**:

```
training-0000.tar
├── 000000/
│   ├── 0000.0001.depth.exr
│   ├── 0000.0001.normal.png
│   ├── ...
│   └── 1006.meta.json
├── 000001/
│   └── ...
└── 000009/
```

Tars are uncompressed (`.tar`, no `.tar.gz` / `.tar.zst`) because every
file inside is already either compressed (PNG, JPEG, OpenEXR-zip) or
small enough that another layer of compression isn't worth the CPU.

Shard naming:

| Split | Filename pattern | Count | Scene id range |
|---|---|---|---|
| Training | `training-NNNN.tar` (`0000`–`1203`) | 1,204 | `000000`–`014648` |
| Test  | `test-NNNN.tar` (`0000`–`0039`) | 40    | `014649`–`015082` |

## Per-scene files

Filename convention: `{pass_id}.{view_id}.{type}.{ext}` for image files and
`{pass_id}.meta.json` for per-pass metadata.

- `pass_id`: 4-digit zero-padded lighting pass index.
- `view_id`: 4-digit zero-padded camera viewpoint index, **usually one of
  `0001`–`0004`**. Most scenes have 4 viewpoints; 114 scenes have 3 (no
  `0004`) and 1 scene (`013220`) has only 1 (`0001`). The number of
  viewpoints is a property of the upstream Blender camera-sampling
  pipeline — see DATACARD.md for details. **Code consuming the dataset
  should infer the view set per scene from filenames rather than
  hard-coding 4 views.**

### Pass classes

| Pass id range | Class | What it represents |
|---|---|---|
| `0000` | geometry + reference mix | Geometry/material buffers (depth, normal, masks) **plus** the full-power "reference" mixed-light render |
| `0001`–`0003` | mix | Three additional mixed-light recompositions (random per-light power and color) |
| `1000` | background | All lights OFF; ambient + environment only |
| `1001`–`1XXX` | OLAT | One Light At a Time. Pass `1001` activates the first light only; `1002` the second; etc. The EXR stores the **background-subtracted** HDR contribution of that single light. Number of OLAT passes equals the number of room lights (typically 3–8). |

### Geometry / material buffers (pass `0000` only)

Per viewpoint:

| File | Format | Channels | Description |
|---|---|---|---|
| `0000.VVVV.depth.exr` | EXR float32 | 1 | Ray depth (camera→hit) |
| `0000.VVVV.normal.png` | PNG 8-bit | 3 | Surface normals remapped from [-1, 1] to [0, 1] |
| `0000.VVVV.diffcol.jpg` | JPEG q98 | 3 | Diffuse / albedo color |
| `0000.VVVV.transind.jpg` | JPEG q98 | 3 | Transmission indirect (light through glass) |
| `0000.VVVV.lgt_obj.png` | PNG 8-bit | 3 | Light fixture mask (per-light turbo-colormap) |
| `0000.VVVV.lgt_src.png` | PNG 8-bit | 3 | Light source emitter mask (per-light turbo-colormap) |
| `0000.VVVV.window.png` | PNG 8-bit | 1 (or 3 after some decoders' alpha-strip) | Window / door region mask |

The two light masks come from Cryptomatte: `lgt_obj` is per-object
(includes the lamp shade / fixture); `lgt_src` is per-material on the
emissive surfaces only. The `window.png` is rendered with a custom
`env_mask_material` that emits white for environment rays.

### Background pass (`1000`)

Per viewpoint: `1000.VVVV.rgb_bg.exr` (HDR, 4-channel half) and
`1000.VVVV.rgb_ldr_bg.png` (LDR 8-bit, AgX-tonemapped).

### OLAT passes (`1001`+)

Per viewpoint: `10XX.VVVV.rgb_olat.exr` (HDR, 3-channel float, BG-subtracted)
and `10XX.VVVV.rgb_ldr_olat.png` (LDR 8-bit AgX preview).

### Mixed-lighting passes (`0000`–`0003`)

Per viewpoint: `00XX.VVVV.rgb_mix.exr` (HDR, 4-channel half) and
`00XX.VVVV.rgb_ldr_mix.png` (LDR 8-bit AgX preview). Pass `0000` is the
full-power reference; passes `0001`–`0003` are random recompositions.

### Per-pass metadata (`XXXX.meta.json`)

One JSON per pass, shared by all viewpoints of that pass.

```json
{
  "scene_mod": {
    "added_ifg_lights": [{"light_id": "...", "position": [x, y, z], "rotation": r}],
    "added_led_signs": [{"position": [...], "emission_strength": N, "sign_id": "...", ...}],
    "room_id": 0,
    "room_center": [x, y]
  },
  "lighting": [
    {
      "id": "<blender object name>",
      "power": <float>,
      "color": [r, g, b],                  // normalized [0, 1]
      "type": "<Ceiling|Floor|Desk|LED_Sign_Text|LED_Sign_Plane|Non_room>",
      "colorful": <bool>,
      "position": [x, y, z],
      "update": <bool>                     // true for the active light in OLAT passes
    }
  ],
  "camera_pose": [
    {
      "position": [x, y, z],
      "rotation": [qw, qx, qy, qz],
      "fov": <float>,
      "smpl_dist": [near, far]
    }
  ],
  "blender_cfg": {
    "resolution": [2048, 1024],
    "cam_type": "PANO",
    "fov": 360,
    "spp": 64,
    "denoise": "OPENIMAGEDENOISE"
  },
  "envmap": {
    "id": "polyhaven_2k/<name>.exr",
    "env_rot": <float>,
    "env_strength": <float>,
    "env_flip": <bool>
  }
}
```

Notes on specific fields:

- `lighting` is the per-pass list of every light in the scene (room **and**
  non-room). Only the **room** lights become OLAT passes, so the length
  of `lighting` is typically larger than the number of OLAT passes.
- `lighting[].update` is `true` only for the single light activated by an
  OLAT pass. For non-OLAT passes it is `false` for every entry.
- `lighting[].id` is the Blender object name; it is stable across passes
  for the same light.
- `camera_pose` has one entry per viewpoint that exists in the scene
  (3 or 4, occasionally 1 — see the viewpoint note above).
- `envmap.id` is a relative path inside the original env-map collection
  (e.g. `polyhaven_2k/winter_river_2k.exr`). The HDRI files themselves
  are **not redistributed** with LuxRemix — download them directly from
  the upstream provider (see [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md)). The `env_*` fields
  describe how the envmap was rotated, scaled, and flipped during render.

## EXR encoding

| Pixel type | Channels | Compression | Where it's used |
|---|---|---|---|
| float32 | 1 | zip | `depth.exr` |
| float32 | 3 | zip | `rgb_olat.exr` |
| half (16-bit) | 4 | zip | `rgb_mix.exr`, `rgb_bg.exr` |

OpenCV (`cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)`)
returns the half-float EXRs as `np.float32` (it auto-promotes); that's
expected and the validator allows either.

## What is NOT in the release

- 3 known-corrupted "orphan" scenes: `007979`, `010472`, `011863`.
- `vis_grid.jpg` debug overlays — recreate with `tools/generate_vis_grid.py`
  if desired.
- The env-map HDRI files (Poly Haven, etc.) — see [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
- The perspective test sets (`test-sv`, `test-mv`).
  Users who want perspective views can regenerate them from the ERP shards
  via `tools/generate_test_sv.py` / `tools/generate_test_mv.py`.
