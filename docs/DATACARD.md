# LuxRemix dataset card

This dataset card follows the structure of *Datasheets for Datasets*
(Gebru et al., 2018) so consumers can quickly assess whether LuxRemix fits
their research question.

## Motivation

LuxRemix exists to support research on **lighting decomposition and
re-mixing of indoor scenes**. Specifically, it provides per-light
one-light-at-a-time (OLAT) HDR renders so that machine-learning models can
learn to:

- decompose a single indoor image into per-light contributions, and
- re-mix those contributions to produce novel illumination conditions.

The dataset was created by Meta Reality Labs to accompany the CVPR 2026
paper *LuxRemix: Lighting Decomposition and Remixing for Indoor Scenes*
(Liang et al.).

## Composition

| | |
|---|---|
| Instances | 12,439 scenes (12,039 training + 400 test) |
| Total tar shards | 1,244 (1,204 training + 40 test), ~8 GB each |
| Total size | ~9.7 TB (1.48 M files) |
| Image resolution | 2048 × 1024, equirectangular panorama |
| Viewpoints per scene | 4 for 12,324 scenes; 3 for 114 scenes; 1 for 1 scene |
| Lights per scene | 2–7 (mean ≈ 5) |
| Pass types | geometry (pass `0000` only), background (`1000`), OLAT (`1001`+), reference + 3 random mixes (`0000`–`0003`) |
| File formats | EXR (HDR), PNG (LDR previews + masks), JPEG (auxiliary), JSON (metadata) |
| Splits | Training: scene ids `000000`–`014648`; Test: `014649`–`015082`. Zero overlap. |

### What each scene contains

- **Geometry / material buffers** rendered once per viewpoint (pass `0000`
  only): depth, surface normals, diffuse color, transmission indirect,
  per-light fixture mask, per-light source-emitter mask, window mask.
- **Background pass** (`1000`): all lights OFF; ambient + environment only.
- **OLAT passes** (`1001`+): one pass per room light, each with the
  background subtracted to isolate that light's contribution. Number of
  passes equals number of room lights.
- **Mixed-light passes** (`0000`–`0003`): the full-power reference render
  plus three random recompositions sampled at render time.
- **Per-pass metadata** (`XXXX.meta.json`): lighting configuration,
  camera poses, Blender configuration, env-map reference.

See [FORMAT.md](FORMAT.md) for the on-disk layout and metadata schema.

## Collection process

Each scene was constructed and rendered as follows:

1. Source indoor scene loaded from Aria Synthetic Environments into Blender.
2. Scene modification: pre-existing "fake" lights removed; procedural
   ceiling / floor / desk lamps from
   [Infinigen](https://github.com/princeton-vl/infinigen) inserted; LED
   sign textures (Blender-generated text or [Twemoji](https://github.com/twitter/twemoji)
   PNG textures) swapped onto picture frames where appropriate.
3. Room selection: one room per scene was picked based on light coverage
   and camera coverage.
4. Camera sampling: up to 4 panoramic camera viewpoints sampled within
   the room (equirectangular, 2048 × 1024). For some scenes the sampler
   only found 3 (or, in one case, 1) valid positions; see "Known
   artifacts" below.
5. Rendering (Blender Cycles, 64 SPP, OpenImageDenoise): geometry buffers
   for pass `0000`, background pass `1000`, one OLAT pass per room light,
   and three random mixed-light passes.
6. Post-processing: background subtraction per OLAT pass; AgX OCIO
   tonemap (via the [EaryChow/AgX](https://github.com/EaryChow/AgX) color
   config, included as a git submodule) applied to produce LDR previews.

Env maps for ambient lighting were drawn from
[Poly Haven](https://polyhaven.com/) (`polyhaven_2k/...`) and an
[UrbanSky](https://cave.cs.columbia.edu/repository/UrbanSky) collection (`urbansky_2k/...`). The HDRI files themselves
are not redistributed — see ACKNOWLEDGMENTS.md.

## Preprocessing / labeling

- **Metadata scrubbing**: internal-only paths and internal asset identifiers
  have been stripped from every `meta.json` before release.
- **Excluded files**: `vis_grid.jpg` debug overlays are not packaged.
  Recreate the vis grid for any scene with `tools/generate_vis_grid.py`.
- **Excluded scenes**: three known-corrupted "orphan" scenes (`007979`,
  `010472`, `011863`) — they are missing files in source — are not in either split.

## Uses

### Intended uses

- Lighting decomposition and re-mixing for indoor scenes (the LuxRemix
  task itself).
- Inverse-rendering research that benefits from ground-truth per-light
  illumination.
- Panorama-conditioned image synthesis where per-light components are
  useful as an intermediate signal.

### Out-of-scope uses

This dataset is licensed for **non-commercial research only**
(see [LICENSE.md](../LICENSE.md)). In addition, do not use it to:

- identify or infer personal information about any individual (the
  source scenes are synthetic and contain no real people, but the
  license still prohibits it);
- redistribute the data without prior written permission from Meta;
- train or evaluate **commercial** products.

## Known artifacts

| Issue | Scope | Notes |
|---|---|---|
| **Partial viewpoint sets** | 115 scenes (~0.92%) | Most scenes have 4 viewpoints; 114 scenes have 3 (no `0004`); scene `013220` has only 1 (`0001`). This reflects the upstream Blender camera sampler accepting fewer cameras when it can't place 4 valid positions. Code consuming the dataset **must infer the view set per scene** from filenames rather than hard-coding 4. The reference dataloader in this repo (`dataset.py`) does this. |
| **Half-float EXR auto-promotion** | All half-float EXRs | OpenCV's `cv2.imread` returns half-float EXRs (`rgb_mix.exr`, `rgb_bg.exr`) as `np.float32`. The on-disk representation is half, but consumers should expect float32 in memory. |
| **Cryptomatte mask determinism** | `lgt_obj.png`, `lgt_src.png` | Per-light color assignment in the Cryptomatte-derived masks is stable within a scene but uses a turbo colormap rather than a fixed per-light palette. Treat as opaque per-light identifiers. |
| **3 orphan scenes** | `007979`, `010472`, `011863` | Excluded from the release. If you encounter these ids elsewhere, ignore them. |

## Distribution

- **Code & docs**: this repository, on github.com.
- **Data**: [https://ai.meta.com/datasets/luxremix-dataset/](https://ai.meta.com/datasets/luxremix-datasets/)
  exposes the shards through a CDN. The portal serves `dataset-shards.txt`,
  a TSV with header `file_name<TAB>cdn_link` and one row per shard
  (filenames preserved). The CDN URLs rotate roughly every six months —
  re-download the file when the old ones stop working. `download.py`
  consumes this TSV directly.
- **License**: Aria Synthetic Environments Dataset License Agreement
  (see [LICENSE.md](../LICENSE.md)).

## Maintenance

- **Issue tracker**: file issues at the repository's GitHub issues page.
- **Contact**: the LuxRemix authors via the project page,
  [https://luxremix.github.io](https://luxremix.github.io).
- **Versioning**: the released dataset is v1.
  Future revisions will be announced via the project page.
