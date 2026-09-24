# Using LuxRemix

This guide walks through the common things you'll want to do with the
dataset, from a clean checkout.

## 1. Install

```bash
git clone https://github.com/<org>/LuxRemix_dataset.git
cd LuxRemix_dataset
git submodule update --init  # AgX OCIO color config
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional but recommended for accurate AgX tonemapping:

```bash
pip install PyOpenColorIO
```

Without `PyOpenColorIO`, the dataloader and `tools/generate_vis_grid.py` fall
back to a Reinhard + sRGB-gamma approximation that is close but not
bit-exact to the released LDR previews.

## 2. Get the shard URL list

Visit the dataset portal:

> [https://ai.meta.com/datasets/luxremix-dataset/](https://ai.meta.com/datasets/luxremix-dataset/)

Accept the license and download `dataset-shards.txt`. The file is a TSV
with header `file_name<TAB>cdn_link` and one row per shard — the
`file_name` column preserves the original shard names (`training-NNNN.tar`
/ `test-NNNN.tar`) and `cdn_link` is the HTTPS URL the script will fetch.
The portal rotates these URLs roughly every six months; just re-download
the list when the old ones stop working.

## 3. Download

### Smoke test: one test shard

```bash
# Keep the header + the first data row so the parser is happy.
(head -1 dataset-shards.txt && sed -n '2p' dataset-shards.txt) > one_test.tsv
python download.py one_test.tsv --output-dir ./lx --split test --unpack
```

This downloads ~8 GB, unpacks it into `./lx/<scene_id>/`, and deletes the
tar after a successful unpack. Expect ~30–60 s of network time.

### Full test split (40 shards, ~300 GB)

```bash
python download.py dataset-shards.txt --output-dir ./lx --split test \
    --unpack --workers 4
```

### Full training split (1,204 shards, ~9.4 TB)

```bash
python download.py dataset-shards.txt --output-dir ./lx --split training \
    --workers 4 --keep-tars
```

Notes:

- `--keep-tars` is convenient if you have the disk and want to keep the
  tars around for re-extraction or archiving.
- `--workers 4` is a sweet spot for ~330 MB/s per-stream throughput;
  higher values risk bandwidth contention.
- The script is **resumable**: if a `.tar` file already exists in
  `--output-dir`, that URL is skipped. Killing and restarting picks up
  where you left off.

## 4. Use the reference PyTorch dataloader

```python
from dataset import LuxRemixDataset
from torch.utils.data import DataLoader

ds = LuxRemixDataset(
    scenes_dir="./lx",          # directory of unpacked scene folders
    perspective_size=512,        # cubemap face size
    frames_per_sample=4,         # multi-view frame count
)

for batch in DataLoader(ds, batch_size=2, num_workers=4):
    rgb_olat = batch["rgb_olat"]       # (B, F, 3, H, W) float32
    rgb_mix  = batch["rgb_mix"]        # (B, F, 3, H, W) float32 (input)
    lgt_mask = batch["lgt_mask"]       # (B, F, 1, H, W)
    light    = batch["light"]          # dict of (B, …) tensors: power, color, type
    ...
```

The dataloader handles the [variable-viewpoint scenes](DATACARD.md#known-artifacts)
gracefully — it samples viewpoints from each scene's actual view set.
See `examples/training_loop.py` for a runnable starter.

## 5. Regenerate perspective views from ERP

The released dataset is ERP panoramas. If you want perspective pinhole
views (the format used by the LuxRemix paper for some evaluations), you
need the test split from step 3 and the camera-parameter file for the
test scenes you want:

| File | Test scenes | Use it for |
|---|---|---|
| `data/camera_params_48.json` | 48 curated scenes | The perspective test sets used in the LuxRemix paper (`test-sv`, `test-mv`); Tables 1 and 2 score a [30-scene subset](#which-test-scenes-were-used-for-tables-1-and-2-of-the-paper) |
| `data/camera_params_352.json` | The other 352 scenes | Additional perspective test data with no paper counterpart |

```bash
# Single-view perspective (the test-sv format)
python tools/generate_test_sv.py --erp-dir ./lx \
    --camera-params data/camera_params_48.json --output-dir ./test-sv

# Multi-view perspective (the test-mv format)
python tools/generate_test_mv.py --erp-dir ./lx \
    --camera-params data/camera_params_48.json --output-dir ./test-mv
```

Pass `data/camera_params_352.json` instead for the remaining scenes. Each
script generates every scene listed in the file it is given.

For the 48 curated scenes, the output is bit-exact with the paper's test
sets, including the per-pass light masks, which are reproduced from
`data/mask_strategies_*.json`. The 352 remaining scenes have no paper
reference: their cameras were sampled with the same procedure, seeded per
scene, and their masks use a seeded fallback, so repeated runs produce
identical output.

## 6. Visualization

```bash
python tools/generate_vis_grid.py ./lx/000000/
```

Produces a labeled grid showing the background, each OLAT pass, and the
reference mix render for one scene. Useful for sanity-checking a
downloaded scene at a glance.

## FAQ

### Why are some scenes missing the 4th viewpoint?

The upstream Blender camera-sampling pipeline accepts fewer viewpoints
when it can't place 4 valid camera positions in the room. 114 scenes
ended up with 3 viewpoints and 1 (scene `013220`) with just 1. This is
the source-data truth, not a packaging defect. See
[DATACARD.md](DATACARD.md#known-artifacts).

### Where are the environment-map HDRI files?

We don't redistribute the HDRIs to respect their upstream licenses.
The `envmap.id` field in each `meta.json` is a portable relative path
(e.g. `polyhaven_2k/winter_river_2k.exr`) — fetch the files from
[Poly Haven](https://polyhaven.com/) directly.

### Why are the half-float EXRs loading as `float32` in OpenCV?

`cv2.imread` silently promotes half-float EXR to `float32` on read.
The on-disk encoding is half (16-bit); in memory you'll see float32.
The reference dataloader and validator both expect this.

### Can I redistribute the dataset?

No — see Section 3 of [LICENSE.md](../LICENSE.md). The license permits use for
non-commercial research only and prohibits redistribution without prior
written permission from Meta.

### Can I use LuxRemix commercially?

No — non-commercial research only. Contact projectaria@meta.com for
inquiries about commercial use.

### Is there a perspective version of the test split?

Not as a download, but you can regenerate it. The original LuxRemix paper
used perspective versions of 48 curated test scenes (`test-sv`,
`test-mv`). Regenerate them from the released ERP shards with
`data/camera_params_48.json` (see step 5 above); the output is bit-exact
with the originals. `data/camera_params_352.json` covers the remaining 352
test scenes.

### Which test scenes were used for Tables 1 and 2 of the paper?

30 of the 48 curated test scenes, with 112 (scene, light) cases in total.
`data/paper_eval_cases.json` lists, for each scene, its `test-sv` view and
the OLAT passes that are scored. After generating the 48 scenes (step 5),
the ground-truth images are:

- Table 1 (single-image lighting decomposition):
  `test-sv/{scene}.{view_id}/{pass_id}.rgb_ldr_olat.png`, 112 images.
- Table 2 (multi-view lighting harmonization):
  `test-mv/{scene}/{view}.{pass_id}.rgb_ldr_olat.png` for every `view` in
  `mv_eval_views`, 1,568 images.
