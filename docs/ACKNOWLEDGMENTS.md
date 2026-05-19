# Acknowledgments

The LuxRemix dataset and accompanying code build on the work of several
open-source projects, asset libraries, and prior datasets. We are grateful
to the maintainers and contributors of each.

## Source scenes

- **Aria Synthetic Environments (ASE)** — Meta Reality Labs.
  - Project: [https://www.projectaria.com/datasets/ase/](https://www.projectaria.com/datasets/ase/)
  - License: [Aria Synthetic Environments Dataset License Agreement](https://www.projectaria.com/datasets/ase/license/) — see [LICENSE.md](../LICENSE.md), which is derived from this license.

## Procedural assets

- **Infinigen** — Princeton VL.
  Procedural Blender assets used for ceiling / floor / desk lamps added
  during scene modification.
  - Repo: [https://github.com/princeton-vl/infinigen](https://github.com/princeton-vl/infinigen)
  - License: BSD-3-Clause.

## Environment maps (HDRIs)

The HDRI files themselves are **not redistributed** with LuxRemix. Each
scene's `meta.json` carries a portable relative path
(`<collection>/<name>.exr`) so consumers can fetch the originals from
the upstream providers:

- **Poly Haven** — `polyhaven_2k/...`
  - Site: [https://polyhaven.com/](https://polyhaven.com/)
  - License: CC0 (public domain).
- **UrbanSky** — `urbansky_2k/...`
  - Source: [https://cave.cs.columbia.edu/repository/UrbanSky](https://cave.cs.columbia.edu/repository/UrbanSky) (Columbia Computer Vision Laboratory).

## LED-sign textures

- **Twemoji** — PNG textures for LED-sign-plane lights.
  - Repo: [https://github.com/twitter/twemoji](https://github.com/twitter/twemoji)
  - License: CC-BY 4.0 (graphics), MIT (code).

## Color management

- **AgX** — EaryChow.
  OCIO color configuration used for HDR → LDR tonemapping in both the
  rendering pipeline and the released LDR previews. Included as a git
  submodule under `colormanagement/`.
  - Repo: [https://github.com/EaryChow/AgX](https://github.com/EaryChow/AgX)
  - License: see the submodule's repository.

## Rendering & ML stack

LuxRemix was rendered with [Blender](https://www.blender.org/) (Cycles
engine, 64 SPP, OpenImageDenoise). The reference dataloader and tooling
rely on [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/),
[NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/),
[requests](https://requests.readthedocs.io/),
[tqdm](https://github.com/tqdm/tqdm), and
[PyOpenColorIO](https://github.com/AcademySoftwareFoundation/OpenColorIO).
We thank the maintainers of each.

## Paper authors

LuxRemix was developed by Ruofan Liang, Norman Müller, Ethan Weber,
Duncan Zauss, Nandita Vijaykumar, Peter Kontschieder, and Christian
Richardt. See [CITATION.cff](../CITATION.cff) and the BibTeX block in [README.md](../README.md#how-to-cite) for the full citation.

## Reporting an attribution gap

If a third-party asset, model, or piece of code used by LuxRemix is
missing from this list, please file an issue. We will update both this
file and the dataset card promptly.
