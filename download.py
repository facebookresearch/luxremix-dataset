#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Download LuxRemix dataset shards from the Meta AI Datasets portal.

Reads a TSV file produced by the dataset portal at
https://ai.meta.com/datasets/luxremix-dataset/ — `dataset-shards.txt`,
with a header line ``file_name\\tcdn_link`` followed by one
``<filename>\\t<https-url>`` row per shard — and downloads each tar in
parallel. Optionally unpacks each tar after download.

The script uses each shard's ``file_name`` (e.g. ``training-0042.tar``)
both for on-disk naming and to infer the split (training vs test), so a
``--split`` flag can filter the list without editing the source file.

Usage:
    # Download everything to /data/luxremix/
    python download.py dataset-shards.txt --output-dir /data/luxremix

    # Test split only, unpack each tar and delete it afterwards
    python download.py dataset-shards.txt --output-dir /data/luxremix \\
        --split test --unpack

    # Training split only, keep tars after download (no unpack)
    python download.py dataset-shards.txt --output-dir /data/luxremix \\
        --split training --keep-tars
"""

import argparse
import csv
import logging
import sys
import tarfile
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB
DEFAULT_REQUEST_TIMEOUT = (30, 300)  # (connect, read) seconds


def read_shard_list(path: Path) -> list[tuple[str, str]]:
    """Parse the portal TSV into `(shard_name, url)` pairs.

    The portal serves a TSV with at least the columns `file_name` and
    `cdn_link`. We trust `file_name` for naming and split inference;
    the CDN URL path is an opaque hash and cannot be used to recover
    the shard name. Extra columns (e.g. a future `sha256`) are ignored.
    """
    required = ("file_name", "cdn_link")
    pairs: list[tuple[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"{path}: TSV is missing required column(s) {missing}; "
                f"got header {reader.fieldnames}. Did you download the "
                f"wrong file from https://ai.meta.com/datasets/luxremix-dataset/ ?"
            )
        for row in reader:
            file_name, url = row["file_name"], row["cdn_link"]
            if not file_name.endswith(".tar"):
                raise SystemExit(
                    f"{path}:{reader.line_num}: file_name does not end in '.tar': "
                    f"{file_name!r}"
                )
            pairs.append((file_name.removesuffix(".tar"), url))
    pairs.sort()
    return pairs


def split_of(shard_name: str) -> str | None:
    """Map shard name → split. Returns None if it's neither training nor test."""
    if shard_name.startswith("training-"):
        return "training"
    if shard_name.startswith("test-"):
        return "test"
    return None


def filter_shards(
    shards: list[tuple[str, str]], wanted_split: str
) -> list[tuple[str, str]]:
    """Keep only `(shard_name, url)` pairs whose shard belongs to the requested split."""
    if wanted_split == "all":
        return shards
    return [(name, url) for name, url in shards if split_of(name) == wanted_split]


def download_with_retries(
    url: str, tar_path: Path, retries: int, logger_prefix: str
) -> None:
    """Stream-download `url` to `tar_path` with retry-on-error."""
    tmp_path = tar_path.with_suffix(tar_path.suffix + ".part")
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            tmp_path.unlink(missing_ok=True)
            with requests.get(url, stream=True, timeout=DEFAULT_REQUEST_TIMEOUT) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            tmp_path.rename(tar_path)
            return
        except Exception as e:
            last_err = e
            backoff = min(2 ** (attempt - 1), 30)
            logger.warning(
                f"{logger_prefix} download failed (attempt {attempt}/{retries}): "
                f"{e}; retrying in {backoff}s"
            )
            time.sleep(backoff)

    tmp_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"{logger_prefix} download failed after {retries} attempts: {last_err}"
    )


def unpack_tar(tar_path: Path, dest_root: Path) -> None:
    """Extract a shard tar into dest_root/ (which produces dest_root/<scene_id>/...)."""
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dest_root)


def process_shard(
    shard_name: str,
    url: str,
    output_dir: Path,
    unpack: bool,
    keep_tars: bool,
    retries: int,
    print_lock: threading.Lock,
) -> dict:
    """Download (and optionally unpack) one shard."""
    split = split_of(shard_name) or "unknown"
    tar_path = output_dir / f"{shard_name}.tar"
    result: dict = {
        "shard_name": shard_name,
        "split": split,
        "downloaded": False,
        "unpacked": False,
        "error": None,
    }

    prefix = f"[{shard_name}]"

    try:
        if tar_path.exists():
            with print_lock:
                logger.info(f"{prefix} tar already present, skipping download")
        else:
            download_with_retries(url, tar_path, retries, prefix)
            result["downloaded"] = True

        if unpack:
            unpack_tar(tar_path, output_dir)
            result["unpacked"] = True
            if not keep_tars:
                tar_path.unlink(missing_ok=True)

    except Exception as e:
        result["error"] = str(e)
        with print_lock:
            logger.error(f"{prefix} {e}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "shard_list",
        type=Path,
        help="TSV file `dataset-shards.txt` from the portal "
        "(header: `file_name<TAB>cdn_link`, one row per shard)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for tars (and unpacked scenes, if --unpack)",
    )
    parser.add_argument(
        "--split",
        choices=["training", "test", "all"],
        default="all",
        help="Filter URLs by shard filename prefix (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of concurrent downloads (default: 4)",
    )
    parser.add_argument(
        "--unpack",
        action="store_true",
        help="Extract each tar into --output-dir after download",
    )
    parser.add_argument(
        "--keep-tars",
        action="store_true",
        help="Keep tar files after unpacking (default: delete to save disk)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        metavar="N",
        help="Per-tar retry budget for transient download errors (default: 5)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if not args.shard_list.is_file():
        raise SystemExit(f"shard list file not found: {args.shard_list}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_shards = read_shard_list(args.shard_list)
    shards = filter_shards(all_shards, args.split)
    if not shards:
        raise SystemExit(
            f"no shards match --split={args.split} (read {len(all_shards)} from "
            f"{args.shard_list})"
        )

    logger.info(
        f"Downloading {len(shards)} shards (split={args.split}, "
        f"workers={args.workers}, unpack={args.unpack})"
    )

    print_lock = threading.Lock()
    results: list[dict] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                process_shard,
                shard_name,
                url,
                args.output_dir,
                args.unpack,
                args.keep_tars,
                args.retries,
                print_lock,
            ): shard_name
            for shard_name, url in shards
        }
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc="shards", unit="shard"
        ):
            results.append(fut.result())

    elapsed = time.perf_counter() - t0
    downloaded = sum(1 for r in results if r["downloaded"])
    cached = sum(1 for r in results if not r["downloaded"] and not r["error"])
    errored = sum(1 for r in results if r["error"])
    unpacked = sum(1 for r in results if r["unpacked"])

    print()
    print("=" * 60)
    print(" DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Total shards:         {len(shards)}")
    print(f"  Downloaded:           {downloaded}")
    print(f"  Already cached:       {cached}")
    print(f"  Errors:               {errored}")
    if args.unpack:
        print(f"  Unpacked:             {unpacked}")
    print(f"  Wall time:            {elapsed:.0f}s")
    print("=" * 60)

    if errored:
        sys.exit(1)


if __name__ == "__main__":
    main()
