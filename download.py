#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Download LuxRemix dataset shards from a list of HTTPS URLs.

Reads a text file with one URL per line (fetched from the dataset portal
at https://ai.meta.com/datasets/luxremix-dataset/) and downloads the
corresponding tar shards in parallel. Optionally unpacks each tar after
download.

The script infers the split (training vs test) from each shard's filename
prefix (``training-`` / ``test-``), so a ``--split`` flag can filter the
URL list without editing the source file.

Usage:
    # Download everything to /data/luxremix/
    python download.py luxremix_urls.txt --output-dir /data/luxremix

    # Test split only, unpack each tar and delete it afterwards
    python download.py luxremix_urls.txt --output-dir /data/luxremix \\
        --split test --unpack

    # Training split only, keep tars after download (no unpack)
    python download.py luxremix_urls.txt --output-dir /data/luxremix \\
        --split training --keep-tars
"""

import argparse
import logging
import os
import sys
import tarfile
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB
DEFAULT_REQUEST_TIMEOUT = (30, 300)  # (connect, read) seconds


def read_url_list(path: Path) -> list[str]:
    """Read non-blank, non-comment lines as URLs."""
    urls: list[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def shard_name_from_url(url: str) -> str:
    """Extract `training-0042` from `https://.../training-0042.tar`."""
    name = os.path.basename(urlparse(url).path)
    if name.endswith(".tar"):
        name = name[:-4]
    return name


def split_of(shard_name: str) -> str | None:
    """Map shard name → split. Returns None if it's neither training nor test."""
    if shard_name.startswith("training-"):
        return "training"
    if shard_name.startswith("test-"):
        return "test"
    return None


def filter_urls(urls: list[str], wanted_split: str) -> list[str]:
    """Keep only URLs whose shard belongs to the requested split."""
    if wanted_split == "all":
        return urls
    out = []
    for u in urls:
        s = split_of(shard_name_from_url(u))
        if s == wanted_split:
            out.append(u)
    return out


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


def process_url(
    url: str,
    output_dir: Path,
    unpack: bool,
    keep_tars: bool,
    retries: int,
    print_lock: threading.Lock,
) -> dict:
    """Download (and optionally unpack) one shard."""
    shard_name = shard_name_from_url(url)
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
        "url_list",
        type=Path,
        help="Text file with one HTTPS URL per line (fetched from the portal)",
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

    if not args.url_list.is_file():
        raise SystemExit(f"URL list file not found: {args.url_list}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_urls = read_url_list(args.url_list)
    urls = filter_urls(all_urls, args.split)
    if not urls:
        raise SystemExit(
            f"no URLs match --split={args.split} (read {len(all_urls)} from "
            f"{args.url_list})"
        )

    logger.info(
        f"Downloading {len(urls)} shards (split={args.split}, "
        f"workers={args.workers}, unpack={args.unpack})"
    )

    print_lock = threading.Lock()
    results: list[dict] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                process_url,
                url,
                args.output_dir,
                args.unpack,
                args.keep_tars,
                args.retries,
                print_lock,
            ): url
            for url in urls
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
    print(f"  Total URLs:           {len(urls)}")
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
