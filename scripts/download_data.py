"""Fetch the SemEval-2014 Task 4 restaurant data into ``data/raw/``.

The dataset is third-party licensed, so it is deliberately not committed to this
repository. Run this once after cloning::

    python scripts/download_data.py

Every file is pinned to a specific upstream commit and verified against a known
SHA-256 digest, so a successful run always produces byte-identical data. If a
digest ever fails, the script refuses the file rather than silently training on
something unexpected.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

CHUNK_BYTES = 64 * 1024
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Source:
    """One dataset file and everything needed to fetch and trust it."""

    filename: str
    url: str
    sha256: str
    size_bytes: int
    note: str


# Both files are pinned by commit SHA rather than a branch name. Branches move;
# a commit SHA is immutable, so this stays reproducible even if either upstream
# repository changes.
SOURCES: tuple[Source, ...] = (
    Source(
        filename="Restaurants_Train_v2.xml",
        url=(
            "https://raw.githubusercontent.com/davidsbatista/"
            "Aspect-Based-Sentiment-Analysis/"
            "a609d9b1a8920abd0dffb1d106404a055e73eb64/"
            "datasets/ABSA-SemEval2014/Restaurants_Train_v2.xml"
        ),
        sha256="223601da1bded6caa4ef9cf91a7007578141ca6d8ed50d5a5c217565f89d2fc5",
        size_bytes=1_235_614,
        note="3041 sentences, 3713 aspect-category annotations, fully labelled.",
    ),
    Source(
        filename="Restaurants_Test_Gold.xml",
        url=(
            "https://raw.githubusercontent.com/shenghaowang/"
            "absa-for-restaurant-reviews/"
            "6a9ee5260ef776363d039231b6803d45aa3ff685/"
            "data/semeval2014/raw/Restaurants_Test_Gold.xml"
        ),
        sha256="f21509cfa37e16534cd5b2da043be487355b64ef48fe8d6aaacaeca6b49cc0fb",
        size_bytes=359_021,
        note=(
            "800 sentences, 1025 annotations WITH polarities. Note this is the "
            "gold file, not Restaurants_Test_Data_phaseB.xml. During the shared "
            "task, phase B handed competitors the gold aspects and asked them to "
            "predict polarity, so its polarity attributes are stripped and it is "
            "useless as a test set. The two files carry identical sentence ids."
        ),
    ),
)


def sha256_of(path: Path) -> str:
    """Digest a file in chunks so memory use stays flat regardless of size."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(source: Source, path: Path) -> bool:
    return path.is_file() and sha256_of(path) == source.sha256


def download(source: Source, destination: Path) -> None:
    """Stream to a temporary file, then move into place once verified.

    Downloading straight to the destination risks leaving a truncated file that
    looks real if the connection drops midway. Writing to a ``.part`` file and
    renaming only after the digest matches makes the operation atomic.
    """
    temporary = destination.with_suffix(destination.suffix + ".part")

    response = requests.get(source.url, stream=True, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()

    with temporary.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            handle.write(chunk)

    actual = sha256_of(temporary)
    if actual != source.sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {source.filename}\n"
            f"  expected {source.sha256}\n"
            f"  actual   {actual}\n"
            "Refusing to use this file. The upstream source may have changed."
        )

    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if a valid local copy already exists",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"destination: {RAW_DIR}\n")

    for source in SOURCES:
        destination = RAW_DIR / source.filename

        if not args.force and is_valid(source, destination):
            print(f"[skip]     {source.filename} (already present, checksum ok)")
            continue

        print(f"[download] {source.filename} ...", end=" ", flush=True)
        try:
            download(source, destination)
        except Exception as error:  # noqa: BLE001 - report and fail clearly
            print("FAILED")
            print(f"           {error}", file=sys.stderr)
            return 1
        print(f"ok ({source.size_bytes:,} bytes)")

    print("\nAll files present and verified:")
    for source in SOURCES:
        print(f"  - {source.filename}: {source.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
