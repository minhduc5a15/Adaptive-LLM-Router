"""Download the pinned RouterBench release and verify every file."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
BASE_URL = f"https://huggingface.co/datasets/withmartian/routerbench/resolve/{REVISION}"
FILES = {
    "routerbench_0shot.pkl": (
        99_567_659,
        "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d",
    ),
    "routerbench_5shot.pkl": (
        171_091_324,
        "fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe",
    ),
    "routerbench_raw.pkl": (
        1_198_134_681,
        "18e919267500a1d49302e0efe25cf7d9aa2f5b74aa209f5230442c2cfbee7971",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, size: int, digest: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest


def download_file(directory: Path, name: str, size: int, digest: str) -> None:
    destination = directory / name
    if verified(destination, size, digest):
        print(f"OK: {destination}", flush=True)
        return

    partial = directory / f"{name}.part"
    print(f"Downloading {name} ({size / 1_000_000:.1f} MB)...", flush=True)
    subprocess.run(
        [
            "curl", "--fail", "--location", "--retry", "3",
            "--silent", "--show-error",
            "--continue-at", "-", "--output", str(partial),
            f"{BASE_URL}/{name}",
        ],
        check=True,
    )
    if not verified(partial, size, digest):
        raise RuntimeError(f"Checksum or size mismatch: {partial}")
    partial.replace(destination)
    print(f"Verified SHA-256: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw",
        help="Destination directory (default: project/data/raw)",
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify existing files without downloading",
    )
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        destination = args.directory / name
        if args.check:
            if not verified(destination, size, digest):
                raise SystemExit(f"Missing or invalid: {destination}")
            print(f"OK: {destination}", flush=True)
        else:
            download_file(args.directory, name, size, digest)


if __name__ == "__main__":
    main()
