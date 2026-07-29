"""Split a large checkpoint into verified chunks and reconstruct it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_checkpoint(checkpoint: Path, output_dir: Path, chunk_mib: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = chunk_mib * 1024 * 1024
    parts = []
    with checkpoint.open("rb") as source:
        index = 0
        while block := source.read(chunk_size):
            part_path = output_dir / f"{checkpoint.name}.part{index:03d}"
            part_path.write_bytes(block)
            parts.append(
                {
                    "name": part_path.name,
                    "size": len(block),
                    "sha256": hashlib.sha256(block).hexdigest(),
                }
            )
            index += 1
    manifest = {
        "schema_version": 1,
        "original_name": checkpoint.name,
        "original_size": checkpoint.stat().st_size,
        "original_sha256": sha256(checkpoint),
        "chunk_size_bytes": chunk_size,
        "parts": parts,
    }
    manifest_path = output_dir / "checkpoint_chunks_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def join_checkpoint(manifest_path: Path, output_path: Path | None) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks_dir = manifest_path.parent
    destination = output_path or chunks_dir / manifest["original_name"]
    with destination.open("wb") as output:
        for part in manifest["parts"]:
            part_path = chunks_dir / part["name"]
            if part_path.stat().st_size != part["size"]:
                raise ValueError(f"Unexpected size for {part_path}")
            if sha256(part_path) != part["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {part_path}")
            with part_path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
    if destination.stat().st_size != manifest["original_size"]:
        raise ValueError("Reconstructed checkpoint has an unexpected size")
    if sha256(destination) != manifest["original_sha256"]:
        raise ValueError("Reconstructed checkpoint SHA-256 mismatch")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("checkpoint", type=Path)
    split_parser.add_argument("output_dir", type=Path)
    split_parser.add_argument("--chunk-mib", type=int, default=12)

    join_parser = subparsers.add_parser("join")
    join_parser.add_argument("manifest", type=Path)
    join_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "split":
        result = split_checkpoint(
            args.checkpoint, args.output_dir, args.chunk_mib
        )
    else:
        result = join_checkpoint(args.manifest, args.output)
    print(result)


if __name__ == "__main__":
    main()
