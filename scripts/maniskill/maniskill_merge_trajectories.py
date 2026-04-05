#!/usr/bin/env python3
"""Merge multiple ManiSkill ICL flat ``.h5`` trajectory files (v2) into one HDF5 for ICL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _expand_inputs_to_h5_files(paths: list[Path]) -> list[Path]:
    """Resolve files; each directory adds all ``*.h5`` and ``*.hdf5`` (sorted by name)."""
    out: list[Path] = []
    for raw in paths:
        p = raw.resolve()
        if p.is_dir():
            found = sorted(
                {*p.glob("*.h5"), *p.glob("*.hdf5")},
                key=lambda x: x.name.lower(),
            )
            if not found:
                raise FileNotFoundError(f"No .h5 or .hdf5 files in directory: {p}")
            out.extend(found)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"Not a file or directory: {p}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Merge ManiSkill ICL trajectory HDF5 files (flat v2). "
            "A directory (e.g. image_snapshots/) expands to all .h5/.hdf5 inside, sorted by name. "
            "Point trajectory_path at -o for training."
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .h5 path (parent dirs created as needed).",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input .h5/.hdf5 files and/or directories (dirs: all snapshots inside, name-sorted).",
    )
    p.add_argument(
        "--no-sort",
        action="store_true",
        help="Disable sort-by-return before writing (default: sort descending like PPO export).",
    )
    p.add_argument(
        "--image-compression",
        type=str,
        default="gzip",
        choices=("gzip", "lzf", "none"),
        help="Image dataset compression in output (default: gzip).",
    )
    p.add_argument(
        "--image-gzip-level",
        type=int,
        default=4,
        help="gzip level when --image-compression gzip (default: 4).",
    )
    args = p.parse_args()

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.data.maniskill_io import merge_maniskill_trajectory_hdf5

    out = args.output.resolve()
    try:
        paths = _expand_inputs_to_h5_files(list(args.inputs))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    for path in paths:
        suf = path.suffix.lower()
        if suf not in (".h5", ".hdf5"):
            print(f"error: expected .h5/.hdf5: {path}", file=sys.stderr)
            return 1

    print(f"Merging {len(paths)} file(s):")
    for path in paths:
        print(f"  {path}")

    try:
        n_ep, n_t = merge_maniskill_trajectory_hdf5(
            paths,
            out,
            sort_by_return=not args.no_sort,
            image_hdf5_compression=str(args.image_compression),
            image_gzip_level=int(args.image_gzip_level),
        )
    except (OSError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {out}  episodes={n_ep}  total_timesteps={n_t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
