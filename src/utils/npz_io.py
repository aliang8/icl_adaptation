"""Shared NPZ I/O: load/save dict of arrays with atomic write for save."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np


def load_npz_arrays(path: Path) -> Dict[str, np.ndarray]:
    """Load all arrays from an .npz into a dict. Raises FileNotFoundError if missing."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.load(path, allow_pickle=False)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k])
    return out


def save_npz_arrays(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Save dict of arrays to .npz with atomic write (temp file in same dir, then replace)."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".npz.tmp", dir=path.parent, prefix=path.stem + "_")
    try:
        os.close(fd)
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, os.fspath(path))
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
