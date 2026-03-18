"""Shared NPZ I/O: load/save dict of arrays with atomic write for save."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def load_npz_arrays(path: Path) -> Dict[str, np.ndarray]:
    """Load all arrays from an .npz into a dict. Raises FileNotFoundError if missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.load(path, allow_pickle=False)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k])
    return out


def save_npz_arrays(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Save dict of arrays to .npz with atomic write (write to .tmp then replace)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)
