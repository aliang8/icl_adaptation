"""Shared video I/O helpers (MP4 frame reading, frame validation)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def read_mp4_frames(path: Path) -> List[np.ndarray]:
    """Read all frames from an MP4 as list of (H,W,3) uint8 (RGB)."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame[:, :, ::-1])  # BGR -> RGB
        cap.release()
        return frames
    except Exception:
        try:
            import imageio

            reader = imageio.get_reader(str(path), "ffmpeg")
            frames = []
            i = 0
            while True:
                try:
                    frames.append(np.asarray(reader.get_data(i), dtype=np.uint8))
                    i += 1
                except (IndexError, RuntimeError):
                    break
            reader.close()
            return frames
        except Exception:
            return []


def ensure_uint8_rgb_frames(frames: np.ndarray) -> np.ndarray:
    """Ensure frames are (T,H,W,3) uint8; clip and convert if needed. Raises if shape invalid."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames (T,H,W,3), got {frames.shape}")
    return frames
