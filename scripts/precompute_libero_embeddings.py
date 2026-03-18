#!/usr/bin/env python3
"""
Precompute vision embeddings for LIBERO episodes and save to episodes/{id:06d}/embeddings.npz.

Requires the converted dataset (manifest.parquet + episodes/ with primary.mp4, wrist.mp4).
Uses the same encoder as training (e.g. DINOv2) so training can load embeddings with
data.use_precomputed_embeddings=true and skip the vision encoder.

Usage:
  python scripts/precompute_libero_embeddings.py --data-dir datasets
  python scripts/precompute_libero_embeddings.py --data-dir datasets --encoder dinov2 --batch-size 8 --max-episodes 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))


def _read_mp4_frames(path: Path):
    """Read all frames from MP4 as list of (H,W,3) uint8."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame[:, :, ::-1])
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


def main():
    parser = argparse.ArgumentParser(
        description="Precompute vision embeddings for LIBERO episodes (saves to episodes/{id}/embeddings.npz)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Data root containing LIBERO-Cosmos-Policy/ (manifest + episodes/)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2",
        choices=["dinov2", "dinov3"],
        help="Vision encoder to use (default: dinov2)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace model name (default: facebook/dinov2-base or dinov3 equivalent)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images per encoder forward (default: 8)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episodes to process (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for encoder (default: cuda:0)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("H", "W"),
        help="Resize frames to HxW (default: 224 224 for DINOv2)",
    )
    args = parser.parse_args()

    import torch
    import pandas as pd
    from tqdm import tqdm

    root = Path(args.data_dir).resolve()
    if (root / "LIBERO-Cosmos-Policy").is_dir():
        root = root / "LIBERO-Cosmos-Policy"
    manifest_path = root / "manifest.parquet"
    episodes_dir = root / "episodes"
    if not manifest_path.is_file() or not episodes_dir.is_dir():
        print("Requires manifest.parquet and episodes/. Run convert_libero_hdf5_to_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(manifest_path)
    episode_ids = df["episode_id"].tolist()
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    h, w = args.resize[0], args.resize[1]
    model_name = args.model_name or (
        "facebook/dinov2-base" if args.encoder == "dinov2"
        else "facebook/dinov3-vits16-pretrain-lvd1689m"
    )

    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(model_name).to(device)
    backbone.eval()
    hidden_size = backbone.config.hidden_size
    num_views = 2
    output_dim = num_views * hidden_size

    def encode_frames(images_list, chunk_size):
        """images_list: list of 2 arrays each (T, H, W, 3). Return (T, output_dim)."""
        views = []
        for v in range(num_views):
            if v >= len(images_list):
                break
            arr = images_list[v]
            if arr.size == 0:
                continue
            T, H, W, C = arr.shape
            if (H, W) != (h, w):
                import cv2
                resized = np.stack([cv2.resize(arr[t], (w, h)) for t in range(T)], axis=0)
            else:
                resized = arr
            x = torch.from_numpy(resized).float().to(device)
            x = x.permute(0, 3, 1, 2)
            x = x / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x = (x - mean) / std
            out_list = []
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                chunk = x[start:end]
                with torch.no_grad():
                    h_out = backbone(pixel_values=chunk).last_hidden_state
                cls_token = h_out[:, 0]
                out_list.append(cls_token.cpu().numpy())
            views.append(np.concatenate(out_list, axis=0))
        if len(views) == 0:
            return None
        while len(views) < num_views:
            views.append(views[-1])
        return np.concatenate(views, axis=-1)

    for ep_id in tqdm(episode_ids, desc="Precomputing embeddings", unit="ep"):
        ep_dir = episodes_dir / f"{ep_id:06d}"
        primary_path = ep_dir / "primary.mp4"
        wrist_path = ep_dir / "wrist.mp4"
        lowdim_path = ep_dir / "lowdim.npz"
        out_path = ep_dir / "embeddings.npz"
        if out_path.is_file():
            continue
        T_expected = None
        if lowdim_path.is_file():
            lowdim = np.load(lowdim_path, allow_pickle=False)
            T_expected = lowdim["proprio"].shape[0]
        images_list = []
        if primary_path.is_file():
            frames = _read_mp4_frames(primary_path)
            if frames:
                images_list.append(np.stack(frames, axis=0))
        if wrist_path.is_file():
            frames = _read_mp4_frames(wrist_path)
            if frames:
                images_list.append(np.stack(frames, axis=0))
        if not images_list:
            continue
        T = images_list[0].shape[0]
        if T_expected is not None and T != T_expected:
            T = min(T, T_expected)
            images_list = [v[:T] for v in images_list]
        emb = encode_frames(images_list, args.batch_size)
        if emb is not None:
            np.savez_compressed(out_path, embeddings=emb.astype(np.float32))

    print(f"Done. Embeddings saved under {episodes_dir} (embeddings.npz per episode).", flush=True)
    print("Train with: data.use_precomputed_embeddings=true", flush=True)


if __name__ == "__main__":
    main()
