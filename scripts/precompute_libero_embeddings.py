#!/usr/bin/env python3
"""
Precompute vision embeddings for LIBERO episodes and save to episodes/{id:06d}/embeddings.npz.

Requires the converted dataset (manifest.parquet + episodes/ with primary.mp4, wrist.mp4).
Both views are required; episodes missing either view are skipped and an error is printed.
Encodes both views in order [primary, wrist]; output shape (T, 2*hidden_size).
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


from src.utils import read_mp4_frames


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all episodes even if embeddings.npz already exists",
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
        print(
            "Requires manifest.parquet and episodes/. Run convert_libero_hdf5_to_dataset.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_parquet(manifest_path)
    episode_ids = df["episode_id"].tolist()
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    h, w = args.resize[0], args.resize[1]
    model_name = args.model_name or (
        "facebook/dinov2-base"
        if args.encoder == "dinov2"
        else "facebook/dinov3-vits16-pretrain-lvd1689m"
    )

    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(model_name).to(device)
    backbone.eval()
    hidden_size = backbone.config.hidden_size
    # Always output both views: [primary, wrist]. Order must match training (libero primary=0, wrist=1).
    num_views = 2
    output_dim = num_views * hidden_size

    def encode_one_view(frames_arr, chunk_size):
        """Encode one view (T, H, W, 3) -> (T, hidden_size). Returns zeros if empty."""
        if frames_arr is None or frames_arr.size == 0:
            return None
        T, H, W, C = frames_arr.shape
        if (H, W) != (h, w):
            import cv2

            resized = np.stack([cv2.resize(frames_arr[t], (w, h)) for t in range(T)], axis=0)
        else:
            resized = frames_arr
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
        return np.concatenate(out_list, axis=0)

    def encode_frames(primary_frames, wrist_frames, chunk_size):
        """Encode both views in order [primary, wrist]. Both views required."""
        if primary_frames is None or primary_frames.size == 0:
            raise ValueError("Primary view (primary.mp4) is missing or empty")
        if wrist_frames is None or wrist_frames.size == 0:
            raise ValueError("Wrist view (wrist.mp4) is missing or empty")
        T = primary_frames.shape[0]
        if wrist_frames.shape[0] != T:
            raise ValueError(
                f"Primary and wrist frame count mismatch: primary={T}, wrist={wrist_frames.shape[0]}"
            )
        views = [
            encode_one_view(primary_frames, chunk_size),
            encode_one_view(wrist_frames, chunk_size),
        ]
        return np.concatenate(views, axis=-1)

    for ep_id in tqdm(episode_ids, desc="Precomputing embeddings", unit="ep"):
        ep_dir = episodes_dir / f"{ep_id:06d}"
        primary_path = ep_dir / "primary.mp4"
        wrist_path = ep_dir / "wrist.mp4"
        lowdim_path = ep_dir / "lowdim.npz"
        out_path = ep_dir / "embeddings.npz"
        if out_path.is_file() and not args.force:
            continue
        T_expected = None
        if lowdim_path.is_file():
            lowdim = np.load(lowdim_path, allow_pickle=False)
            T_expected = lowdim["proprio"].shape[0]
        primary_frames = None
        wrist_frames = None
        if primary_path.is_file():
            frames = read_mp4_frames(primary_path)
            if frames:
                primary_frames = np.stack(frames, axis=0)
        if wrist_path.is_file():
            frames = read_mp4_frames(wrist_path)
            if frames:
                wrist_frames = np.stack(frames, axis=0)
        if primary_frames is None and wrist_frames is None:
            continue
        try:
            T = primary_frames.shape[0] if primary_frames is not None else wrist_frames.shape[0]
            if T_expected is not None and T != T_expected:
                T = min(T, T_expected)
                if primary_frames is not None:
                    primary_frames = primary_frames[:T]
                if wrist_frames is not None:
                    wrist_frames = wrist_frames[:T]
            emb = encode_frames(primary_frames, wrist_frames, args.batch_size)
        except ValueError as e:
            print(f"Episode {ep_id}: {e}", file=sys.stderr)
            continue
        if emb is not None:
            np.savez_compressed(out_path, embeddings=emb.astype(np.float32))

    print(f"Done. Embeddings saved under {episodes_dir} (embeddings.npz per episode).", flush=True)
    print("Train with: data.use_precomputed_embeddings=true", flush=True)


if __name__ == "__main__":
    main()
