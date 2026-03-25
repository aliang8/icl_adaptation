#!/usr/bin/env python3
"""
Convert RoboArena DataDump (evaluation_sessions layout) → episode folders + Parquet manifest.

Input layout (verified on DataDump_08-05-2025):
  evaluation_sessions/<session_id>/
    - metadata.yaml   (language_instruction, policies.A/B/....policy_name, binary_success)
    - A_<policy_name>/, B_<policy_name>/, ...
        - <policy_name>_<timestamp>_npz_file.npz   (single key "data": (T,) object array of dicts)
        - <policy_name>_<timestamp>_video_left.mp4
        - <policy_name>_<timestamp>_video_wrist.mp4

NPZ format: npz["data"] is shape (T,) dtype=object; each element is a dict:
  - cartesian_position: length 6
  - joint_position: length 7
  - gripper_position: length 1
  - action: length 8
Proprio = concat(cartesian_position, joint_position, gripper_position) per step → (T, 14).
Actions = (T, 8).

Output:
  episodes/<task_slug>/<idx:06d>/: primary.mp4, wrist.mp4, lowdim.npz
  manifest.parquet (includes success, partial_success; lowdim rewards[-1] = partial_success)

Usage:
  python scripts/convert_roboarena_to_dataset.py --input-dir datasets/DataDump_08-05-2025
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))


def _task_to_slug(task: str) -> str:
    s = (task or "unknown").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s or "unknown"


def _load_roboarena_npz(npz_path: Path):
    """
    Load RoboArena npz: single key "data", (T,) object array of dicts with
    cartesian_position (6), joint_position (7), gripper_position (1), action (8).
    Returns (proprio (T,14), actions (T,8)) as float32.
    """
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    if "data" not in data:
        raise ValueError(f"npz has keys {list(data.keys())}, expected 'data'")
    rows = data["data"]
    if not len(rows):
        return np.zeros((0, 14), dtype=np.float32), np.zeros((0, 8), dtype=np.float32)
    t0 = rows[0]
    if not isinstance(t0, dict):
        raise ValueError(f"data[0] is {type(t0).__name__}, expected dict")
    proprio_list = []
    action_list = []
    for t in rows:
        cp = np.asarray(t["cartesian_position"], dtype=np.float32)
        jp = np.asarray(t["joint_position"], dtype=np.float32)
        gp = np.asarray(t["gripper_position"], dtype=np.float32)
        proprio_list.append(np.concatenate([cp, jp, gp]))
        action_list.append(np.asarray(t["action"], dtype=np.float32))
    proprio = np.stack(proprio_list)
    actions = np.stack(action_list)
    return proprio, actions


def _read_n_steps(lowdim_path: Path) -> int:
    import numpy as np

    d = np.load(lowdim_path, allow_pickle=False)
    return int(np.asarray(d["proprio"]).shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RoboArena DataDump to episodes/{task}/{idx}/ + manifest"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="DataDump root or .../evaluation_sessions"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output root (default: same as input-dir)"
    )
    parser.add_argument("--symlink", action="store_true", help="Symlink videos instead of copying")
    parser.add_argument(
        "--skip-existing", action="store_true", default=True, help="Skip if lowdim.npz exists"
    )
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Print shapes and sample paths (default: on).",
    )
    parser.add_argument(
        "--no-debug", dest="debug", action="store_false", help="Disable debug prints."
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        sys.exit(f"Install dependencies: pip install numpy pandas. {e}")
    try:
        import yaml
    except ImportError:
        sys.exit("Install PyYAML: pip install pyyaml")
    from tqdm import tqdm

    input_root = Path(args.input_dir).resolve()
    sessions_dir = (
        input_root
        if input_root.name == "evaluation_sessions"
        else input_root / "evaluation_sessions"
    )
    if input_root.name == "evaluation_sessions":
        input_root = input_root.parent
    if not sessions_dir.is_dir():
        sys.exit(
            f"Expected evaluation_sessions/ under {input_root}. Pass DataDump root or .../evaluation_sessions"
        )

    output_root = Path(args.output_dir).resolve() if args.output_dir else input_root
    episodes_root = output_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)

    # Build episode list: one per (session, policy_dir). Each policy dir has one npz and matching videos.
    rows: list[dict] = []
    for session_path in sorted(sessions_dir.iterdir()):
        if not session_path.is_dir():
            continue
        meta_path = session_path / "metadata.yaml"
        if not meta_path.is_file():
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        task = (
            meta.get("language_instruction")
            or meta.get("task")
            or meta.get("task_name")
            or meta.get("instruction")
            or meta.get("task_id")
            or session_path.name
        )
        if not isinstance(task, str):
            task = str(task)
        policies_meta = meta.get("policies") or {}

        for policy_dir in sorted(session_path.iterdir()):
            if not policy_dir.is_dir():
                continue
            npz_files = list(policy_dir.glob("*_npz_file.npz"))
            if not npz_files:
                npz_files = list(policy_dir.glob("*.npz"))
            if not npz_files:
                continue
            npz_path = npz_files[0]
            # Prefix from npz stem: e.g. paligemma_fast_droid_2025_04_28_19_37_59_npz_file -> paligemma_fast_droid_2025_04_28_19_37_59
            stem = npz_path.stem
            if stem.endswith("_npz_file"):
                prefix = stem[: -len("_npz_file")]
            else:
                prefix = stem
            left_video = policy_dir / f"{prefix}_video_left.mp4"
            wrist_video = policy_dir / f"{prefix}_video_wrist.mp4"
            if not left_video.is_file():
                # Fallback: any *_video_left.mp4 in dir (if only one run)
                left_candidates = list(policy_dir.glob("*_video_left.mp4"))
                left_video = left_candidates[0] if len(left_candidates) == 1 else left_video
            if not wrist_video.is_file():
                wrist_candidates = list(policy_dir.glob("*_video_wrist.mp4"))
                wrist_video = wrist_candidates[0] if len(wrist_candidates) == 1 else wrist_video

            policy_letter = (
                policy_dir.name.split("_")[0] if "_" in policy_dir.name else policy_dir.name[:1]
            ) or "A"
            policy_info = policies_meta.get(policy_letter) or {}
            success = bool(policy_info.get("binary_success", policy_info.get("success", False)))
            partial_success = float(policy_info.get("partial_success", 0.0))

            rows.append(
                {
                    "session_path": session_path,
                    "policy_name": policy_dir.name,
                    "task": task,
                    "success": success,
                    "partial_success": partial_success,
                    "npz_path": npz_path,
                    "left_path": left_video if left_video.is_file() else None,
                    "wrist_path": wrist_video if wrist_video.is_file() else None,
                }
            )

    if not rows:
        session_dirs = [p for p in sorted(sessions_dir.iterdir()) if p.is_dir()]
        with_meta = sum(1 for p in session_dirs if (p / "metadata.yaml").is_file())
        with_npz = sum(
            1
            for p in session_dirs
            for sub in p.iterdir()
            if sub.is_dir() and list(sub.glob("*.npz"))
        )
        print(
            f"Found {len(session_dirs)} session dirs, {with_meta} with metadata.yaml, {with_npz} policy dirs with .npz.",
            flush=True,
        )
        if session_dirs:
            sample = next(
                (s for s in session_dirs if (s / "metadata.yaml").is_file()), session_dirs[0]
            )
            for sub in sample.iterdir():
                if sub.is_dir():
                    print(
                        f"  Sample policy dir {sub.name}: {[p.name for p in sub.iterdir()]}",
                        flush=True,
                    )
                    break
        sys.exit(1)

    rows.sort(key=lambda r: (r["task"], r["session_path"].name, r["policy_name"]))
    if args.debug:
        print(f"[debug] Found {len(rows)} episodes.", flush=True)
        if rows:
            r0 = rows[0]
            print(
                f"[debug] Sample: task={r0['task'][:50]}..., npz={r0['npz_path']}, left={r0.get('left_path')}, wrist={r0.get('wrist_path')}",
                flush=True,
            )

    task_count: dict[str, int] = {}
    episode_list: list[tuple[str, int, dict]] = []
    for r in rows:
        slug = _task_to_slug(r["task"])
        if slug == "unknown":
            import ipdb

            ipdb.set_trace()
            raise ValueError(
                "task slug is 'unknown'; language_instruction/task may be missing or empty in metadata.yaml. "
                "Inspect r['task'], r['session_path'] in ipdb."
            )
        idx = task_count.get(slug, 0)
        task_count[slug] = idx + 1
        episode_list.append((slug, idx, r))

    manifest_columns = [
        "episode_id",
        "task_description",
        "success",
        "partial_success",
        "n_steps",
        "primary_path",
        "wrist_path",
        "lowdim_path",
    ]
    manifest_rows = []
    global_episode_id = 0

    for task_slug, idx, r in tqdm(episode_list, desc="Converting episodes", unit="ep"):
        ep_dir = episodes_root / task_slug / f"{idx:06d}"
        lowdim_path = ep_dir / "lowdim.npz"

        if args.skip_existing and lowdim_path.is_file():
            rel = f"episodes/{task_slug}/{idx:06d}"
            manifest_rows.append(
                {
                    "episode_id": global_episode_id,
                    "task_description": r["task"],
                    "success": r["success"],
                    "partial_success": r["partial_success"],
                    "n_steps": _read_n_steps(lowdim_path),
                    "primary_path": f"{rel}/primary.mp4"
                    if (ep_dir / "primary.mp4").is_file()
                    else None,
                    "wrist_path": f"{rel}/wrist.mp4" if (ep_dir / "wrist.mp4").is_file() else None,
                    "lowdim_path": f"{rel}/lowdim.npz",
                }
            )
            global_episode_id += 1
            continue

        ep_dir.mkdir(parents=True, exist_ok=True)

        primary_dst = ep_dir / "primary.mp4"
        wrist_dst = ep_dir / "wrist.mp4"
        if r.get("left_path") and r["left_path"].is_file():
            if args.symlink and not primary_dst.exists():
                primary_dst.symlink_to(r["left_path"].resolve())
            elif not args.symlink:
                shutil.copy2(r["left_path"], primary_dst)
        if r.get("wrist_path") and r["wrist_path"].is_file():
            if args.symlink and not wrist_dst.exists():
                wrist_dst.symlink_to(r["wrist_path"].resolve())
            elif not args.symlink:
                shutil.copy2(r["wrist_path"], wrist_dst)

        proprio, actions = _load_roboarena_npz(r["npz_path"])
        T = min(len(proprio), len(actions))
        if args.debug and global_episode_id == 0:
            print(
                f"[debug] First episode: proprio.shape={proprio.shape}, actions.shape={actions.shape}, T={T}",
                flush=True,
            )
        proprio = np.asarray(proprio[:T], dtype=np.float32)
        actions = np.asarray(actions[:T], dtype=np.float32)
        dones = np.zeros(T, dtype=np.float32)
        if T > 0:
            dones[-1] = 1.0
        rewards = np.zeros(T, dtype=np.float32)
        if T > 0:
            rewards[-1] = r["partial_success"]
        np.savez_compressed(
            lowdim_path, proprio=proprio, actions=actions, dones=dones, rewards=rewards
        )

        rel = f"episodes/{task_slug}/{idx:06d}"
        manifest_rows.append(
            {
                "episode_id": global_episode_id,
                "task_description": r["task"],
                "success": r["success"],
                "partial_success": r["partial_success"],
                "n_steps": T,
                "primary_path": f"{rel}/primary.mp4" if primary_dst.is_file() else None,
                "wrist_path": f"{rel}/wrist.mp4" if wrist_dst.is_file() else None,
                "lowdim_path": f"{rel}/lowdim.npz",
            }
        )
        global_episode_id += 1

    manifest_df = (
        pd.DataFrame(manifest_rows) if manifest_rows else pd.DataFrame(columns=manifest_columns)
    )
    unique_tasks = sorted(manifest_df["task_description"].unique().tolist(), key=str)
    task_to_id = {t: i for i, t in enumerate(unique_tasks)}
    manifest_df["task_id"] = manifest_df["task_description"].map(task_to_id)
    manifest_path = output_root / "manifest.parquet"
    manifest_df.to_parquet(manifest_path, index=False)

    print(
        f"Saved manifest to {manifest_path} ({len(manifest_df)} episodes, {len(unique_tasks)} tasks)",
        flush=True,
    )
    print(
        f"Episodes under {episodes_root} (by task: {list(task_count.keys())[:10]}{'...' if len(task_count) > 10 else ''})",
        flush=True,
    )


if __name__ == "__main__":
    main()
