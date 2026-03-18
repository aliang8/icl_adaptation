"""
ICRT-style dataset: HDF5 with verb_to_episode (task name -> episodes), multi-view images, proprio, actions.

Dataset config JSON (see ICRT DATASET.md):
  dataset_path, hdf5_keys, epi_len_mapping_json, verb_to_episode,
  image_keys, proprio_keys, action_keys.

Reference: https://github.com/Max-Fu/icrt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def load_dataset_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load ICRT dataset_config.json. config_path can be str or Path."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"ICRT dataset config not found: {path}")
    with path.open() as f:
        return json.load(f)


def _resolve_config_paths(config: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    """Resolve relative paths in config relative to config_dir (config file's parent)."""
    out = dict(config)
    config_dir = config_dir.resolve()
    for key in ("dataset_path", "hdf5_keys", "epi_len_mapping_json", "verb_to_episode"):
        if key not in out:
            continue
        val = out[key]
        if isinstance(val, str):
            p = Path(val)
            if not p.is_absolute():
                out[key] = str((config_dir / p).resolve())
        elif isinstance(val, list):
            out[key] = [
                str((config_dir / Path(p)).resolve()) if not Path(p).is_absolute() else p
                for p in val
            ]
    return out


def load_verb_to_episode(verb_to_episode_path: str) -> Dict[str, List[str]]:
    """Load verb_to_episode: task name -> list of episode keys."""
    with open(verb_to_episode_path) as f:
        return json.load(f)


def load_epi_len_mapping(epi_len_mapping_path: str) -> Dict[str, int]:
    """Load episode key -> length mapping."""
    with open(epi_len_mapping_path) as f:
        return json.load(f)


def get_task_instructions_from_verbs(verb_to_episode: Dict[str, List[str]]) -> List[str]:
    """Return list of task instruction strings (one per task) in a fixed order."""
    return list(verb_to_episode.keys())


def load_hdf5_keys(hdf5_keys_paths: List[str]) -> List[str]:
    """Load and merge hdf5 key lists from one or more JSON files."""
    keys = []
    for p in hdf5_keys_paths:
        with open(p) as f:
            keys.extend(json.load(f))
    return keys


def open_icrt_hdf5(config_path: Union[str, Path]):
    """
    Open HDF5 file(s) from dataset config. Returns (list of h5py.File, keys_to_file map).
    Caller must close the files when done. Paths in config are resolved relative to config file's parent.
    """
    import h5py

    path = Path(config_path).resolve()
    config = load_dataset_config(path)
    config = _resolve_config_paths(config, path.parent)
    dataset_paths = config["dataset_path"]
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]
    hdf5_keys = config["hdf5_keys"]
    if isinstance(hdf5_keys, str):
        hdf5_keys = [hdf5_keys]

    for p in dataset_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"ICRT HDF5 file not found: {p} (from {path})")
    for k in hdf5_keys:
        if not Path(k).exists():
            raise FileNotFoundError(f"ICRT HDF5 keys file not found: {k} (from {path})")

    files = [h5py.File(p, "r") for p in dataset_paths]
    key_lists = [json.load(open(k)) for k in hdf5_keys]
    keys_to_file = {}
    for f, klist in zip(files, key_lists):
        for k in klist:
            keys_to_file[k] = f
    return files, keys_to_file


def _key_in_group(grp, key: str):
    """Resolve key in group: try full key then suffix after last '/' (e.g. observation/cartesian_position -> cartesian_position)."""
    if key in grp:
        return grp[key]
    short = key.split("/")[-1] if "/" in key else key
    if short in grp:
        return grp[short]
    return None


def read_episode_obs_act(
    h5file,
    episode_key: str,
    image_keys: List[str],
    proprio_keys: List[str],
    action_keys: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[str]]:
    """
    Read one episode: observations (images + proprio) and actions.
    Returns (obs_dict, action_dict, language_instruction or None).
    Keys can be full path (e.g. observation/cartesian_position) or short name; both are tried.
    """
    grp = h5file[episode_key]
    obs_grp = grp.get("observation", grp)
    act_grp = grp.get("action", grp)

    obs = {}
    for k in image_keys:
        ds = _key_in_group(obs_grp, k)
        if ds is not None:
            buf = ds[:]
            if buf.dtype == np.uint8 and buf.ndim == 1:
                buf = np.frombuffer(buf.tobytes(), dtype=np.uint8)
            obs[k] = buf
    for k in proprio_keys:
        ds = _key_in_group(obs_grp, k)
        if ds is not None:
            obs[k] = np.array(ds[:], dtype=np.float32)

    actions = {}
    for k in action_keys:
        ds = _key_in_group(act_grp, k)
        if ds is not None:
            actions[k] = np.array(ds[:], dtype=np.float32)

    lang = None
    for name in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        if name in grp:
            val = grp[name]
            if hasattr(val, "asstr"):
                lang = val.asstr()[()]
            else:
                lang = str(val[()])
            break

    return obs, actions, lang


def _build_verb_to_episode_from_keys(all_keys: List[str]) -> Dict[str, List[str]]:
    """Build verb_to_episode from key list: group by task (strip date and _N suffix)."""
    by_task: Dict[str, List[str]] = {}
    for k in all_keys:
        # e.g. "real_episode_2024-05-31-close-drawer_0" -> task "close-drawer"
        if "real_episode_" in k:
            rest = k.replace("real_episode_", "")
            # drop trailing _N
            name = rest.rsplit("_", 1)[0] if "_" in rest else rest
            # name = "2024-05-31-close-drawer"; task = part after date
            parts = name.split("-", 3)
            task = parts[-1] if len(parts) >= 4 else name
        else:
            task = k.rsplit("_", 1)[0] if "_" in k else k
        by_task.setdefault(task, []).append(k)
    return by_task


def load_icrt_trajectories(
    config_path: str,
    proprio_keys: List[str],
    action_keys: List[str],
    min_trajectory_length: int = 30,
    max_trajectory_length: int = 450,
    max_episodes: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], List[str]]:
    """
    Load ICRT HDF5 into (trajectories, prompt_trajectories_per_task, task_instructions).
    Each trajectory has observations (proprio only, for ICLTrajectoryDataset), actions, rewards, terminals.
    """
    import h5py

    path = Path(config_path).resolve()
    config = load_dataset_config(path)
    config = _resolve_config_paths(config, path.parent)
    dataset_paths = config.get("dataset_path")
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]
    hdf5_keys_paths = config.get("hdf5_keys")
    if isinstance(hdf5_keys_paths, str):
        hdf5_keys_paths = [hdf5_keys_paths]

    all_keys = []
    for p in hdf5_keys_paths:
        with Path(p).open() as f:
            all_keys.extend(json.load(f))

    verb_to_episode_path = config.get("verb_to_episode")
    if verb_to_episode_path and Path(verb_to_episode_path).is_file():
        verb_to_episode = load_verb_to_episode(verb_to_episode_path)
    else:
        verb_to_episode = _build_verb_to_episode_from_keys(all_keys)

    task_instructions = get_task_instructions_from_verbs(verb_to_episode)
    files, keys_to_file = open_icrt_hdf5(config_path)
    n_episodes = sum(len(v) for v in verb_to_episode.values())
    n_in_file = sum(1 for keys in verb_to_episode.values() for ek in keys if ek in keys_to_file)
    skipped_no_key = 0
    skipped_no_obs = 0
    skipped_no_act = 0
    skipped_length = 0
    try:
        trajectories = []
        task_to_trajs: Dict[str, List[Dict]] = {t: [] for t in task_instructions}

        for task_name, episode_keys in verb_to_episode.items():
            for ep_key in episode_keys:
                if ep_key not in keys_to_file:
                    skipped_no_key += 1
                    continue
                if max_episodes and len(trajectories) >= max_episodes:
                    break
                h5file = keys_to_file[ep_key]
                obs_dict, action_dict, lang = read_episode_obs_act(
                    h5file, ep_key, config.get("image_keys", []), proprio_keys, action_keys
                )
                # Stack proprio in key order -> observations (T, state_dim)
                obs_list = []
                for k in proprio_keys:
                    if k in obs_dict:
                        arr = np.asarray(obs_dict[k], dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        obs_list.append(arr)
                if not obs_list:
                    skipped_no_obs += 1
                    continue
                observations = np.concatenate(obs_list, axis=-1)
                act_list = []
                for k in action_keys:
                    if k in action_dict:
                        arr = np.asarray(action_dict[k], dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        act_list.append(arr)
                if not act_list:
                    skipped_no_act += 1
                    continue
                actions = np.concatenate(act_list, axis=-1)
                T = observations.shape[0]
                if T != actions.shape[0]:
                    T = min(observations.shape[0], actions.shape[0])
                    observations = observations[:T]
                    actions = actions[:T]
                if T < min_trajectory_length or T > max_trajectory_length:
                    skipped_length += 1
                    continue
                rewards = np.zeros(T, dtype=np.float32)
                rewards[-1] = 1.0
                terminals = np.zeros(T, dtype=np.float32)
                terminals[-1] = 1.0
                traj = {
                    "observations": observations,
                    "actions": actions,
                    "rewards": rewards,
                    "terminals": terminals,
                    "task_description": lang or task_name,
                }
                trajectories.append(traj)
                if task_name in task_to_trajs:
                    task_to_trajs[task_name].append(traj)
            if max_episodes and len(trajectories) >= max_episodes:
                break

        prompt_per_task = [task_to_trajs.get(ti, [])[:5] for ti in task_instructions]
        if not any(prompt_per_task):
            prompt_per_task = [trajectories[:5]] * max(1, len(task_instructions))
        if not trajectories:
            raise ValueError(
                f"No ICRT trajectories loaded (config: {config_path}). "
                f"Episodes in verb_to_episode: {n_episodes}, in keys_to_file: {n_in_file}. "
                f"Skipped: no_key={skipped_no_key}, no_proprio={skipped_no_obs}, no_action={skipped_no_act}, length_filter={skipped_length} "
                f"(min_len={min_trajectory_length}, max_len={max_trajectory_length}). "
                "Check HDF5 structure (observation/action groups and key names) and length limits."
            )
        return trajectories, prompt_per_task, task_instructions
    finally:
        for f in files:
            f.close()
