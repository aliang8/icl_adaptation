# V-D4RL (Offline DreamerV2) Training

This repo supports V-D4RL via the `VD4RL` loader (`src/data/vd4rl_loader.py`).
It reads **64px** runs from `*.npz` (DreamerV2 / offline DV2) and **84px** runs from `*.hdf5` (DrQ-v2 / BC-CQL), matching the upstream note that each codebase uses its native format.

Upstream dataset repo: [v-d4rl](https://github.com/conglu1997/v-d4rl).

## 1) Download the dataset

The V-D4RL datasets are here on Google Drive:
[https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI](https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI)

You can download the entire folder with `gdown`:

```bash
# install once (uv-based)
uv pip install gdown

# download folder contents
gdown --folder "https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI" -O /path/to/vd4rl_data
```

Notes:
- `gdown --folder` recursively downloads files in the folder. For very large folders, you may need to rerun if downloads partially fail.
  Reference: [gdown](https://pypi.org/project/gdown/) and [Baeldung folder download examples](https://www.baeldung.com/linux/google-drive-download-folder-shell).

Upstream reference:
[https://github.com/conglu1997/v-d4rl](https://github.com/conglu1997/v-d4rl)

The code expects the following layout (as in the upstream README):

```text
<vd4rl_data>/                    # use this as paths.data_root (same level as upstream README)
  main/
    walker_walk/
      random/
        64px/
          *.npz
        84px/
          *.hdf5
    ...
  distracting/
    cheetah_medium_expert/
      84px/
        easy/                    # or medium/, hard/ — set data.vd4rl_split accordingly
          shard_*_reward_*.hdf5
    ...
  multitask/
    ...
```

## 2) Point `paths.data_root` at the parent folder

Set `paths.data_root` to `<vd4rl_data>`: the directory whose **immediate children** are `main/`, `distracting/`, `multitask/` (not the parent that only contains a `vd4rl/` subfolder unless you set `data_root` to that `vd4rl` folder itself).

Example: if data lives at `/scr2/shared/.../datasets/vd4rl/distracting/cheetah_medium_expert/84px/easy/`, then `paths.data_root=/.../datasets/vd4rl`, `vd4rl_suite=distracting`, `vd4rl_task=cheetah_medium_expert`, `vd4rl_split=easy`, `vd4rl_pixel_size=84px`.

The default V-D4RL config uses:
- `data.vd4rl_suite=main`
- `data.vd4rl_task=walker_walk`
- `data.vd4rl_split=random`
- `data.vd4rl_pixel_size=64px`

## 3) Run training (Walker Walk / random)

Example (turns on the vision DT model and uses the included V-D4RL data config):

```bash
uv run python -m src.train \
  --override data=[base,vd4rl_walker_random] \
  --override model=vd4rl_dt \
  paths.data_root=/path/to/<vd4rl_data>
```

Notes:
- `train.py` infers `state_dim` and `action_dim` from the first loaded trajectory, so the `state_dim/act_dim` values inside `model=vd4rl_dt` are not critical.
- **84px / `*.hdf5`:** install `h5py` (`uv pip install h5py` or `uv sync --extra icrt`). The loader splits shards on `step_type == FIRST` like upstream `drqbc/utils.py`.
- At the start of training the code will (by default) save a few MP4s for debugging under:
  - `outputs/<...>/viz/training_sample_debug/`
  - controlled by `experiment.save_training_sample_videos`, `experiment.num_training_sample_videos`, and `experiment.training_sample_video_fps`.

### State-only training command (no vision encoder)

If you want to train with state vectors only (no image encoder path), use the base transformer model and disable vision:

```bash
uv run python -m src.train \
  --override data=[base,vd4rl_walker_random] \
  --override model=transformer \
  --override data.use_vision=false \
  paths.data_root=/path/to/<vd4rl_data>
```

This still uses V-D4RL `*.npz` files, but trains with the flattened observation vectors produced by the loader (instead of feeding image frames into the vision encoder).

## 4) Switch to another task/split/pixel size

You can reuse the same config and override the nested fields:

```bash
uv run python -m src.train \
  --override data=[base,vd4rl_walker_random] \
  --override model=vd4rl_dt \
  --override data.vd4rl_task=cheetah_run \
  --override data.vd4rl_split=medium_replay \
  --override data.vd4rl_pixel_size=84px \
  paths.data_root=/path/to/<vd4rl_data>
```

### Merging multiple splits

Set `data.vd4rl_splits` to a YAML list (e.g. `random`, `medium_replay`). Each leaf directory is loaded and trajectories are concatenated; `data.vd4rl_split` is ignored when `vd4rl_splits` is non-empty. With a **single** split, `vd4rl_max_episodes` still limits loaded files as before; with **multiple** splits, all episodes from each split are loaded first, then the merged list is truncated to `vd4rl_max_episodes` if set.

### Eval (dm_control Walker-walk, not Gymnasium Walker2d)

Offline V-D4RL training uses **dm_control** tasks ([v-d4rl](https://github.com/conglu1997/v-d4rl)), not the same MDP as `Walker2d-v5` from Gymnasium MuJoCo. When `data.env_name=VD4RL`, eval rollouts default to **`VD4RL/dmc/<vd4rl_task>`** (e.g. `VD4RL/dmc/walker_walk`): `dm_control.suite` + pixel wrapper, with the same downsampling as `data.vd4rl_obs_downsample` and render size from `data.vd4rl_pixel_size`. Install **`uv sync --extra d4rl`** (adds `dm_control`). Override with `data.eval_env_name` (e.g. `VD4RL/dmc/cheetah_run`).

Key config fields:
- `data.vd4rl_suite`: `main` | `distracting` | `multitask`
- `data.vd4rl_task`: e.g. `walker_walk`, `cheetah_run`, `cheetah_medium_expert`, `humanoid_walk`
- `data.vd4rl_split`: single split when `vd4rl_splits` is unset.
- `data.vd4rl_splits`: optional list of splits to load and merge (overrides use of `vd4rl_split` for loading).
- `data.vd4rl_pixel_size`: `64px` (npz) or `84px` (hdf5)

## 5) Observations downsampling

V-D4RL images are converted into flattened observations by downsampling:
- `data.vd4rl_obs_downsample` (default: `16`)

This affects the observation dimensionality and thus the effective `state_dim` used by the model (again inferred at runtime).

