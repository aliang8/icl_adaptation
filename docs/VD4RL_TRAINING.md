# V-D4RL (Offline DreamerV2) Training

This repo supports V-D4RL via the `VD4RL` loader (`src/data/vd4rl_loader.py`).
It reads pixel trajectories stored as `*.npz` and converts them into the trajectory dict format expected by `ICLTrajectoryDataset`.

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
<vd4rl_data>/
  main/
    walker_walk/
      random/
        64px/
          *.npz
    cheetah_run/
      random/
        64px/
          *.npz
    ...
```

## 2) Point `paths.data_root` at the parent folder

Set `paths.data_root` to `<vd4rl_data>`, i.e. the directory that contains `main/`, `distracting/`, or `multitask/`.

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
- `train.py` infers `state_dim` and `action_dim` from the first loaded `*.npz`, so the `state_dim/act_dim` values inside `model=vd4rl_dt` are not critical.
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

Key config fields:
- `data.vd4rl_suite`: `main` | `distracting` | `multitask`
- `data.vd4rl_task`: e.g. `walker_walk`, `cheetah_run`, `humanoid_walk`
- `data.vd4rl_split`: e.g. `random`, `medium_replay`, `medium`, `medium_expert`, `expert`
- `data.vd4rl_pixel_size`: `64px` or `84px`

## 5) Observations downsampling

V-D4RL images are converted into flattened observations by downsampling:
- `data.vd4rl_obs_downsample` (default: `16`)

This affects the observation dimensionality and thus the effective `state_dim` used by the model (again inferred at runtime).

