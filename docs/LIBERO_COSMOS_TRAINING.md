# LIBERO-Cosmos-Policy: training and in-distribution evaluation

This doc covers **downloading**, **training**, and **evaluation** for [nvidia/LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy) (HuggingFace). Training is ICRT-style (in-context trajectories + language); evaluation is **offline** on a held-out set, with metrics **per suite** (in-distribution: libero_spatial, libero_object, libero_goal, libero_10).

## 1. Prerequisites

- **uv** (or pip)
- **datasets** (HuggingFace): `pip install datasets` or `uv sync --extra icrt`

## 2. Download and prepare data

1. **Download HDF5** with the HuggingFace CLI ([dataset](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy)):

```bash
huggingface-cli download nvidia/LIBERO-Cosmos-Policy --repo-type dataset --local-dir datasets/LIBERO-Cosmos-Policy
```

This creates `datasets/LIBERO-Cosmos-Policy/all_episodes/*.hdf5` (one file per episode).

2. **Convert** to the recommended layout (one episode per folder, MP4 + NPZ, Parquet manifest and sample index):

```bash
uv run python scripts/convert_libero_hdf5_to_dataset.py --input-dir datasets/LIBERO-Cosmos-Policy
```

**Output layout** (preferred by the loader):

- **episodes/{episode_id:06d}/** — one folder per episode:
  - `primary.mp4`, `wrist.mp4` — image streams (one file per camera)
  - `lowdim.npz` — proprio, actions, dones, rewards
- **manifest.parquet** — episode_id, task_description, success, n_steps, paths
- **sample_index.parquet** — precomputed (query_episode_id, query_start, query_len, prompt_episode_ids[], prompt_starts[], prompt_lens[]) for in-context learning; no same-task search at training time.

Optional: `--horizon 32 --num-context 3 --max-prompt-steps 85` to match your training config. Requires **pandas**, **imageio** (and **imageio-ffmpeg** for MP4). The loader prefers this format and falls back to legacy **data/** (HuggingFace Arrow) if manifest.parquet is missing.

## 3. Train

```bash
uv run python -m src.train \
  --override data=[base,libero_cosmos] \
  experiment.eval_every_steps=5000 \
  experiment.max_steps=100000 \
  --run-name libero_cosmos_run
```

- `data=[base,libero_cosmos]` merges `configs/data/base.yaml` with `configs/data/libero_cosmos.yaml` (state_dim=9, act_dim=7, language, etc.).
- Training loads trajectories from `data_dir/LIBERO-Cosmos-Policy/` (manifest + episodes/ or legacy data/) and writes checkpoints under `outputs/icl_adaptation/<date>/<run_name>__seed_<X>__<hash>/`.
- At startup, **dataset stats** print observation keys: **Image keys (config)** = `primary_images_jpeg`, `wrist_images_jpeg` (primary + wrist camera in Hub/Parquet) and **Proprio keys (config)** = `proprio`. The current HDF5 loader provides proprio only; use Parquet or an image-enabled loader for vision.

**Eval rollouts:** `env_name` is the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark suite: **libero_10**, **libero_spatial**, **libero_object**, **libero_goal**, or **libero_90**. Eval rollouts need the `libero` package installed so we can create the sim env. Without it you get `ModuleNotFoundError: No module named 'libero'` at eval; you can still use offline eval via `scripts/run_libero_eval.py` (see §5) or set `experiment.eval_every_steps=0` to skip rollouts.

**Installing LIBERO (sim for eval rollouts):** Clone LIBERO **inside** this project so the code can add it to `sys.path` and import it when you run from the project root:

```bash
# 1. Clone LIBERO into this repo (so you have icl_adaptation/LIBERO/)
cd /path/to/icl_adaptation
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# 2. Install our libero extra (robosuite, bddl, h5py). No need to pip install LIBERO.
uv sync --extra libero
```

When you run `uv run python -m src.train ...` from `icl_adaptation`, the LIBERO path is set so `import libero` finds `LIBERO/libero/`. If you see `ModuleNotFoundError: No module named 'robosuite.environments.manipulation.single_arm_env'`, re-sync so robosuite 1.4.x is installed: `uv sync --extra libero` (we pin `robosuite>=1.4.0,<1.5` for LIBERO compatibility).

To **disable** env-based evaluation (no rollouts, same as ICRT-MT):

```bash
uv run python -m src.train \
  --override data=[base,libero_cosmos] experiment.eval_every_steps=0 \
  --run-name libero_cosmos_run
```

To **save eval rollout videos** (when using an env that supports rendering, e.g. MuJoCo), set `experiment.save_eval_video=true`. Videos are written to `run_dir/viz/samples/step_XXXXX/videos/` (one per rollout). Requires Gymnasium or gym with `RecordVideo` (or `Monitor`) available.

To **plot predicted vs GT actions** (ICRT [eval_plot](https://github.com/Max-Fu/icrt/blob/main/scripts/eval_plot.py)-style) on a few trajectories at each eval step, set `experiment.run_action_compare_eval=true`. Plots are saved to `run_dir/viz/action_compare/step_XXXXX/demo_*/action_dim_*.png` and metrics `eval/action_mse_mean`, `eval/action_mse_std` are logged. Use `experiment.num_action_compare_demos` to choose how many demos to plot (default 3).

## 4. Model variants (vision encoder backbones)

The default **libero_cosmos** config is **state-only** (`use_vision: false`). When you use a vision-enabled setup (e.g. a dataset that provides images and `data.use_vision=true`, `model=vla_dt`), you can switch the vision backbone with overrides.

**Vision encoder** (`model.vision_encoder_type`):

| Type        | Description |
|------------|-------------|
| **patch**  | (default) Trainable per-patch CNN, pooled per view then concatenated. |
| **crossmae** | ViT per-patch (CrossMAE); use with `model.vision_encoder_attention_pool=true` for one token per view. |
| **dinov2** / **dinov3** | DINOv2/DINOv3; one embedding per image (CLS) or patch + attention pooling. |
| **paligemma** | PaliGemma-style SigLIP vision tower; pooled. |

**Attention pooling (ICRT-style):** Set `model.vision_encoder_attention_pool=true` to compress patch tokens with a learned query (one state token per view per timestep). Recommended for **crossmae** and optionally **dinov2**.

Examples (use with a vision-enabled data config, e.g. one that sets `use_vision: true` and provides `image_keys`):

**Patch (default VLA-DT):**
```bash
uv run python -m src.train \
  --override data=[base,libero_cosmos] model=vla_dt data.use_vision=true \
  model.vision_encoder_type=patch --run-name libero_patch_run
```

**DINOv2 / DINOv3** (defaults: `facebook/dinov2-base`, `facebook/dinov3-vits16-pretrain-lvd1689m`):
```bash
uv run python -m src.train \
  --override data=[base,libero_cosmos] model=vla_dt data.use_vision=true \
  model.vision_encoder_type=dinov2 --run-name libero_dinov2_run

uv run python -m src.train \
  --override data=[base,libero_cosmos] model=vla_dt data.use_vision=true \
  model.vision_encoder_type=dinov3 --run-name libero_dinov3_run
```

**CrossMAE with attention pooling:**
```bash
uv run python -m src.train \
  --override data=[base,libero_cosmos] model=vla_dt data.use_vision=true \
  model.vision_encoder_type=crossmae model.vision_encoder_attention_pool=true \
  --run-name libero_crossmae_run
```

**PaliGemma-style SigLIP:**
```bash
uv run python -m src.train \
  --override data=[base,libero_cosmos] model=vla_dt data.use_vision=true \
  model.vision_encoder_type=paligemma --run-name libero_paligemma_run
```

Note: LIBERO-Cosmos as shipped does not load images; the dataset uses proprio only. The above examples assume a vision-enabled data path (e.g. after adding `image_keys` and vision fields to the LIBERO config, or when using the same overrides with another vision dataset).

## 5. Run in-distribution evaluation (held-out)

After training, run **offline** eval (no simulator). Val episodes = last 10% of episodes in the dataset (by `episode_index`):

```bash
uv run python scripts/run_libero_eval.py \
  --ckpt outputs/icl_adaptation/<date>/libero_cosmos_run__seed_412__<hash>/ckpts/best/checkpoint.pt \
  --data-dir datasets
```

Optional:

- `--output-dir <path>` – write metrics here (default: `<run_dir>/eval`).
- `--max-val-episodes 500` – cap number of val episodes.

Output:

- **`<run_dir>/eval/libero_eval_metrics.json`** (or under `--output-dir`) with:
  - `eval/action_mse_mean`, `eval/action_mse_std`, `eval/success_rate`, `eval/num_episodes`
  - **`by_suite`**: for each of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`:
    - `action_mse_mean`, `action_mse_std`, `num_episodes`, `success_rate`

Success rate is the fraction of val episodes that were successful (from dataset labels); action MSE is teacher-forced prediction error on held-out data.

## 6. Summary

| Step        | Command |
|------------|--------|
| Install    | `uv sync --extra icrt` (or `pip install datasets`) |
| Download   | `huggingface-cli download nvidia/LIBERO-Cosmos-Policy --repo-type dataset --local-dir datasets/LIBERO-Cosmos-Policy` |
| Convert    | `uv run python scripts/convert_libero_hdf5_to_dataset.py --input-dir datasets/LIBERO-Cosmos-Policy` |
| Train      | `uv run python -m src.train --override data=[base,libero_cosmos] --run-name libero_cosmos_run` |
| Train (VLA + vision backbone) | `uv run python -m src.train --override data=[base,libero_cosmos] model=vla_dt data.use_vision=true model.vision_encoder_type=dinov2 --run-name libero_dinov2_run` (see §4 for patch / dinov2 / dinov3 / crossmae / paligemma) |
| Eval (ID)  | `uv run python scripts/run_libero_eval.py --ckpt <path-to-ckpt> --data-dir datasets` |

Everything is **modular**: `scripts/convert_libero_hdf5_to_dataset.py`, `src/data/libero_dataset.py`, config `configs/data/libero_cosmos.yaml`, and `scripts/run_libero_eval.py` can be run and read independently.
