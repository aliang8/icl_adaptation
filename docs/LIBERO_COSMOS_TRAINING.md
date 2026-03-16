# LIBERO-Cosmos-Policy: training and in-distribution evaluation

This doc covers **downloading**, **training**, and **evaluation** for [nvidia/LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy) (HuggingFace). Training is ICRT-style (in-context trajectories + language); evaluation is **offline** on a held-out set, with metrics **per suite** (in-distribution: libero_spatial, libero_object, libero_goal, libero_10).

## 1. Prerequisites

- **uv** (or pip)
- **datasets** (HuggingFace): `pip install datasets` or `uv sync --extra icrt`

## 2. Download and prepare data

**Recommended:** Download the repo with the HuggingFace CLI so you get **HDF5** (one file per episode with task/success). Then run our script to build the manifest. See “Recommended: HDF5 layout” below.

If you skip the CLI and only run the script, it uses the Hub’s Parquet; that view has no per-episode metadata, so you may get a single long “episode.”

```bash
uv run python scripts/download_libero_cosmos.py --output-dir datasets
```

Options:

- `--split-fraction 0.9` – fraction of episodes per suite for train (default 0.9; rest for val).
- `--seed 42` – random seed for train/val split.
- `--streaming` – build manifest via streaming (no full download); data stays in HF cache.

Output:

- `datasets/LIBERO-Cosmos-Policy/manifest.json` – train/val episode indices (start, end, task_description, success, suite).
- Optionally `datasets/LIBERO-Cosmos-Policy/data/` – local copy if not using streaming.

The [dataset](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy) is auto-converted to Parquet (~643k rows, train split). Rows are timesteps; columns include `actions`, `proprio`, `primary_images_jpeg`, `wrist_images_jpeg`, and (when present) `task_description`, `success`. The script infers episode boundaries from `episode_index` if present, else from **task_description** change between rows. Suites are inferred from task text (spatial/object/goal/libero_10).

**Verify download (recommended):** run once with `--verify` to confirm columns and episode count without writing files: `uv run python scripts/download_libero_cosmos.py --output-dir datasets --verify`. You should see many episodes and ~643k total rows; then run again without `--verify` to write the manifest and save data.

**If you see only 1 trajectory (or “Total steps: 1”)** when training, the manifest was likely built from too little data (e.g. `--streaming` stopped early, or a tiny cache). Re-run the download **without** `--streaming` so the full dataset is loaded, then check `manifest.json` for a non-trivial `train_episodes` list.

### Recommended: HDF5 layout (Cosmos Policy)

The [dataset README](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy) uses **HDF5**: `all_episodes/` has one `.hdf5` file per episode with `proprio`, `actions`, and attributes `task_description`, `success`. The Hub’s Parquet view drops that metadata, so for correct episodes use the HDF5 layout as in [Cosmos Policy LIBERO.md](https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md):

```bash
hf download nvidia/LIBERO-Cosmos-Policy --repo-type dataset --local-dir LIBERO-Cosmos-Policy
mv LIBERO-Cosmos-Policy datasets/
uv run python scripts/download_libero_cosmos.py --output-dir datasets
```

The script sees `all_episodes/*.hdf5`, builds a manifest with one episode per file, and training loads each trajectory from HDF5 (with task/success from file). Requires `h5py` (e.g. `uv sync --extra icrt`). If you only run the script without `hf download`, you get Parquet and may see one giant episode; use the HDF5 flow above for proper episode boundaries.

## 3. Train

```bash
uv run python -m src.train \
  --override data=[base,libero_cosmos] \
  experiment.eval_every_steps=5000 \
  experiment.max_steps=100000 \
  --run-name libero_cosmos_run
```

- `data=[base,libero_cosmos]` merges `configs/data/base.yaml` with `configs/data/libero_cosmos.yaml` (state_dim=9, act_dim=7, language, etc.).
- Training loads trajectories from the manifest + HuggingFace (or local) data and writes checkpoints under `outputs/icl_adaptation/<date>/<run_name>__seed_<X>__<hash>/`.
- At startup, **dataset stats** print observation keys: **Image keys (config)** = `primary_images_jpeg`, `wrist_images_jpeg` (primary + wrist camera in Hub/Parquet) and **Proprio keys (config)** = `proprio`. The current HDF5 loader provides proprio only; use Parquet or an image-enabled loader for vision.

**Eval rollouts:** `env_name` is the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark suite: **libero_10**, **libero_spatial**, **libero_object**, **libero_goal**, or **libero_90**. Install the LIBERO sim with: `uv sync --extra libero` (or `pip install -e ".[libero]"`). Then eval rollouts run in that suite (task 0). Without the extra, rollouts are skipped and a warning is logged; you still get **offline** evaluation via `scripts/run_libero_eval.py` (see §5). See also [Cosmos Policy LIBERO.md](https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md). To skip rollout attempts entirely, set `experiment.eval_every_steps=0`.

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
  model.vision_encoder_type=patch experiment.eval_every_steps=0 --run-name libero_patch_run
```

**DINOv2:**
```bash
uv run python -m src.train \
  --override data=[base,libero_cosmos] model=vla_dt data.use_vision=true \
  model.vision_encoder_type=dinov2 experiment.eval_every_steps=0 --run-name libero_dinov2_run
```

**CrossMAE with attention pooling:**
```bash
uv run python -m src.train \
  --override data=[base,libero_cosmos] model=vla_dt data.use_vision=true \
  model.vision_encoder_type=crossmae model.vision_encoder_attention_pool=true \
  experiment.eval_every_steps=0 --run-name libero_crossmae_run
```

**PaliGemma-style SigLIP:**
```bash
uv run python -m src.train \
  --override data=[base,libero_cosmos] model=vla_dt data.use_vision=true \
  model.vision_encoder_type=paligemma experiment.eval_every_steps=0 --run-name libero_paligemma_run
```

Note: LIBERO-Cosmos as shipped does not load images; the dataset uses proprio only. The above examples assume a vision-enabled data path (e.g. after adding `image_keys` and vision fields to the LIBERO config, or when using the same overrides with another vision dataset).

## 5. Run in-distribution evaluation (held-out)

After training, run **offline** eval on the held-out episodes from the manifest (no simulator):

```bash
uv run python scripts/run_libero_eval.py \
  --ckpt outputs/icl_adaptation/<date>/libero_cosmos_run__seed_412__<hash>/ckpts/best/checkpoint.pt \
  --manifest datasets/LIBERO-Cosmos-Policy/manifest.json \
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
| Download   | `uv run python scripts/download_libero_cosmos.py --output-dir datasets` |
| Train      | `uv run python -m src.train --override data=[base,libero_cosmos] --run-name libero_cosmos_run` |
| Train (VLA + vision backbone) | `uv run python -m src.train --override data=[base,libero_cosmos] model=vla_dt data.use_vision=true model.vision_encoder_type=dinov2 --run-name libero_dinov2_run` (see §4 for patch/crossmae/paligemma) |
| Eval (ID)  | `uv run python scripts/run_libero_eval.py --ckpt <path-to-ckpt> --manifest datasets/LIBERO-Cosmos-Policy/manifest.json --data-dir datasets` |

Everything is **modular**: download script, `src/data/libero_dataset.py`, config `configs/data/libero_cosmos.yaml`, and `scripts/run_libero_eval.py` can be run and read independently.
