# LIBERO-Cosmos-Policy: training and in-distribution evaluation

This doc covers **downloading**, **training**, and **evaluation** for [nvidia/LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy) (HuggingFace). Training is ICRT-style (in-context trajectories + language); evaluation is **offline** on a held-out set, with metrics **per suite** (in-distribution: libero_spatial, libero_object, libero_goal, libero_10).

## 1. Prerequisites

- **uv** (or pip)
- **datasets** (HuggingFace): `pip install datasets` or `uv sync --extra icrt`

## 2. Download and prepare data

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

The script infers episode boundaries from the dataset (e.g. `episode_index` if present, else task_description change). Suites are inferred from task text (spatial/object/goal/libero_10).

## 3. Train

```bash
uv run python -m src.train \
  --override data=libero_cosmos \
  experiment.eval_every_steps=5000 \
  experiment.max_steps=100000 \
  --run-name libero_cosmos_run
```

- `data=libero_cosmos` uses `configs/data/libero_cosmos.yaml` (state_dim=9, act_dim=7, language, etc.).
- Training loads trajectories from the manifest + HuggingFace (or local) data and writes checkpoints under `outputs/icl_adaptation/<date>/<run_name>__seed_<X>__<hash>/`.

To **disable** env-based evaluation (no rollouts, same as ICRT-MT):

```bash
uv run python -m src.train \
  --override data=libero_cosmos experiment.eval_every_steps=0 \
  --run-name libero_cosmos_run
```

## 4. Run in-distribution evaluation (held-out)

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

## 5. Summary

| Step        | Command |
|------------|--------|
| Install    | `uv sync --extra icrt` (or `pip install datasets`) |
| Download   | `uv run python scripts/download_libero_cosmos.py --output-dir datasets` |
| Train      | `uv run python -m src.train --override data=libero_cosmos --run-name libero_cosmos_run` |
| Eval (ID)  | `uv run python scripts/run_libero_eval.py --ckpt <path-to-ckpt> --manifest datasets/LIBERO-Cosmos-Policy/manifest.json --data-dir datasets` |

Everything is **modular**: download script, `src/data/libero_dataset.py`, config `configs/data/libero_cosmos.yaml`, and `scripts/run_libero_eval.py` can be run and read independently.
