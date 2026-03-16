# ICRT-MT training (real-robot, no eval)

This doc is **only** for training on the **ICRT-MT** real-robot dataset. There is **no environment-based evaluation** (no rollouts in sim or real); we only check that training runs and loss goes down.

## How full-trajectory context works

- **Per task:** The loader groups trajectories by task (e.g. `close-drawer`, `open-drawer`) and keeps up to **5 demo trajectories per task** as the context pool (`prompt_trajectories_per_task`).
- **Per sample:** For each training sample (one trajectory at one start index), the dataset:
  1. Finds the **task** of that trajectory (`task_id = traj_idx // total_epi_per_task`).
  2. Takes the **context pool** for that task (the up-to-5 demos).
  3. Calls **`sample_context_trajectories(pool, n=num_context_trajectories, ascending=context_sort_ascending, sampling=context_sampling)`** to pick **3** trajectories from the pool (with your config: `num_context_trajectories: 3`, `context_sort_ascending: true`, `context_sampling: random` from base). So it randomly samples 3 from the pool, then **sorts them by return ascending** (worst → best).
  4. **Full-trajectory mode:** Each of the 3 chosen trajectories is used in **full**, but each is **capped to `max_prompt_trajectory_length`** steps (e.g. 85): the **last** N steps of each demo are kept so you get the end of each trajectory. Those (up to 3 × 85) steps are **concatenated**, then the combined sequence is **trimmed to the last 256 steps** (`max_total_prompt_length: 256`). So each context trajectory is full up to a per-demo cap, then the total prompt is capped.
- **Summary:** Context is **same-task**, **3 demos** chosen at random from up to 5 per task, ordered by return ascending; each demo is capped to **max_prompt_trajectory_length** steps (last N), concatenated, then total capped to **max_total_prompt_length**.

## 1. Prerequisites

- **uv** (or conda + pip)
- **ICRT extra**: `huggingface_hub`, `h5py`, `matplotlib`

```bash
uv sync --extra icrt
# or: pip install huggingface_hub h5py matplotlib
```

- **Git LFS** (for large HDF5 files):

```bash
sudo apt install git-lfs && git lfs install
```

## 2. Download ICRT-MT

```bash
uv run python scripts/download_icrt_dataset.py --output-dir datasets
```

Data lands under `datasets/ICRT-MT/` (or your `paths.data_root`) with `dataset_config.json` and HDF5 files. Paths are resolved from the **repo root** so the dataset is found at `<repo>/datasets/ICRT-MT/dataset_config.json`. Override from CLI: `paths.data_root=/abs/path/to/datasets`.

## 3. (Optional) Inspect data

```bash
uv run python scripts/visualize_icrt_data.py --out-dir outputs/icrt_viz --max-episodes 2
```

## 4. Train (no eval)

Use the ICRT-MT data config and turn off eval so the training loop only does forward/backward and checkpointing:

```bash
uv run python -m src.train \
  --override data=[base,icrt_mt] model=vla_dt \
  experiment.eval_every_steps=0 \
  experiment.max_steps=50000 \
  --run-name icrt_mt_run
```

- `data=[base,icrt_mt]` merges `configs/data/base.yaml` with `configs/data/icrt_mt.yaml` (points to `datasets/ICRT-MT/dataset_config.json`).
- `model=vla_dt` selects the Vision-Language-Action DT (`configs/model/vla_dt.yaml`) with vision and language conditioning.
- `experiment.eval_every_steps=0` disables the eval hook (no rollout viz, no eval metrics).
- Training will write to the usual run directory under `outputs/icl_adaptation/<date>/icrt_mt_run__seed_<X>__<hash>/`.

## 5. Check that it trains

- Watch **train loss** in the progress bar and in logs.
- Checkpoints are saved under the run dir: `ckpts/last/`, `ckpts/best/` (when eval is disabled, “best” is still updated from the placeholder metric if you leave eval on; with `eval_every_steps=0` there is no eval, so only `ckpts/last/` and any periodic `ckpts/step_*/` are used).
- Logs: `run_dir/logs/train.log` and TensorBoard/W&B if enabled.

## 6. Full-trajectory context

To use **full in-context trajectories** (whole demos, capped by `max_total_prompt_length` and `max_prompt_trajectory_length`), override `context_style`. `icrt_mt.yaml` already sets `max_total_prompt_length: 256` and `max_prompt_trajectory_length: 85`:

```bash
uv run python -m src.train \
  --override data=[base,icrt_mt] model=vla_dt data.context_style=full_trajectory \
  experiment.eval_every_steps=0 experiment.max_steps=50000 \
  --run-name icrt_mt_full_run
```

## 7. Query history length (K)

The **query** (current trajectory) is the **last K steps** ending at the prediction timestep. K is set by `data.query_history_length`; if unset (null), K defaults to `horizon`.

- **K=1 (OpenVLA-style):** Only the current observation is fed; the model predicts the next action from prompt + current state. Use `data.query_history_length=1`.
- **K>1 (PromptDT-style):** The last K steps (state, action, reward, RTG, timestep) are fed; the model still predicts actions autoregressively over those K steps. Use e.g. `data.query_history_length=8` or leave default (same as horizon).

Example (OpenVLA-style, single-step query):

```bash
uv run python -m src.train \
  --override data=[base,icrt_mt] model=vla_dt data.query_history_length=1 \
  experiment.eval_every_steps=0 experiment.max_steps=50000 \
  --run-name icrt_mt_k1_run
```

## 8. Resume

```bash
uv run python -m src.train \
  --resume outputs/icl_adaptation/<date>/icrt_mt_run__seed_412__<hash>/ckpts/last/checkpoint.pt \
  --override data=[base,icrt_mt] model=vla_dt experiment.eval_every_steps=0
```

(Use the same `data.query_history_length` as the run you are resuming.)

## Summary

| Step        | Command |
|------------|--------|
| Install deps | `uv sync --extra icrt` |
| Download     | `uv run python scripts/download_icrt_dataset.py --output-dir datasets` |
| Train (no eval) | `uv run python -m src.train --override data=[base,icrt_mt] model=vla_dt experiment.eval_every_steps=0 --run-name icrt_mt_run` |
| Train (full traj) | `uv run python -m src.train --override data=[base,icrt_mt] model=vla_dt data.context_style=full_trajectory experiment.eval_every_steps=0 --run-name icrt_mt_full_run` |
| Train (K=1 / OpenVLA-style) | `uv run python -m src.train --override data=[base,icrt_mt] model=vla_dt data.query_history_length=1 experiment.eval_every_steps=0 --run-name icrt_mt_k1_run` |
| Resume       | `uv run python -m src.train --resume <path-to-ckpt> --override data=[base,icrt_mt] model=vla_dt experiment.eval_every_steps=0` |

No sim or real env is used; this flow is only to verify that ICRT-MT training runs end-to-end.
