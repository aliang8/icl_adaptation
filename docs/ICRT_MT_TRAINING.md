# ICRT-MT training (real-robot, no eval)

This doc is **only** for training on the **ICRT-MT** real-robot dataset. There is **no environment-based evaluation** (no rollouts in sim or real); we only check that training runs and loss goes down.

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
  --override data=icrt_mt \
  experiment.eval_every_steps=0 \
  experiment.max_steps=50000 \
  --run-name icrt_mt_run
```

- `data=icrt_mt` uses `configs/data/icrt_mt.yaml` (points to `datasets/ICRT-MT/dataset_config.json`).
- `experiment.eval_every_steps=0` disables the eval hook (no rollout viz, no eval metrics).
- Training will write to the usual run directory under `outputs/icl_adaptation/<date>/icrt_mt_run__seed_<X>__<hash>/`.

## 5. Check that it trains

- Watch **train loss** in the progress bar and in logs.
- Checkpoints are saved under the run dir: `ckpts/last/`, `ckpts/best/` (when eval is disabled, “best” is still updated from the placeholder metric if you leave eval on; with `eval_every_steps=0` there is no eval, so only `ckpts/last/` and any periodic `ckpts/step_*/` are used).
- Logs: `run_dir/logs/train.log` and TensorBoard/W&B if enabled.

## 6. Resume

```bash
uv run python -m src.train \
  --resume outputs/icl_adaptation/<date>/icrt_mt_run__seed_412__<hash>/ckpts/last/checkpoint.pt \
  --override data=icrt_mt experiment.eval_every_steps=0
```

## Summary

| Step        | Command |
|------------|--------|
| Install deps | `uv sync --extra icrt` |
| Download     | `uv run python scripts/download_icrt_dataset.py --output-dir datasets` |
| Train (no eval) | `uv run python -m src.train --override data=icrt_mt experiment.eval_every_steps=0 --run-name icrt_mt_run` |
| Resume       | `uv run python -m src.train --resume <path-to-ckpt> --override data=icrt_mt experiment.eval_every_steps=0` |

No sim or real env is used; this flow is only to verify that ICRT-MT training runs end-to-end.
