# Setup

This guide gets you from a fresh clone to running training (HalfCheetah or ICRT-style robot data).

---

## Prerequisites

- **Python**: 3.9 or 3.10 (3.10 recommended).
- **GPU**: Optional but recommended for training; CPU works for small runs.
- **Disk**: ~500 MB for HalfCheetah data; more for ICRT-MT (HDF5 + images).
- **Git LFS**: Required only for downloading ICRT-MT (large files). Install with `sudo apt install git-lfs && git lfs install` (Linux) or see [git-lfs.github.com](https://git-lfs.github.com).

---

## Step 1: Clone and enter the project

```bash
cd /path/to/icl_adaptation   # or wherever you cloned the repo
```

---

## Step 2: Create environment and install dependencies

Pick **one** of the following.

### Option A: With [uv](https://github.com/astral-sh/uv)

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

This creates a virtualenv in `.venv` and installs the base dependencies from `pyproject.toml`.

### Option B: With conda + pip

```bash
conda create -n icl_adaptation python=3.10 -y
conda activate icl_adaptation
pip install -e .
```

**Verify:** From the project root, run `python -c "import src.train; print('OK')"`. If that succeeds, the base setup is correct.

---

## Step 3: Install optional dependencies (depending on what you want to run)

| Goal | What to install | Command |
|------|-----------------|--------|
| **HalfCheetah (Minari)** | Minarie2e + Gymnasium + MuJoCo 2 | `uv sync --extra d4rl` |
| **ICRT-MT dataset + viz** | HuggingFace Hub + HDF5 + matplotlib | `uv sync --extra icrt` |
| **Reward relabeling** (Robometer-4B / RoboDopamine-8B) | robometer from GitHub, without its full dependency tree | `uv pip install "git+https://github.com/robometer/robometer.git" --no-deps` then `uv sync --extra reward-relabel` |

You can install both if you plan to use HalfCheetah and ICRT-style data. **Robometer is not on PyPI**; install it from [GitHub](https://github.com/robometer/robometer). Using `--no-deps` avoids pulling in all of its requirements (e.g. vLLM); the project’s existing torch/transformers deps are enough for the reward model code paths.

---

## Step 4: Download data (choose one path)

### Path A: HalfCheetah (Minari)

1. Install Minari (see Step 3).
2. Download datasets into `datasets/`:

   ```bash
   uv run python scripts/download_d4rl_halfcheetah.py --output-dir datasets --qualities medium expert medium_expert
   ```

3. Check that files exist:  
   `datasets/HalfCheetah-v2/medium_expert/trajectories.pkl` (and similarly for `medium`, `expert`).

### Path B: ICRT-MT (language + multi-view images)

1. Run `uv sync --extra icrt` (see Step 3).
2. (Recommended) Install Git LFS for large files:  
   `sudo apt install git-lfs && git lfs install`
3. Download the dataset:

   ```bash
   uv run python scripts/download_icrt_dataset.py --output-dir datasets
   ```

4. Check that the folder exists:  
   `datasets/ICRT-MT/` and `datasets/ICRT-MT/dataset_config.json`.

---

## Step 5: Run training

- **HalfCheetah (with W&B):**
  ```bash
  uv run python -m src.train --wandb --run-name halfcheetah-run1 --override data=[base,halfcheetah]
  ```

- **Resume from checkpoint:**
  ```bash
  uv run python -m src.train --resume outputs/checkpoints/checkpoint_latest.pt --override data=[base,halfcheetah]
  ```

- **Override config (e.g. steps, batch size):**
  ```bash
  uv run python -m src.train --override data=[base,halfcheetah] experiment.max_steps=5000 data.batch_size=64
  ```

---

## Environment (reference)

If you only need the exact commands for creating the env (without the full walkthrough above):

### With [uv](https://github.com/astral-sh/uv)

```bash
cd /path/to/icl_adaptation
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
```

### With conda + pip

```bash
conda create -n icl_adaptation python=3.10 -y
conda activate icl_adaptation
pip install -e .
```

---

## Optional: Minari (HalfCheetah) in one go

Minari uses Gymnasium and MuJoCo 2 (no `mujoco_py`). Run `uv sync --extra d4rl`, then follow **Step 4 (Path A)** and **Step 5** above. Alternatively, use the all-in-one script:

```bash
chmod +x scripts/run_halfcheetah.sh
./scripts/run_halfcheetah.sh
```

---

## Training entrypoint (uv)

| Action | Command |
|--------|--------|
| Start fresh | `uv run python -m src.train` |
| Resume | `uv run python -m src.train --resume outputs/checkpoints/checkpoint_latest.pt` |
| Eval only | `uv run python -m src.train --eval-only --resume outputs/checkpoints/checkpoint_best.pt` |
| Export artifact | `uv run python -m src.train --export-only outputs/checkpoints/checkpoint_best.pt` |
| Overrides | `uv run python -m src.train --override experiment.max_steps=1000 data.batch_size=64` |

---

## Context trajectories and robotics

- **N in-context trajectories**: In config (e.g. `configs/data/halfcheetah.yaml`) set `num_context_trajectories: 3`, `context_sort_ascending: true`, and optionally `context_sampling: stratified`. Training will sample N trajectories per segment (same task), sort by increasing return, and feed them as context. See README “Data and context”.
- **Robotics dense rewards**: For robot rollouts without dense rewards, run the offline Robometer step once before training:  
  `uv run python scripts/score_trajectories_robometer.py --input <pkl> --output <pkl> --task "task"`.  
  See [Robometer-4B](https://huggingface.co/robometer/Robometer-4B) and the script docstring.

---

## Optional: ICRT-style (language + multi-view images)

For robot manipulation with **language instructions** and **multi-view camera images** (exterior + wrist), similar to [ICRT](https://github.com/Max-Fu/icrt) (ICRA 2025):

1. **Install deps:** `uv sync --extra icrt` (includes huggingface_hub, h5py, matplotlib for download and visualization).
2. **Install Git LFS** (for large HDF5 files): `sudo apt install git-lfs && git lfs install`.
3. **Download data:** See **Step 4 (Path B)** above. This fetches [Ravenh97/ICRT-MT](https://huggingface.co/datasets/Ravenh97/ICRT-MT) to `datasets/ICRT-MT/` and writes `dataset_config.json`.
4. **Visualize dataset** (task distribution, episode lengths, sample frames):
   ```bash
   uv run python scripts/visualize_icrt_data.py
   uv run python scripts/visualize_icrt_data.py --out-dir outputs/icrt_viz --max-episodes 2 --sample-frames 5
   ```
   Outputs PNGs in `outputs/icrt_viz/` (or `--out-dir`).
5. **Config and model:**
   - Data: `configs/data/icrt_mt.yaml` (sets `dataset_config_json`, `image_keys`, `use_vision`, `use_language`).
   - Model: `VLADecisionTransformer` in `src/models/vla_dt.py` (Meta-DT + optional vision encoder + language embedding). Use `--override data=[base,icrt_mt] model=vla_dt` when training on ICRT-MT.

---

## Quick reference: common commands

| What | Command |
|------|--------|
| Create env (uv) | `uv venv && source .venv/bin/activate && uv sync` |
| Create env (conda) | `conda create -n icl_adaptation python=3.10 -y && conda activate icl_adaptation && pip install -e .` |
| Install HalfCheetah deps | `uv sync --extra d4rl` |
| Download HalfCheetah | `uv run python scripts/download_d4rl_halfcheetah.py --output-dir datasets --qualities medium expert medium_expert` |
| Install ICRT deps | `uv sync --extra icrt` |
| Download ICRT-MT | `uv run python scripts/download_icrt_dataset.py --output-dir datasets` |
| Visualize ICRT-MT | `uv run python scripts/visualize_icrt_data.py` (optional: `--out-dir outputs/icrt_viz --max-episodes 2`) |
| Train (HalfCheetah) | `uv run python -m src.train --wandb --override data=[base,halfcheetah]` |
| Resume training | `uv run python -m src.train --resume outputs/checkpoints/checkpoint_latest.pt --override data=[base,halfcheetah]` |
| Override config | `uv run python -m src.train --override data=[base,halfcheetah] experiment.max_steps=5000 data.batch_size=64` |

---

## Troubleshooting

- **`mujoco_py` / Cython errors**  
  This project uses **Minari** for HalfCheetah, not the legacy D4RL package that can pull in `mujoco_py`. Install `minari[all]` (see Step 3 / Optional: Minari). If you see Cython errors from another package, uninstall `mujoco_py` and use Minari only.

- **`transformer_engine` or `accelerate` import errors**  
  Use a clean environment with only the dependencies in `pyproject.toml`. Avoid pulling in extras that depend on `accelerate`/`transformer_engine` if you don’t need them.

- **Minari not found**  
  Run `uv sync --extra d4rl` or `pip install "minari[all]"` so the download script and env can use Minari datasets.

- **ICRT-MT download fails or files missing**  
  Ensure Git LFS is installed (`git lfs install`) and that you have enough disk space. You can also download the dataset manually from [HuggingFace ICRT-MT](https://huggingface.co/datasets/Ravenh97/ICRT-MT) and place it under `datasets/ICRT-MT/`, then create or adjust `dataset_config.json` to match the file paths.
