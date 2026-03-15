# Setup

## Environment

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

## Optional: Minari (HalfCheetah / offline RL)

We use **Minari** for HalfCheetah mixed-expertise data. Minari uses Gymnasium and MuJoCo 2 and does **not** use the old `mujoco_py` (no Cython compilation).

**uv:**
```bash
uv sync --extra d4rl
```
This installs `minari[all]` (Gymnasium, MuJoCo 2, dataset download).

**pip:**
```bash
pip install "minari[all]"
```

---

## Quick start: HalfCheetah via Minari (download → train)

**1. Install Minari**  

With **uv**:
```bash
uv sync --extra d4rl
```

With pip:
```bash
pip install "minari[all]"
```

**2. Download HalfCheetah datasets (mixed expertise)**  
Saves to `datasets/HalfCheetah-v2/<quality>/trajectories.pkl`. Use `medium_expert` for a mix of medium and expert returns.

With **uv**:
```bash
uv run python scripts/download_d4rl_halfcheetah.py --output-dir datasets --qualities medium expert medium_expert
```

With pip:
```bash
python scripts/download_d4rl_halfcheetah.py --output-dir datasets --qualities medium expert medium_expert
```

**3. Train Meta-DT with W&B and Loguru**  

With **uv**:
```bash
uv run python -m src.train --wandb --run-name halfcheetah-run1 --override data=halfcheetah
```

With pip:
```bash
python -m src.train --wandb --run-name halfcheetah-run1 --override data=halfcheetah
```

**4. All-in-one script (download + train)**  
Uses `uv run python` if `uv` is on your PATH, otherwise `python`.

```bash
chmod +x scripts/run_halfcheetah.sh
./scripts/run_halfcheetah.sh
```

**5. Optional overrides**  
```bash
uv run python -m src.train --wandb --override data=halfcheetah experiment.max_steps=100000 experiment.eval_every_steps=2000
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

## Troubleshooting

- **`mujoco_py` / Cython errors**  
  This project uses **Minari** for HalfCheetah, not the legacy D4RL package that can pull in `mujoco_py`. Install `minari[all]` (see “Optional: Minari” above). If you see Cython errors from another package, uninstall `mujoco_py` and use Minari only.

- **`transformer_engine` or `accelerate` import errors**  
  Use a clean environment with only the dependencies in `pyproject.toml`. Avoid pulling in extras that depend on `accelerate`/`transformer_engine` if you don’t need them.

- **Minari not found**  
  Run `uv sync --extra d4rl` or `pip install "minari[all]"` so the download script and env can use Minari datasets.
