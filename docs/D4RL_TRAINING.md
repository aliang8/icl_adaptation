# D4RL-style offline RL (HalfCheetah via Minari)

This doc is the **end-to-end command list** for MuJoCo **HalfCheetah** data that matches the classic **D4RL** quality splits (medium, expert, medium-replay, etc.). We load it through **Minari** (Gymnasium + MuJoCo 2), **not** the legacy `d4rl` / `mujoco_py` stack.

## Rewards: no relabeling step

Trajectories already include **environment rewards** per step (`rewards` in each episode). You **do not** run:

- `scripts/compute_dense_rewards.py` (robot MP4 / manifest layout)
- `scripts/score_trajectories_robometer.py` (video / frame scoring)

Those are for robotics datasets without dense rewards. For HalfCheetah, **download → train** is enough.

**Model:** use default **`model=transformer`** with `data=[base,halfcheetah]`. To train on **prompt + query** (not query-only), override e.g. `model.query_loss_only=false`. RTG conditioning is already on in `configs/model/transformer.yaml` (`condition_rtg: true`).

---

## 1. Prerequisites

- Python 3.9+ (3.10 recommended)
- Optional: GPU for training
- **[uv](https://github.com/astral-sh/uv)** recommended (or use `python` / `pip` equivalents)

From the repo root:

```bash
cd /path/to/icl_adaptation
```

---

## 2. Environment and dependencies

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync
uv sync --extra d4rl        # Minari + native mujoco (Gymnasium HalfCheetah-v5 eval); no mujoco_py
```

**Gymnasium vs ManiSkill:** do not install **`mani-skill`** into this D4RL training environment. It pins **`gymnasium==0.29.1`**, which lacks MuJoCo **v5** env modules and breaks **`HalfCheetah-v5`** / Minari `recover_environment()` (`ModuleNotFoundError: ... half_cheetah_v5`). Use a **separate venv** and `scripts/maniskill/requirements.txt` for ManiSkill PPO only (`docs/MANISKILL.md`).

`d4rl` here is the **project extra** in `pyproject.toml` (Minari-based), not necessarily the PyPI `d4rl` package.

---

## 3. Download datasets

Writes `trajectories.pkl` under `datasets/HalfCheetah-v2/<quality>/`.

**Recommended (medium + expert + combined mix):**

```bash
uv run python scripts/download_d4rl_halfcheetah.py \
  --output-dir datasets \
  --qualities medium expert medium_expert
```

**Other qualities** (if supported by Minari for your install):

```bash
uv run python scripts/download_d4rl_halfcheetah.py \
  --output-dir datasets \
  --qualities medium expert medium_replay medium_expert
```

**Skip re-download** (only pack existing Minari caches):

```bash
uv run python scripts/download_d4rl_halfcheetah.py \
  --output-dir datasets \
  --qualities medium expert medium_expert \
  --no-download
```

**Verify:**

```bash
ls datasets/HalfCheetah-v2/medium_expert/trajectories.pkl
ls datasets/HalfCheetah-v2/medium/trajectories.pkl
ls datasets/HalfCheetah-v2/expert/trajectories.pkl
```

---

## 3b. RTG scaling (`data.rtg_scale`)

Step rewards in `trajectories.pkl` stay **as stored** (raw env rewards). Only the **return-to-go targets** divide cumulative reward by **`data.rtg_scale`** so DT RTG tokens are ~O(1). Default in `configs/data/base.yaml` is **`1.0`**; HalfCheetah uses **`5000`** in `configs/data/halfcheetah.yaml`.

---

## 4. Train

**Minimal** (default `model=transformer`):

```bash
uv run python -m src.train \
  --wandb \
  --run-name halfcheetah-run1 \
  --override data=[base,halfcheetah]
```

**Resume:**

```bash
uv run python -m src.train \
  --resume outputs/<project>/<date>/<run_dir>/ckpts/last/checkpoint.pt \
  --override data=[base,halfcheetah]
```

**Override training length / batch size:**

```bash
uv run python -m src.train \
  --override data=[base,halfcheetah] \
  experiment.max_steps=5000 \
  data.batch_size=4 \
  paths.data_root=/project2/biyik_1165/aliang80/datasets \
  experiment.eval_every_steps=100
```

**Prompt + query action loss** (optional):

```bash
uv run python -m src.train \
  --override data=[base,halfcheetah] \
  model.query_loss_only=false
```

**Use a single quality** (e.g. expert only), align with what you downloaded:

```bash
uv run python -m src.train \
  --override data=[base,halfcheetah] \
  data.data_quality=expert
```

Config pieces live in:

- `configs/data/base.yaml` + `configs/data/halfcheetah.yaml` (merged via `data=[base,halfcheetah]`)
- `configs/model/transformer.yaml` (default Meta-DT)
- Loader: `src/data/d4rl_loader.py` → `datasets/HalfCheetah-v2/<data_quality>/trajectories.pkl`

---

## 4b. Policy checkpoints and offline eval

**Checkpointing (already in training)** — While training, `src.engine.trainer.Trainer` saves resumable checkpoints under your run directory:

| Location | When |
|----------|------|
| `<run_dir>/ckpts/last/checkpoint.pt` | Every `experiment.save_latest_every_steps` (if set) |
| `<run_dir>/ckpts/best/checkpoint.pt` | When eval improves `experiment.best_metric_name` (if `experiment.save_best`) |
| `<run_dir>/ckpts/step_XXXXXX/checkpoint.pt` | Every `experiment.save_periodic_every_steps` (if set) |

Each file contains `model`, optimizer/scheduler state, `global_step`, `config` (full Hydra snapshot), RNG, etc. Optional end-of-run export: **`model_export.pt`** under `<run_dir>/artifacts/inference/` (weights + config + `state_mean` / `state_std`) when `experiment.export_final` is true — see `src/engine/checkpointing.py`.

**Note:** `src.train --eval-only` currently logs a placeholder only; it does **not** run Gymnasium rollouts.

**Standalone eval (same rollouts as training eval)** — Load a checkpoint and call `run_rollouts_and_save_viz` (prompt or zero-shot mode, RTG handling, plots under `viz/samples/...`). The script merges **current Hydra defaults** with the **config stored in the checkpoint**, then applies your CLI overrides (so you can fix `paths.data_root` if data moved).

```bash
uv run python scripts/run_d4rl_policy_eval.py \
  --checkpoint outputs/<project>/<date>/<run>/ckpts/best/checkpoint.pt \
  --override paths.data_root=/path/to/your/datasets \
  --override data=[base,halfcheetah]
```

Useful flags:

- `--step 20000` — label for output folder `viz/samples/step_020000/`
- `--output-dir /tmp/my_eval_run` — where to write viz (default: inferred parent of `ckpts/` from the checkpoint path)
- `--strict` — `load_state_dict(strict=True)` if the architecture must match exactly
- `--weights-only` — use `torch.load(..., weights_only=True)` only for tensor-only artifacts; full training checkpoints load with the default (`weights_only=False`) so NumPy stats and config unpickle correctly on PyTorch 2.6+.

Align **`data.data_quality`**, **`data.rtg_scale`**, **`experiment.eval_*`**, and **`model.*`** with the training run (they are restored from the checkpoint config; overrides can change them intentionally).

**Export a light inference artifact from a full checkpoint:**

```bash
uv run python -m src.train \
  --export-only outputs/.../ckpts/best/checkpoint.pt \
  --override data=[base,halfcheetah]
```

Writes `artifacts/inference/model_export.pt` next to that run (includes normalization stats when the trainer passed them at export time).

---

## 5. One-shot script (download + train)

From repo root:

```bash
chmod +x scripts/run_halfcheetah.sh
./scripts/run_halfcheetah.sh
```

This runs the download command in §3, then training with W&B and `data=[base,halfcheetah]`. Extra arguments are forwarded to `src.train`, e.g. `./scripts/run_halfcheetah.sh experiment.max_steps=5000`.

---

## 6. Quick reference

| Step | Command |
|------|--------|
| Install Minari stack | `uv sync --extra d4rl` |
| Download (default qualities) | `uv run python scripts/download_d4rl_halfcheetah.py --output-dir datasets --qualities medium expert medium_expert` |
| Train | `uv run python -m src.train --wandb --override data=[base,halfcheetah]` |
| Resume | `uv run python -m src.train --resume <path/to/checkpoint.pt> --override data=[base,halfcheetah]` |
| Eval saved policy (rollouts + viz) | `uv run python scripts/run_d4rl_policy_eval.py --checkpoint <run>/ckpts/best/checkpoint.pt --override paths.data_root=... data=[base,halfcheetah]` |
| Export inference-only weights | `uv run python -m src.train --export-only <run>/ckpts/best/checkpoint.pt --override data=[base,halfcheetah]` |
| RTG divisor override | e.g. `data.rtg_scale=5000` |

---

## 7. Troubleshooting

- **Long prompts / CUDA index errors in GPT-2** — **`model.n_positions`** must be ≥ transformer **token** length (**`≈ 3 × (prompt_steps + query_steps)`** with `condition_rtg`). **`model.max_ep_len`** must exceed max **timestep index** in data. Defaults are raised in `configs/model/transformer.yaml`; override with e.g. `model.n_positions=32768` if needed.
- **`max_total_prompt_length` vs real context length** — With `context_style: full_trajectory`, each sample’s prompt is **padded or trimmed** to exactly `max_total_prompt_length` (`_pad_or_trim_prompt` in `dataset.py`). If you use 3×1000-step demos but set `max_total_prompt_length: 5000`, you get **2000 padded steps** (masked), not “5000 steps of data.” Set it to **`num_context_trajectories * max_episode_steps`** (e.g. 3000) when episodes are full length. The model then sees **`max_total_prompt_length`** prompt timesteps (plus the query segment).
- **Eval: `HalfCheetah-v2` / `gymnasium-robotics` / `mujoco_py`** — Config and dataset paths stay `HalfCheetah-v2` (D4RL / Minari naming). **Eval rollouts** use **`HalfCheetah-v5`** automatically (same 17-dim obs, 6-dim action; modern MuJoCo in Gymnasium). No `mujoco_py` required. **Eval videos** use Gymnasium’s `render_mode="rgb_array"` at `gym.make` time (no `render(mode=...)` on wrappers).
- **Prompt / batch logging** — Training always logs **`[train_batch]`** once (first step): tensor shapes, first two rows’ **masked `prompt_r` sums**, and DTBatch layout. The dataset logs **`[prompt_context_returns]`** for the first **4** samples per process: **full-episode env return** per context trajectory in **prompt order**. With `num_workers>0`, each worker may emit up to 4 `[prompt_context_returns]` lines.
- **Eval CUDA OOM** — (1) **Sequence length:** each `get_action` runs the transformer on **full prompt + query** padded to `model.max_length`; with `condition_rtg`, length ≈ **`3 × (prompt_timesteps + max_length)`** (~12k tokens for a 4000-step prompt). (2) **Autograd:** eval must run under **`torch.inference_mode()`**; otherwise each step’s `action` was concatenated into `actions_t` and **chained ~1000 forwards into one graph**, blowing VRAM. Training wraps eval in **`model.eval()` + `inference_mode()`**; rollouts also **detach** actions before re-feeding. Further mitigations: lower **`data.max_total_prompt_length`**, shorter **`max_episode_steps`**, disable eval video, or **`torch.cuda.empty_cache()`** for fragmentation. Logs: **`Eval transformer seq`**, **`[eval rollout] env_timestep=...`**.
- **Minari not found** → `uv sync --extra d4rl`
- **`ModuleNotFoundError: No module named 'mujoco'` / `DependencyNotInstalled: MuJoCo is not installed`** — Run **`uv sync --extra d4rl`** (includes PyPI **`mujoco`** for `gym.make("HalfCheetah-v5")`). Or: `pip install mujoco` / `pip install "gymnasium[mujoco]"`.
- **`mujoco_py` / Cython errors** → This path uses **Minari only**; avoid installing legacy D4RL that pulls `mujoco_py`. See **[SETUP.md](../SETUP.md)** troubleshooting.
- **More environments (Hopper, Walker, Ant)** → Not wired in this repo yet; README mentions them as future/extension. This doc is accurate for **HalfCheetah** only.

For the full generic setup walkthrough, see **[SETUP.md](../SETUP.md)** (Step 3–5, Path A).
