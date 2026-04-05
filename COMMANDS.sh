uv run python -m src.train \
  --override data=[base,halfcheetah] \
  experiment.max_steps=20000 \
  data.batch_size=8 \
  paths.data_root=/scr2/shared/icl_adaptation/datasets \
  experiment.eval_every_steps=1000 \
  model.query_loss_only=false \
  --wandb 


uv run python -m src.train \
  --override data=[base,vd4rl] \
  --override model=transformer \
  --override data.use_vision=false \
  data.batch_size=16 \
  paths.data_root=/scr2/shared/icl_adaptation/datasets/vd4rl \
  experiment.eval_every_steps=500 \
  model.query_loss_only=false \
  --wandb 

# Regular DT
CUDA_VISIBLE_DEVICES=7 uv run python -m src.train \
  --override data=[base,halfcheetah] \
  experiment.max_steps=100000 \
  data.batch_size=64 \
  data.num_context_trajectories=0 \
  experiment.eval_every_steps=2000 \
  experiment.eval_num_trials=1 \
  experiment.eval_target_return=6000 \
  experiment.num_eval_rollouts=3 \
  data.rtg_scale=1000 \
  data.data_quality=medium_expert \
  data.horizon=20 \
  model.max_length=20 \
  model.query_loss_only=false \
  optim.lr=1e-4 \
  system.run_name=halfcheetah-medium-expert_rtg_1000_lr_1e-4 \
  paths.data_root=/scr2/shared/icl_adaptation/datasets \
  --wandb

# Offline rollouts + viz → <experiment_root>/offline_eval/ (paths.data_root from checkpoint + Hydra merge)
uv run python -m src.eval \
  --checkpoint outputs/icl_adaptation/2026-03-31/train__seed_412/ckpts/step_000000/checkpoint.pt \
  --step 0 \
  --num-episodes 3

# Multiple trial DT
# HalfCheetah multi-pool: use Hydra list (no comma ambiguity): data.data_quality=[medium,medium_expert]
uv run python -m src.train \
  --override data=[base,halfcheetah] \
  experiment.max_steps=100000 \
  data.batch_size=4 \
  data.num_context_trajectories=3 \
  experiment.eval_every_steps=10 \
  experiment.eval_num_trials=3 \
  experiment.eval_target_return=6000 \
  experiment.num_eval_rollouts=3 \
  data.rtg_scale=1000 \
  data.data_quality=[medium,medium_expert] \
  data.horizon=20 \
  model.max_length=20 \
  model.query_loss_only=false \
  optim.lr=1e-4 \
  system.run_name=halfcheetah-medium-expert_rtg_1000_lr_1e-4_context_3 \
  paths.data_root=/scr2/shared/icl_adaptation/datasets \
  --wandb

# ManiSkill ICL (PickCube-v1): reads paths.data_root/maniskill/PickCube-v1/trajectories.h5.
# PickCube PPO (scripts/maniskill/ppo_train_icldata.py): by default stitches the full on-policy rollout
# stream into that HDF5 (--icl-save-rollout-buffer, on by default). Set --icl-data-root to the same path as
# paths.data_root below (not a separate "replay buffer" file). Optional: --icl-collect-episodes for final-policy
# RGB episodes; --no-icl-save-rollout-buffer if you only want collect-episodes export.
# State-only: keep model.use_vision/data.use_vision false unless trajectories include images.
#
# Use .venv-maniskill (not `uv run`): periodic eval imports mani_skill in the same interpreter as train.
# Prereq from repo root: pip install -r scripts/maniskill/requirements.txt &&
#   pip install -r scripts/maniskill/requirements_icl_train.txt
#   (inside .venv-maniskill). Then:
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
./.venv-maniskill/bin/python -m src.train \
  --override data=[base,maniskill_pickcube] \
  experiment.max_steps=100000 \
  data.batch_size=16 \
  data.num_context_trajectories=5 \
  experiment.eval_every_steps=10 \
  experiment.eval_num_trials=3 \
  experiment.eval_target_return=20 \
  experiment.num_eval_rollouts=3 \
  data.rtg_scale=1.0 \
  data.horizon=20 \
  model.max_length=20 \
  model.use_vision=false \
  data.use_vision=false \
  model.query_loss_only=false \
  optim.lr=1e-4 \
  system.run_name=maniskill-pickcube_rtg1_ctx5_lr_1e-4 \
  paths.data_root=/scr2/shared/icl_adaptation/datasets \
  --wandb

# ManiSkill PickCube: same data as above but **no in-context demos** — standard DT with RTG conditioning
# (data.num_context_trajectories=0 -> max_total_prompt_length resolves to 0; query segment only).
# data.context_style=subsampled matches fixed-horizon DT segments (ignored for prompt when N=0).
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=3 ./.venv-maniskill/bin/python -m src.train \
  --override data=[base,maniskill_pickcube] \
  experiment.max_steps=100000 \
  data.batch_size=256 \
  data.num_context_trajectories=0 \
  data.randomize_num_context_trajectories=false \
  data.context_style=subsampled \
  experiment.eval_every_steps=1000 \
  experiment.eval_num_trials=1 \
  experiment.eval_target_return=20 \
  experiment.num_eval_rollouts=4 \
  data.rtg_scale=1.0 \
  data.horizon=20 \
  model.max_length=20 \
  model.use_vision=false \
  data.use_vision=false \
  model.query_loss_only=false \
  optim.lr=1e-4 \
  system.run_name=maniskill-pickcube_dt_rtg_only_lr_1e-4 \
  paths.data_root=/scr2/shared/icl_adaptation/datasets \
  --wandb