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

# ManiSkill ICL (PickCube-v1): expects paths.data_root/maniskill/PickCube-v1/trajectories.h5 (or .pkl)
# Generate data: python scripts/maniskill/ppo_train_icldata.py --env-id PickCube-v1 --icl-data-root <same as paths.data_root>
# State-only trajectories (no RGB): keep use_vision=false. If trajectories.h5 has images, set both to true.
uv run python -m src.train \
  --override data=[base,maniskill_pickcube] \
  experiment.max_steps=100000 \
  data.batch_size=4 \
  data.num_context_trajectories=3 \
  experiment.eval_every_steps=500 \
  experiment.eval_num_trials=3 \
  experiment.eval_target_return=50 \
  experiment.num_eval_rollouts=3 \
  data.rtg_scale=1.0 \
  data.horizon=20 \
  model.max_length=20 \
  model.use_vision=false \
  data.use_vision=false \
  model.query_loss_only=false \
  optim.lr=1e-4 \
  system.run_name=maniskill-pickcube_rtg1_ctx3_lr_1e-4 \
  paths.data_root=/scr2/shared/icl_adaptation/datasets \
  --wandb