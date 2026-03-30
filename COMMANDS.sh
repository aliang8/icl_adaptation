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

CUDA_VISIBLE_DEVICES=7 uv run python -m src.train \
  --override data=[base,halfcheetah] \
  experiment.max_steps=20000 \
  data.batch_size=32 \
  data.num_context_trajectories=0 \
  experiment.eval_every_steps=1000 \
  experiment.eval_num_trials=1 \
  experiment.eval_target_return=6000 \
  data.rtg_scale=1 \
  data.data_quality=expert \
  data.horizon=100 \
  model.max_length=100 \
  model.query_loss_only=false \
  paths.data_root=/scr2/shared/icl_adaptation/datasets \
  --wandb