"""
PPO training for ManiSkill (from upstream ``examples/baselines/ppo/ppo.py``) with ICL export.

Writes ``<icl_data_root>/maniskill/<env_id>/trajectories_state_5M.h5`` by default — flat HDF5 (concatenated
timesteps, ``episode_starts`` / ``episode_lengths``, gzip chunks) for ``get_icl_trajectory_dataset`` /
``train.py`` (``data.env_name=ManiSkill/<env_id>``).

With ``--icl-shard-max-episodes N`` (N>0), rollout episodes go to rotating files
``trajectories_shard_00000.h5``, ``trajectories_shard_00001.h5``, … (each at most N episodes, **training
completion order** within a shard; no sort-by-return). Optional ``--icl-shard-ram-flush-episodes K`` (K>0)
writes the **current** shard incrementally (first chunk creates the HDF5, later chunks **append**) so RAM
need not hold a full N-episode shard. A small ``icl_shards_manifest.json`` lists shard names for merging
or multi-file training. Optional ``--icl-save-episode-fraction F`` (0<F≤1, default 1) subsamples **completed
on-policy rollout** episodes before HDF5 export (e.g. ``0.33`` keeps about one third on average); does not
affect PPO learning, RGB snapshots, or ``--icl-collect-episodes``.

By default **stitches the full on-policy PPO rollout stream** (all training iterations) into
episode dicts: **state**, **actions**, **rewards** (same per-step values as PPO learning: optional terminal ``--success-reward-bonus``, then ``--reward_scale``),
plus optional **episode_meta** (from ``final_info`` when present; **return** / **r** / mean **reward**
scaled like PPO: +``success_reward_bonus`` on success then ×``reward_scale``).
**RGB:** (1) use ``--icl-image-snapshot-every-steps N`` … periodic **snapshot** rollouts → ``trajectories_image_shard_*.h5``,
or (2) ``--icl-rollout-render-rgb`` to ``render()`` every **training** rollout step and embed frames in the **same**
stitched on-policy episodes as state (``images_view_*`` in ``trajectories_shard_*.h5`` / final HDF5). (2) disables (1).
For **snapshot** RGB only: if ``--icl-image-snapshot-shard-max-episodes`` is 0 but ``--icl-shard-max-episodes`` is set,
snapshot shards use ``min(icl_shard_max_episodes, 1000)`` episodes per file.
Optional
``--icl-collect-episodes`` appends final-policy RGB
episodes into the main ``trajectories.h5``. Use ``--icl-rgb-resize-hw 128`` (default) to store frames at
128×128 (human render camera is set to that size in ManiSkill so ``env.render()`` is not full-res then resized).
Set ``0`` for native task camera resolution. RGB is embedded in the same ``.h5`` as ``images_view_*``.
``--icl-snapshot-hdf5-image-compression`` (``gzip`` / ``lzf`` / ``none``) applies to ``images_view_*``
on every export path that writes RGB (snapshots, final file, shards, ``--icl-export-only``).
RGB rollouts use ``--icl-rgb-collect-num-envs`` parallel ManiSkill envs (GPU ``physx_cuda``; ``physx_cpu`` forces 1).

Requires: ``pip install mani-skill tyro`` (see ``docs/MANISKILL.md``). Weights & Biases is **on** by default
(``--track``); use ``--no-track`` for offline runs. Eval runs every ``eval_freq`` policy iterations and is logged
as ``eval/*`` (and ``train/*`` episode metrics when episodes complete during rollouts).
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Any, Optional

import gymnasium as gym

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.ic_replay_buffer_hdf5 import (
    append_ppo_rollout_to_episode_buffers,
    append_trajectories_hdf5,
    flush_episode_buffers,
    render_batch_to_rgb_list,
    save_trajectories_hdf5,
)
from src.data.maniskill_io import (
    collect_episodes_vector_env,
    episode_meta_from_final_info,
    episode_success_from_batched_final_info,
    scale_episode_meta_for_icl_export,
)


def _icl_task_dir(icl_data_root: str, env_id: str) -> _Path:
    safe = env_id.replace("/", "_").replace(" ", "_")
    return _Path(icl_data_root).expanduser().resolve() / "maniskill" / safe


def _icl_state_shard_path(icl_data_root: str, env_id: str, shard_index: int) -> _Path:
    return _icl_task_dir(icl_data_root, env_id) / f"trajectories_shard_{int(shard_index):05d}.h5"


def _icl_monolith_export_path(icl_data_root: str, env_id: str) -> _Path:
    """Single HDF5 when not using rollout shards (legacy on-disk name)."""
    return _icl_task_dir(icl_data_root, env_id) / "trajectories_state_5M.h5"


def _icl_image_snapshot_shard_path(icl_data_root: str, env_id: str, shard_index: int) -> _Path:
    d = _icl_task_dir(icl_data_root, env_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"trajectories_image_shard_{int(shard_index):05d}.h5"

# ManiSkill specific imports
import mani_skill.envs  # noqa: F401 — register envs
import src.envs.maniskill_pickcube_placed_only  # noqa: F401 — PickCube-v1-PlaceOnly
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project; None uses your default W&B account"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """If True, parallel envs reset on task ``terminated`` (e.g. early success). If False (default),
    ``ManiSkillVectorEnv`` uses ``ignore_terminations=True``: episodes continue until time-limit
    ``truncated`` at max horizon. Override with ``--partial-reset``."""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    sim_backend: str = "physx_cuda"
    """Simulation backend for ``gym.make`` (W&B ``gpu`` == ``physx_cuda`` in ManiSkill)."""
    reward_mode: Optional[str] = "dense"
    """If set (e.g. ``normalized_dense``), passed to ``gym.make``; ``None`` uses the task default."""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor (applied after optional ``success_reward_bonus``)."""
    success_reward_bonus: float = 0.0
    """If non-zero, add this to the **env** reward on the terminal step when ``final_info`` reports success (``success_once`` / ``success_at_end`` / ``success``). Applied **before** ``reward_scale`` on PPO rollouts and in ``collect_episodes_vector_env`` (then ``reward_scale`` is applied to every stored step), so **RGB snapshot** shards match PPO step rewards. Same for ``--icl-collect-episodes`` / ``--icl-export-only``."""
    eval_freq: int = 25
    """run vectorized eval every this many policy **iterations** (update steps), starting at iteration 1"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # ICL dataset export (this repo): see module docstring
    icl_data_root: str = "datasets"
    """Root under cwd; writes under ``<icl_data_root>/maniskill/<env_id>/``."""
    icl_shard_max_episodes: int = 0
    """If >0, each ``trajectories_shard_XXXXX.h5`` holds at most this many episodes (disk shard cap). 0=single file at end."""
    icl_shard_ram_flush_episodes: int = 0
    """If >0 (and ``icl_shard_max_episodes`` >0), flush at least this many completed episodes from RAM to the **current** shard file as soon as possible—creating the file with the first chunk, then **appending** until the shard reaches ``icl_shard_max_episodes``. Reduces peak RAM vs buffering a full shard. 0=only flush when a full shard of episodes is ready (legacy)."""
    icl_save_episode_fraction: float = 0.5
    """Each **completed** on-policy rollout episode is written to ICL HDF5 with this probability (seeded ``random``). ``1.0`` = all. E.g. ``0.33`` keeps about one third on average, reducing disk use. Does not apply to ``--icl-image-snapshot-*`` or ``--icl-collect-episodes``."""
    icl_save_rollout_buffer: bool = True
    """If True (training only), export on-policy PPO rollouts as stitched episodes (state only unless RGB rollout); use ``icl_save_episode_fraction`` to subsample."""
    icl_collect_episodes: int = 0
    """After training, roll out this many **additional** episodes with the final policy (RGB via render). 0=skip."""
    icl_max_steps_per_episode: int = 512
    """Max env steps per recorded episode (``icl_collect_episodes`` path only)."""
    icl_export_only: bool = False
    """If True, skip PPO; load ``--checkpoint`` and only write ICL ``trajectories.h5``."""
    icl_image_snapshot_every_steps: int = 0
    """If >0, collect RGB snapshots when training crosses each ``N`` env-step boundary (after update); shards need ``icl_image_snapshot_shard_max_episodes`` or ``icl_shard_max_episodes`` > 0."""
    icl_image_snapshot_episodes: int = 8
    """Episodes per RGB snapshot (``icl_image_snapshot_every_steps`` > 0)."""
    icl_image_snapshot_max_steps: int = 512
    """Max env steps per episode in RGB snapshots."""
    icl_image_snapshot_shard_max_episodes: int = 0
    """Episodes per ``trajectories_image_shard_*.h5``. If 0 and ``icl_shard_max_episodes`` > 0, uses ``min(icl_shard_max_episodes, 1000)`` for RGB (state shards can stay larger). Set explicitly for bigger RGB shards (more RAM before first flush)."""
    icl_rgb_resize_hw: int = 128
    """If >0, set ManiSkill ``
    human_render_camera_configs`` to H=W and store RGB at that size (resize is a no-op if render matches). 0=native."""
    icl_snapshot_hdf5_image_compression: str = "lzf"
    """``gzip`` / ``lzf`` / ``none`` for ``images_view_*`` whenever this script writes RGB to HDF5."""
    icl_rgb_collect_num_envs: int = 8
    """Parallel envs for RGB rollouts (snapshots + ``icl_collect_episodes``). 1=serial. >1 needs ``physx_cuda``."""
    icl_rollout_render_rgb: bool = False
    """If True with ``icl_save_rollout_buffer``, call ``env.render()`` every training rollout timestep and store RGB in the same ICL episodes (HDF5 ``images_view_*``). Needs batched ``rgb_array`` (``physx_cuda`` when ``num_envs``>1). **Disables** ``icl_image_snapshot_*``. Very slow at large ``num_envs`` — lower ``num_envs`` or use smaller ``icl_shard_max_episodes`` / ``icl_rollout_rgb_shard_max_episodes``."""
    icl_rollout_rgb_shard_max_episodes: int = 0
    """With ``icl_rollout_render_rgb`` only: optional **extra** cap. Flush uses ``min(icl_shard_max_episodes, this)`` when both >0; if this is 0, only ``icl_shard_max_episodes`` applies (same queue for state+RGB—no second shard series)."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def _icl_rgb_resize_hw(args: Args) -> Optional[int]:
    h = int(args.icl_rgb_resize_hw)
    return h if h > 0 else None


# When inheriting from ``icl_shard_max_episodes``, cap RGB separately: state shards are small per
# episode but buffered RGB is huge — inheriting 10_000 would mean no ``trajectories_image_shard_*.h5``
# until 10_000 snapshot episodes (or a clean process exit). SIGKILL then yields zero image files.
_ICL_RGB_SNAPSHOT_SHARD_CAP_WHEN_INHERITING = 1000


def _icl_rollout_shard_flush_cap(args: Args) -> int:
    """Episodes per ``trajectories_shard_*.h5`` flush when writing rollout ICL (state ± RGB)."""
    base = int(args.icl_shard_max_episodes)
    if not bool(args.icl_rollout_render_rgb):
        return base
    cap = int(args.icl_rollout_rgb_shard_max_episodes)
    if cap > 0 and base > 0:
        return min(base, cap)
    if cap > 0:
        return cap
    return base


def _icl_effective_image_snapshot_shard_cap(args: Args) -> int:
    """Episodes per RGB snapshot HDF5 file.

    Explicit ``icl_image_snapshot_shard_max_episodes`` wins. If 0 and ``icl_shard_max_episodes`` > 0,
    use ``min(icl_shard_max_episodes, _ICL_RGB_SNAPSHOT_SHARD_CAP_WHEN_INHERITING)`` so image shards
    flush to disk without waiting for a full state-shard-sized RGB buffer.
    """
    k = int(args.icl_image_snapshot_shard_max_episodes)
    if k > 0:
        return k
    r = int(args.icl_shard_max_episodes)
    if r <= 0:
        return 0
    return min(r, _ICL_RGB_SNAPSHOT_SHARD_CAP_WHEN_INHERITING)


def _mani_skill_env_kwargs(args: Args) -> dict:
    kw = dict(obs_mode="state", render_mode="rgb_array", sim_backend=args.sim_backend)
    if args.reward_mode is not None:
        kw["reward_mode"] = args.reward_mode
    if args.control_mode is not None:
        kw["control_mode"] = args.control_mode
    rh = int(args.icl_rgb_resize_hw)
    if rh > 0:
        # Avoid full-res GPU render + CPU/GPU downscale: match rgb_array camera to HDF5 target (e.g. 128).
        kw["human_render_camera_configs"] = dict(width=rh, height=rh)
    return kw


@dataclass
class _IclRolloutShardState:
    """Tracks the open rollout shard HDF5 (append after first ``save_trajectories_hdf5``)."""

    open_shard_index: int = 0
    n_episodes_in_open_shard: int = 0


def _icl_rollout_shard_file_count(state: _IclRolloutShardState) -> int:
    return int(state.open_shard_index) + (1 if int(state.n_episodes_in_open_shard) > 0 else 0)


def _icl_flush_rollout_shards_incremental(
    episodes: list,
    state: _IclRolloutShardState,
    *,
    shard_max: int,
    ram_flush_every: int,
    end_of_run: bool,
    icl_data_root: str,
    env_id: str,
    image_hdf5_compression: str,
) -> None:
    """Move episodes from RAM to shard HDF5s (mutates ``episodes`` and ``state``).

    With ``ram_flush_every`` > 0 and ``end_of_run`` False, writes chunks of up to that many
    episodes when possible, appending to the current shard until it holds ``shard_max`` episodes,
    then rotates to the next file. With ``ram_flush_every`` 0, only flushes when at least
    ``room`` episodes are available (legacy full-shard-from-RAM behavior, but ``room`` respects
    a partially filled on-disk shard).
    """
    if int(shard_max) <= 0 or not episodes:
        return
    ram_flush_every = int(ram_flush_every)

    while episodes:
        room = int(shard_max) - int(state.n_episodes_in_open_shard)
        if room <= 0:
            state.open_shard_index += 1
            state.n_episodes_in_open_shard = 0
            room = int(shard_max)
        if end_of_run:
            take = min(len(episodes), room)
        elif ram_flush_every > 0:
            if len(episodes) >= room:
                take = room
            elif len(episodes) >= ram_flush_every:
                take = min(ram_flush_every, room, len(episodes))
            else:
                break
        else:
            if len(episodes) >= room:
                take = room
            else:
                break

        chunk = episodes[:take]
        del episodes[:take]
        p = _icl_state_shard_path(icl_data_root, env_id, state.open_shard_index)
        if int(state.n_episodes_in_open_shard) == 0:
            save_trajectories_hdf5(
                chunk,
                p,
                sort_by_return=False,
                image_hdf5_compression=str(image_hdf5_compression),
            )
            how = "wrote"
        else:
            append_trajectories_hdf5(p, chunk)
            how = "appended"
        state.n_episodes_in_open_shard += int(take)
        print(
            f"[ICL shard] {how} episodes={len(chunk)} -> {p.name} "
            f"(episodes_in_shard={state.n_episodes_in_open_shard}/{shard_max})",
            flush=True,
        )
        if int(state.n_episodes_in_open_shard) >= int(shard_max):
            state.open_shard_index += 1
            state.n_episodes_in_open_shard = 0


def _icl_write_shards_manifest(
    icl_data_root: str,
    env_id: str,
    shard_max_episodes: int,
    n_shard_files: int,
) -> None:
    if n_shard_files <= 0:
        return
    base = _icl_task_dir(icl_data_root, env_id)
    base.mkdir(parents=True, exist_ok=True)
    files = [f"trajectories_shard_{i:05d}.h5" for i in range(n_shard_files)]
    doc = {
        "env_id": env_id,
        "icl_shard_max_episodes": int(shard_max_episodes),
        "n_shards": int(n_shard_files),
        "h5_files": files,
        "note": "RGB is embedded as images_view_* in each .h5 when present.",
    }
    mp = base / "icl_shards_manifest.json"
    mp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"[ICL] Wrote {mp.resolve()}", flush=True)


def _icl_flush_image_snapshot_shards(
    buf: list,
    *,
    shard_max: int,
    icl_data_root: str,
    env_id: str,
    image_compression: str,
    next_shard_index: int,
) -> int:
    """Write as many full ``shard_max``-episode shards from ``buf`` as possible; return next shard index."""
    while int(shard_max) > 0 and len(buf) >= int(shard_max):
        chunk = buf[: int(shard_max)]
        del buf[: int(shard_max)]
        p = _icl_image_snapshot_shard_path(icl_data_root, env_id, next_shard_index)
        save_trajectories_hdf5(
            chunk,
            p,
            sort_by_return=True,
            image_hdf5_compression=str(image_compression),
        )
        print(
            f"[ICL image snapshot shard] episodes={len(chunk)} -> {p.name}",
            flush=True,
        )
        next_shard_index += 1
    return next_shard_index


def _make_rgb_collect_vector_env(args: Args, num_episodes_cap: int, *, tag: str = "") -> Any:
    """Vector env for ``collect_episodes_vector_env`` (batched RGB). ``num_envs>1`` requires GPU sim."""
    want = max(1, min(int(args.icl_rgb_collect_num_envs), int(num_episodes_cap)))
    kw = _mani_skill_env_kwargs(args)
    n = want
    sim = str(args.sim_backend).strip().lower()
    if sim == "physx_cpu":
        if n > 1:
            print(
                f"[ICL RGB{tag}] physx_cpu: using num_envs=1 (parallel RGB needs physx_cuda).",
                flush=True,
            )
        n = 1
    kw["reconfiguration_freq"] = 0 if n > 1 else 1
    env = gym.make(args.env_id, num_envs=n, **kw)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, n, ignore_terminations=False, record_metrics=True)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(
                nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01 * np.sqrt(2)
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb as _wandb

            _wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()
        if self.log_wandb:
            import wandb as _wandb

            _wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if bool(args.icl_rollout_render_rgb):
        if not bool(args.icl_save_rollout_buffer) or bool(args.evaluate):
            raise SystemExit(
                "icl_rollout_render_rgb requires --icl-save-rollout-buffer (default) and not --evaluate"
            )
        if int(args.icl_image_snapshot_every_steps) > 0:
            print(
                "[ICL] icl_rollout_render_rgb is on: ignoring --icl-image-snapshot-* (no snapshot collection).",
                flush=True,
            )
        sim_be = str(args.sim_backend).strip().lower()
        if sim_be == "physx_cpu" and int(args.num_envs) > 1:
            raise SystemExit(
                "icl_rollout_render_rgb with num_envs>1 requires physx_cuda for batched rgb_array render"
            )
        rcfg = args.reconfiguration_freq
        if rcfg is not None and int(rcfg) != 0:
            print(
                "[ICL] Warning: icl_rollout_render_rgb is most reliable with --reconfiguration-freq 0 "
                "(batched render across envs).",
                flush=True,
            )
    if int(args.icl_shard_ram_flush_episodes) < 0:
        raise SystemExit("icl_shard_ram_flush_episodes must be >= 0")
    if int(args.icl_shard_ram_flush_episodes) > 0 and int(args.icl_shard_max_episodes) <= 0:
        raise SystemExit(
            "icl_shard_ram_flush_episodes requires icl_shard_max_episodes > 0 (per-file episode cap)"
        )
    _save_frac = float(args.icl_save_episode_fraction)
    if _save_frac <= 0.0 or _save_frac > 1.0:
        raise SystemExit("icl_save_episode_fraction must be in (0, 1]")
    if _save_frac < 1.0:
        print(
            f"[ICL] icl_save_episode_fraction={_save_frac} (on-policy rollout HDF5 only; PPO unchanged)",
            flush=True,
        )
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.icl_export_only:
        if not args.checkpoint:
            raise SystemExit("icl_export_only requires --checkpoint")
        if args.icl_collect_episodes <= 0:
            raise SystemExit(
                "icl_export_only requires --icl-collect-episodes >= 1 "
                "(no training rollouts to stitch in export-only mode)."
            )
        env_one = _make_rgb_collect_vector_env(args, args.icl_collect_episodes, tag=" export_only")
        agent = Agent(env_one).to(device)
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        al = torch.from_numpy(env_one.single_action_space.low).to(device)
        ah = torch.from_numpy(env_one.single_action_space.high).to(device)
        _rsz = _icl_rgb_resize_hw(args)
        t0 = time.perf_counter()
        trajs = collect_episodes_vector_env(
            env_one,
            agent,
            device,
            args.icl_collect_episodes,
            args.icl_max_steps_per_episode,
            al,
            ah,
            rgb_resize_hw=_rsz,
            success_reward_bonus=float(args.success_reward_bonus),
            reward_scale=float(args.reward_scale),
        )
        t1 = time.perf_counter()
        print(
            f"[ICL export_only] collection finished: episodes={len(trajs)} "
            f"collect_s={t1 - t0:.2f} -> writing HDF5...",
            flush=True,
        )
        out_path = _icl_monolith_export_path(args.icl_data_root, args.env_id)
        save_trajectories_hdf5(
            trajs,
            out_path,
            image_hdf5_compression=str(args.icl_snapshot_hdf5_image_compression),
        )
        t2 = time.perf_counter()
        print(
            f"[ICL export_only] Saved {len(trajs)} trajectories to {out_path.resolve()} "
            f"(rgb_hw={_rsz or 'native'} collect_s={t1 - t0:.2f} hdf5_s={t2 - t1:.2f} total_s={t2 - t0:.2f})"
        )
        env_one.close()
        sys.exit(0)

    # env setup
    env_kwargs = _mani_skill_env_kwargs(args)
    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )
    envs = ManiSkillVectorEnv(
        envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs,
        args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb

            config = vars(args)
            _rm = args.reward_mode if args.reward_mode is not None else "normalized_dense"
            # Merge without duplicate keys: env_kwargs may already contain reward_mode (W&B repro JSON).
            _train_log = {**env_kwargs}
            _train_log.update(
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode=_rm,
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            config["env_cfg"] = _train_log
            _eval_log = {**env_kwargs}
            _eval_log.update(
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                reward_mode=_rm,
                env_horizon=max_episode_steps,
                partial_reset=args.eval_partial_reset,
            )
            config["eval_env_cfg"] = _eval_log
            _wb_kw = dict(
                project=args.wandb_project_name,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"],
            )
            if args.wandb_entity:
                _wb_kw["entity"] = args.wandb_entity
            wandb.init(**_wb_kw)
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # Per-step reward used for GAE, value targets, policy loss, and ICL rollout HDF5 (single source).
    # Filled in the rollout loop as: env r_t, optional +success_reward_bonus on terminal success, then *reward_scale.
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print("####")
    print(
        f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}"
    )
    print(
        f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}"
    )
    print("####")
    action_space_low, action_space_high = (
        torch.from_numpy(envs.single_action_space.low).to(device),
        torch.from_numpy(envs.single_action_space.high).to(device),
    )

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    icl_trajs: list = []
    icl_buffers: list = []
    if args.icl_save_rollout_buffer and not args.evaluate:
        if bool(args.icl_rollout_render_rgb):
            icl_buffers = [{"obs": [], "act": [], "rew": [], "rgb": []} for _ in range(args.num_envs)]
        else:
            icl_buffers = [{"obs": [], "act": [], "rew": []} for _ in range(args.num_envs)]

    snapshot_cenv = None
    snapshot_al, snapshot_ah = None, None
    icl_image_snapshot_shard_cap = 0
    if (
        not args.evaluate
        and not bool(args.icl_rollout_render_rgb)
        and args.icl_image_snapshot_every_steps > 0
        and args.icl_image_snapshot_episodes > 0
    ):
        icl_image_snapshot_shard_cap = _icl_effective_image_snapshot_shard_cap(args)
        if icl_image_snapshot_shard_cap <= 0:
            raise SystemExit(
                "RGB snapshots only write trajectories_image_shard_*.h5 in the ManiSkill task folder. "
                "Set --icl-image-snapshot-shard-max-episodes K or --icl-shard-max-episodes K "
                "(the latter is used when the former is 0)."
            )
        snapshot_cenv = _make_rgb_collect_vector_env(
            args, args.icl_image_snapshot_episodes, tag=" snapshot"
        )
        snapshot_al = torch.from_numpy(snapshot_cenv.single_action_space.low).to(device)
        snapshot_ah = torch.from_numpy(snapshot_cenv.single_action_space.high).to(device)

    last_icl_image_snap_boundary = -1
    icl_image_snapbuf: list = []
    icl_image_snap_shard_next = 0
    icl_rollout_shard_state = _IclRolloutShardState()
    sh_max = int(args.icl_shard_max_episodes)
    sh_flush = _icl_rollout_shard_flush_cap(args)
    if sh_max < 0:
        raise SystemExit("icl_shard_max_episodes must be >= 0")
    if bool(args.icl_rollout_render_rgb) and sh_flush <= 0:
        print(
            "[ICL] Warning: icl_rollout_render_rgb with icl_shard_max_episodes=0 and "
            "icl_rollout_rgb_shard_max_episodes=0 — all RGB rollout episodes stay in RAM until training ends.",
            flush=True,
        )

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if (iteration - 1) % args.eval_freq == 0:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = (
                        eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    )
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += int(mask.sum().item())
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(
                f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes"
            )
            if logger is not None:
                logger.add_scalar("eval/iteration", float(iteration), global_step)
                logger.add_scalar("eval/num_completed_episodes", float(num_episodes), global_step)
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if args.evaluate:
                break
        if args.save_model and (iteration - 1) % args.eval_freq == 0:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_time = time.time()
        icl_meta_grid = None
        if args.icl_save_rollout_buffer and not args.evaluate:
            episode_done_after = torch.zeros((args.num_steps, args.num_envs), device=device)
            icl_meta_grid = np.full((args.num_steps, args.num_envs), None, dtype=object)
        # One (S,N,H,W,3) buffer per rollout: avoids list-of-rows + np.stack copy and cuts allocator churn.
        rollout_rgb_grid: Optional[np.ndarray] = None
        _rh_pre = _icl_rgb_resize_hw(args)
        if args.icl_save_rollout_buffer and not args.evaluate and args.icl_rollout_render_rgb:
            if _rh_pre is not None:
                rh = int(_rh_pre)
                rollout_rgb_grid = np.zeros(
                    (int(args.num_steps), int(args.num_envs), rh, rh, 3), dtype=np.uint8
                )
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            if args.icl_save_rollout_buffer and not args.evaluate and args.icl_rollout_render_rgb:
                try:
                    _fr = envs.render()
                except Exception as _e:
                    raise RuntimeError(
                        "icl_rollout_render_rgb: envs.render() failed (need render_mode rgb_array on training env)"
                    ) from _e
                _rsz_rr = _icl_rgb_resize_hw(args)
                _row = render_batch_to_rgb_list(_fr, int(args.num_envs), _rsz_rr)
                row = np.stack(_row, axis=0)
                if rollout_rgb_grid is None:
                    h, w = int(row.shape[1]), int(row.shape[2])
                    rollout_rgb_grid = np.zeros(
                        (int(args.num_steps), int(args.num_envs), h, w, 3), dtype=np.uint8
                    )
                rollout_rgb_grid[step] = row

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            r_step = reward.view(-1).float()
            sb = float(args.success_reward_bonus)
            if sb != 0.0 and "final_info" in infos:
                fi = infos["final_info"]
                dm = infos["_final_info"]
                if isinstance(fi.get("episode"), dict):
                    ep = fi["episode"]
                    for e in range(args.num_envs):
                        if bool(dm[e].item() if torch.is_tensor(dm) else dm[e]):
                            if episode_success_from_batched_final_info(ep, e):
                                r_step[e] += sb
            # Learner signal (and ICL): (env + bonus on last step if success) * scale — not raw env reward.
            rewards[step] = r_step * float(args.reward_scale)
            if args.icl_save_rollout_buffer and not args.evaluate:
                episode_done_after[step] = next_done

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                if logger is not None and isinstance(final_info.get("episode"), dict):
                    for k, v in final_info["episode"].items():
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                if icl_meta_grid is not None and isinstance(final_info.get("episode"), dict):
                    ep = final_info["episode"]
                    for e in range(args.num_envs):
                        if bool(
                            done_mask[e].item() if torch.is_tensor(done_mask) else done_mask[e]
                        ):
                            raw_meta = episode_meta_from_final_info(ep, e)
                            succ = episode_success_from_batched_final_info(ep, e)
                            icl_meta_grid[step, e] = scale_episode_meta_for_icl_export(
                                raw_meta,
                                reward_scale=float(args.reward_scale),
                                success_reward_bonus=float(args.success_reward_bonus),
                                success=succ,
                            )
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = (
                        agent.get_value(infos["final_observation"][done_mask]).view(-1)
                    )
        rollout_time = time.time() - rollout_time
        if args.icl_save_rollout_buffer and not args.evaluate:
            # `rewards` above: same (bonus then scale) tensor as GAE/returns; stored as-is in shards / HDF5.
            _rgb_grid = None
            if args.icl_rollout_render_rgb:
                if rollout_rgb_grid is None:
                    raise RuntimeError("icl_rollout_render_rgb: RGB grid was never allocated")
                _rgb_grid = rollout_rgb_grid
            append_ppo_rollout_to_episode_buffers(
                obs.detach().cpu().numpy(),
                actions.detach().cpu().numpy(),
                rewards.detach().cpu().numpy(),
                (episode_done_after.detach().cpu().numpy() > 0.5),
                icl_buffers,
                icl_trajs,
                episode_meta_grid=icl_meta_grid,
                rgb_grid=_rgb_grid,
                episode_keep_fraction=_save_frac,
            )
            if sh_flush > 0:
                _icl_flush_rollout_shards_incremental(
                    icl_trajs,
                    icl_rollout_shard_state,
                    shard_max=sh_flush,
                    ram_flush_every=int(args.icl_shard_ram_flush_episodes),
                    end_of_run=False,
                    icl_data_root=args.icl_data_root,
                    env_id=args.env_id,
                    image_hdf5_compression=str(args.icl_snapshot_hdf5_image_compression),
                )
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]  # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1:  # initialize
                        lam_coef_sum = 0.0
                        reward_term_sum = 0.0  # the sum of the second term
                        value_term_sum = 0.0  # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = (
                        args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    )
                    value_term_sum = (
                        args.gae_lambda * args.gamma * value_term_sum
                        + args.gamma * real_next_values
                    )

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                    )  # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar(
            "time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step
        )

        if snapshot_cenv is not None and not bool(args.icl_rollout_render_rgb):
            Nsnap = int(args.icl_image_snapshot_every_steps)
            boundary = (global_step // Nsnap) * Nsnap
            if boundary > 0 and boundary > last_icl_image_snap_boundary:
                _rsz_snap = _icl_rgb_resize_hw(args)
                print(
                    f"[ICL image snapshot] start: step_boundary={boundary} global_step={global_step} "
                    f"episodes={args.icl_image_snapshot_episodes} max_steps={args.icl_image_snapshot_max_steps} "
                    f"num_envs={snapshot_cenv.num_envs} rgb_hw={_rsz_snap or 'native'} "
                    f"shard_cap={icl_image_snapshot_shard_cap} next_shard={icl_image_snap_shard_next:05d} "
                    f"stem=trajectories_image_shard_*.h5",
                    flush=True,
                )
                agent.eval()
                t_snap0 = time.perf_counter()
                snap_trajs = collect_episodes_vector_env(
                    snapshot_cenv,
                    agent,
                    device,
                    args.icl_image_snapshot_episodes,
                    args.icl_image_snapshot_max_steps,
                    snapshot_al,
                    snapshot_ah,
                    rgb_resize_hw=_rsz_snap,
                    success_reward_bonus=float(args.success_reward_bonus),
                    reward_scale=float(args.reward_scale),
                )
                t_snap1 = time.perf_counter()
                print(
                    f"[ICL image snapshot] collection finished: episodes={len(snap_trajs)} "
                    f"collect_s={t_snap1 - t_snap0:.2f} -> buffer/flush "
                    f"(shard_cap={icl_image_snapshot_shard_cap} episodes, "
                    f"compression={args.icl_snapshot_hdf5_image_compression})...",
                    flush=True,
                )
                icl_image_snapbuf.extend(snap_trajs)
                t_w0 = time.perf_counter()
                _shard_before = icl_image_snap_shard_next
                icl_image_snap_shard_next = _icl_flush_image_snapshot_shards(
                    icl_image_snapbuf,
                    shard_max=icl_image_snapshot_shard_cap,
                    icl_data_root=args.icl_data_root,
                    env_id=args.env_id,
                    image_compression=str(args.icl_snapshot_hdf5_image_compression),
                    next_shard_index=icl_image_snap_shard_next,
                )
                t_w1 = time.perf_counter()
                _n_flushed = icl_image_snap_shard_next - _shard_before
                print(
                    f"[ICL image snapshot] done: recorded_episodes={len(snap_trajs)} "
                    f"step_boundary={boundary} global_step={global_step} "
                    f"rgb_buffer_episodes={len(icl_image_snapbuf)} "
                    f"full_shards_written_this_round={_n_flushed} "
                    f"collect_s={t_snap1 - t_snap0:.2f} flush_s={t_w1 - t_w0:.2f} "
                    f"total_s={t_w1 - t_snap0:.2f}",
                    flush=True,
                )
                last_icl_image_snap_boundary = boundary

    if snapshot_cenv is not None and icl_image_snapbuf and not bool(args.icl_rollout_render_rgb):
        n_tail = len(icl_image_snapbuf)
        p_tail = _icl_image_snapshot_shard_path(
            args.icl_data_root, args.env_id, icl_image_snap_shard_next
        )
        t_tail0 = time.perf_counter()
        save_trajectories_hdf5(
            icl_image_snapbuf,
            p_tail,
            sort_by_return=True,
            image_hdf5_compression=str(args.icl_snapshot_hdf5_image_compression),
        )
        t_tail1 = time.perf_counter()
        print(
            f"[ICL image snapshot shard] final partial episodes={n_tail} -> {p_tail.name} "
            f"hdf5_s={t_tail1 - t_tail0:.2f}",
            flush=True,
        )
        icl_image_snapbuf.clear()

    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        out_path = _icl_monolith_export_path(args.icl_data_root, args.env_id)
        _rsz = _icl_rgb_resize_hw(args)
        icl_extra_notes: list[str] = []
        if args.icl_save_rollout_buffer:
            flush_episode_buffers(icl_buffers, icl_trajs)
        if args.icl_collect_episodes > 0:
            cenv = _make_rgb_collect_vector_env(args, args.icl_collect_episodes, tag=" collect")
            al = torch.from_numpy(cenv.single_action_space.low).to(device)
            ah = torch.from_numpy(cenv.single_action_space.high).to(device)
            t_rgb0 = time.perf_counter()
            extra = collect_episodes_vector_env(
                cenv,
                agent,
                device,
                args.icl_collect_episodes,
                args.icl_max_steps_per_episode,
                al,
                ah,
                rgb_resize_hw=_rsz,
                success_reward_bonus=float(args.success_reward_bonus),
                reward_scale=float(args.reward_scale),
            )
            t_rgb1 = time.perf_counter()
            icl_trajs.extend(extra)
            cenv.close()
            icl_extra_notes.append(
                f"rgb_episodes={len(extra)} rgb_hw={_rsz or 'native'} rgb_collect_s={t_rgb1 - t_rgb0:.2f}"
            )

        if sh_flush > 0:
            t_h5_0 = time.perf_counter()
            _icl_flush_rollout_shards_incremental(
                icl_trajs,
                icl_rollout_shard_state,
                shard_max=sh_flush,
                ram_flush_every=int(args.icl_shard_ram_flush_episodes),
                end_of_run=True,
                icl_data_root=args.icl_data_root,
                env_id=args.env_id,
                image_hdf5_compression=str(args.icl_snapshot_hdf5_image_compression),
            )
            t_h5_1 = time.perf_counter()
            n_shard_files = _icl_rollout_shard_file_count(icl_rollout_shard_state)
            if n_shard_files > 0:
                _icl_write_shards_manifest(
                    args.icl_data_root,
                    args.env_id,
                    sh_flush,
                    n_shard_files,
                )
                tail = " ".join(icl_extra_notes + [f"flush_s={t_h5_1 - t_h5_0:.2f}"])
                print(
                    f"[ICL] Sharded export: {n_shard_files} file(s) under "
                    f"{out_path.parent.resolve()}  {tail}",
                    flush=True,
                )
            elif args.icl_save_rollout_buffer or args.icl_collect_episodes > 0:
                print(
                    "[ICL] No trajectories to save (enable --icl-save-rollout-buffer or --icl-collect-episodes).",
                    flush=True,
                )
        else:
            to_save = list(icl_trajs)
            if to_save:
                t_h5_0 = time.perf_counter()
                save_trajectories_hdf5(
                    to_save,
                    out_path,
                    image_hdf5_compression=str(args.icl_snapshot_hdf5_image_compression),
                )
                t_h5_1 = time.perf_counter()
                tail = " ".join(icl_extra_notes + [f"hdf5_s={t_h5_1 - t_h5_0:.2f}"])
                print(f"[ICL] Saved {len(to_save)} trajectories to {out_path.resolve()} {tail}")
            elif args.icl_save_rollout_buffer or args.icl_collect_episodes > 0:
                print(
                    "[ICL] No trajectories to save (enable --icl-save-rollout-buffer or --icl-collect-episodes)."
                )
        logger.close()
    if snapshot_cenv is not None:
        snapshot_cenv.close()
    envs.close()
    eval_envs.close()
