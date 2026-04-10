"""Eval visuals: frame overlays, vision embeddings, and RTG/reward matplotlib figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.meta_dt import MetaDecisionTransformer
from src.utils.eval_utils import pad_ragged_1d


def annotate_eval_frame(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    import cv2

    out = np.asarray(frame).copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = max(0.45, min(w, h) / 480.0)
    thick = max(1, int(sc * 2))
    y = int(18 * sc + 12)
    for line in lines:
        if not line:
            continue
        cv2.putText(out, line, (10, y), font, sc, (255, 255, 255), thick, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), font, sc, (0, 0, 0), max(1, thick - 1), cv2.LINE_AA)
        y += int(24 * sc + 10)
    return out


def cum_return_per_frame(frames: List, ep_rewards: List[float]) -> List[float]:
    if not frames:
        return []
    n_f, n_r = len(frames), len(ep_rewards)
    if n_f == n_r + 1:
        out = [0.0]
        cum = 0.0
        for i in range(n_r):
            cum += float(ep_rewards[i])
            out.append(cum)
        return out
    if n_f == n_r:
        out = []
        cum = 0.0
        for i in range(n_r):
            cum += float(ep_rewards[i])
            out.append(cum)
        return out
    cum = 0.0
    out = []
    for i in range(n_f):
        if i == 0:
            out.append(0.0)
        elif i - 1 < n_r:
            cum += float(ep_rewards[i - 1])
            out.append(cum)
        else:
            out.append(cum)
    return out


def annotated_rollout_frames(
    model: Any,
    frames: List[np.ndarray],
    rtg_per_frame: List[float],
    trial_tag: str,
    cum_return_per_frame_vals: Optional[List[float]] = None,
) -> List[np.ndarray]:
    if not isinstance(model, MetaDecisionTransformer):
        raise TypeError(f"eval expects MetaDecisionTransformer, got {type(model)}")
    layout = getattr(model, "_sequence_token_layout", "rtg_state_action")
    cond_rtg = layout == "rtg_state_action"
    out: List[np.ndarray] = []
    for t, f in enumerate(frames):
        lines = [f"{trial_tag}  t={t}"]
        if cond_rtg and t < len(rtg_per_frame):
            lines.append(f"RTG target: {rtg_per_frame[t]:.4f}")
        elif layout == "state_action_reward":
            lines.append("policy: (s,a,r) no RTG")
        else:
            lines.append("RTG: off")
        if cum_return_per_frame_vals is not None and t < len(cum_return_per_frame_vals):
            lines.append(f"return: {cum_return_per_frame_vals[t]:.2f}")
        out.append(annotate_eval_frame(f, lines))
    return out


def preprocess_frames_for_encoder(
    frames: List[np.ndarray],
    device: Any,
    size: Tuple[int, int],
) -> Any:
    if not frames:
        return None
    h, w = size
    try:
        import cv2

        resized = np.stack(
            [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames],
            axis=0,
        )
    except Exception:
        resized = np.stack([np.asarray(f) for f in frames], axis=0)
        if resized.shape[1:3] != (h, w):
            return None
    x = torch.from_numpy(resized).float().to(device)
    x = x.permute(0, 3, 1, 2)
    x = x / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


def raise_missing_eval_vision_images(
    *,
    model: Any,
    env_name: str,
    env: Any,
    timestep: int,
) -> None:
    use_vision_cfg = getattr(model, "use_vision", None)
    extra = ""
    if use_vision_cfg is not None:
        extra = f" model.use_vision={use_vision_cfg!r};"
    raise RuntimeError(
        "Vision inference requires camera frames from the eval environment, but "
        f"image_embeddings could not be built (timestep t={timestep}).{extra} "
        f"env_name={env_name!r} env_type={type(env).__name__}. "
        "Implement get_current_images() to return uint8 (H, W, 3) RGB (primary and optionally wrist), "
        "and for Gymnasium or ManiSkill create the env with render_mode='rgb_array' so render() "
        "returns pixels."
    )


def vision_encoder_num_views(vision_encoder: Any) -> int:
    nv = getattr(vision_encoder, "num_views", None)
    if nv is not None:
        return int(nv)
    inner = getattr(vision_encoder, "encoder", None)
    if inner is not None:
        nv2 = getattr(inner, "num_views", None)
        if nv2 is not None:
            return int(nv2)
    return 2


def encode_rollout_images(
    image_list: List[Tuple[Any, Any]],
    model: Any,
    device: Any,
) -> Optional[Any]:
    if not isinstance(model, MetaDecisionTransformer):
        raise TypeError(f"eval expects MetaDecisionTransformer, got {type(model)}")
    vision_encoder = model.vision_encoder
    if vision_encoder is None or not image_list:
        return None
    primary_frames = []
    wrist_frames = []
    for p, w in image_list:
        if p is not None:
            primary_frames.append(p)
        if w is not None:
            wrist_frames.append(w)
    if not primary_frames and not wrist_frames:
        return None
    if not primary_frames:
        primary_frames = list(wrist_frames)
    if not wrist_frames:
        wrist_frames = list(primary_frames)
    n_views = vision_encoder_num_views(vision_encoder)
    view_tensors = []
    enc_hw = getattr(model, "vision_encoder_img_size", None)
    if enc_hw is None:
        raise RuntimeError(
            "model.vision_encoder_img_size is required for eval image preprocessing "
            f"(got {type(model).__name__} with vision_encoder set)."
        )
    enc_size = (int(enc_hw[0]), int(enc_hw[1]))
    for vi in range(n_views):
        frames = primary_frames if vi == 0 else wrist_frames
        vt = preprocess_frames_for_encoder(frames, device, size=enc_size)
        if vt is None:
            return None
        view_tensors.append(vt)
    with torch.no_grad():
        emb = vision_encoder(view_tensors)
    if emb is not None and emb.dim() == 3:
        return emb
    return None


def save_eval_rtg_reward_figure(
    reward_rows: List[np.ndarray],
    rtg_rows: List[np.ndarray],
    *,
    rtg_scale: float,
    step: int,
    out_path: Path,
    condition_rtg: bool,
    eval_num_trials: int = 1,
    num_rollouts: int = 1,
    eval_context_mode: str = "prompt",
) -> List[Path]:
    """Write reward/RTG PNGs under ``out_path``; return all paths (may include a ``_mean_std`` variant)."""
    out_paths: List[Path] = []
    if not reward_rows or all(np.asarray(r).size == 0 for r in reward_rows):
        return out_paths
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _serif = ["Palatino", "Palatino Linotype", "DejaVu Serif"]
    bg = "#FAFAF8"
    c_r = "#2E86AB"
    c_rtg = "#8B3A62"

    K = max(1, int(eval_num_trials))
    N = max(1, int(num_rollouts))
    zero_shot_multi = eval_context_mode == "zero_shot_adaptation" and K > 1
    if zero_shot_multi:
        n_in = len(reward_rows)
        if n_in != N * K:
            print(
                f"[eval] rtg_reward: zero-shot expected {N * K} traces (sessions × trials), got {n_in}; "
                "using aggregate plot.",
                flush=True,
            )
            zero_shot_multi = False

    if zero_shot_multi:
        with plt.rc_context(
            {
                "font.family": "serif",
                "font.serif": _serif,
                "axes.facecolor": "#FFFFFF",
                "figure.facecolor": bg,
                "axes.edgecolor": "#333333",
                "axes.labelcolor": "#222222",
                "xtick.color": "#222222",
                "ytick.color": "#222222",
                "grid.alpha": 0.35,
                "grid.linestyle": "-",
            }
        ):
            out_path.parent.mkdir(parents=True, exist_ok=True)

            def _stitch_session(s: int) -> Tuple[np.ndarray, np.ndarray]:
                r_parts: List[np.ndarray] = []
                g_parts: List[np.ndarray] = []
                for t_k in range(K):
                    idx = s * K + t_k
                    r_parts.append(np.asarray(reward_rows[idx], dtype=np.float64).reshape(-1))
                    if rtg_rows and idx < len(rtg_rows):
                        g_parts.append(np.asarray(rtg_rows[idx], dtype=np.float64).reshape(-1))
                r_cat = np.concatenate(r_parts) if r_parts else np.zeros(0, dtype=np.float64)
                g_cat = np.concatenate(g_parts) if g_parts else np.zeros(0, dtype=np.float64)
                return r_cat, g_cat

            use_rtg = bool(
                condition_rtg
                and rtg_rows
                and not all(
                    np.asarray(rtg_rows[s * K + t_k]).size == 0
                    for s in range(N)
                    for t_k in range(K)
                )
            )

            # Trial boundaries on the stitched axis (from session 0 lengths).
            trial_bounds: List[float] = []
            cum = 0
            for t_k in range(K - 1):
                cum += float(len(np.asarray(reward_rows[t_k], dtype=np.float64).reshape(-1)))
                trial_bounds.append(cum)

            if use_rtg:
                fig, axes = plt.subplots(
                    N,
                    2,
                    figsize=(10.5, max(4.0, 1.85 * N)),
                    squeeze=False,
                    sharex="col",
                    sharey="col",
                    constrained_layout=True,
                )
            else:
                fig, axes = plt.subplots(
                    N,
                    1,
                    figsize=(9.0, max(3.5, 1.6 * N)),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                    constrained_layout=True,
                )

            for s in range(N):
                r_st, g_st = _stitch_session(s)
                if r_st.size == 0:
                    continue
                xr = np.arange(r_st.size, dtype=np.float64)
                if use_rtg:
                    ax_rw = axes[s, 0]
                    ax_gt = axes[s, 1]
                else:
                    ax_rw = axes[s, 0]
                ax_rw.plot(xr, r_st, color=c_r, linewidth=1.5)
                for xb in trial_bounds:
                    ax_rw.axvline(xb, color="#bbbbbb", linestyle="--", linewidth=0.9, alpha=0.9)
                ax_rw.set_ylabel(f"S{s + 1}")
                ax_rw.grid(True, axis="y", alpha=0.4)
                if use_rtg and g_st.size > 0:
                    xg = np.arange(g_st.size, dtype=np.float64)
                    ax_gt.plot(xg, g_st, color=c_rtg, linewidth=1.5)
                    for xb in trial_bounds:
                        ax_gt.axvline(xb, color="#bbbbbb", linestyle="--", linewidth=0.9, alpha=0.9)
                    ax_gt.grid(True, axis="y", alpha=0.4)

            if use_rtg:
                axes[0, 0].set_title("Reward / step")
                axes[0, 1].set_title("RTG token (before action)")
                axes[-1, 0].set_xlabel("env steps")
                axes[-1, 1].set_xlabel("env steps")
            else:
                axes[0, 0].set_title("Reward / step")
                axes[-1, 0].set_xlabel("env steps")

            fig.suptitle(
                f"Eval step {step} · {K} trials × {N} session(s) · rtg_scale={rtg_scale:g}",
                fontsize=11,
                y=1.02,
                color="#1a1a1a",
            )
            fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=bg)
            plt.close(fig)
            out_paths.append(out_path)

            # Second figure: one row, mean ± std across sessions (stitched env-step axis).
            mean_path = out_path.with_name(out_path.stem + "_mean_std" + out_path.suffix)
            r_stitched_list = [_stitch_session(s)[0] for s in range(N)]
            R_pad = pad_ragged_1d(r_stitched_list)
            T_r = int(R_pad.shape[1])
            if T_r > 0:
                r_mean = np.nanmean(R_pad, axis=0)
                r_std = np.nanstd(R_pad, axis=0, ddof=0)
                r_std = np.nan_to_num(r_std, nan=0.0, posinf=0.0, neginf=0.0)
                x_r = np.arange(T_r, dtype=np.float64)

                agg_two_col = False
                T_plot = T_r
                if use_rtg:
                    g_stitched_list = [_stitch_session(s)[1] for s in range(N)]
                    G_pad = pad_ragged_1d(g_stitched_list)
                    T_g = int(G_pad.shape[1])
                    T_plot = min(T_r, T_g)
                    agg_two_col = T_plot >= 1

                if agg_two_col:
                    x_r = x_r[:T_plot]
                    r_mean = r_mean[:T_plot]
                    r_std = r_std[:T_plot]
                    g_mean = np.nanmean(G_pad[:, :T_plot], axis=0)
                    g_std = np.nanstd(G_pad[:, :T_plot], axis=0, ddof=0)
                    g_std = np.nan_to_num(g_std, nan=0.0, posinf=0.0, neginf=0.0)
                    x_g = np.arange(T_plot, dtype=np.float64)

                    fig_m, (ax_mr, ax_mg) = plt.subplots(
                        1,
                        2,
                        figsize=(10.5, 4.0),
                        sharex=True,
                        constrained_layout=True,
                    )
                    ax_mr.fill_between(
                        x_r, r_mean - r_std, r_mean + r_std, color=c_r, alpha=0.28, linewidth=0
                    )
                    ax_mr.plot(x_r, r_mean, color=c_r, linewidth=2.0, label="mean ± std")
                    for xb in trial_bounds:
                        if xb < T_plot:
                            ax_mr.axvline(
                                xb, color="#bbbbbb", linestyle="--", linewidth=0.9, alpha=0.9
                            )
                    ax_mr.set_title("Reward / step (mean ± std across sessions)")
                    ax_mr.set_ylabel("env reward")
                    ax_mr.set_xlabel("env steps")
                    ax_mr.grid(True, axis="y", alpha=0.4)
                    ax_mr.legend(loc="upper right", fontsize=8, framealpha=0.92)

                    ax_mg.fill_between(
                        x_g, g_mean - g_std, g_mean + g_std, color=c_rtg, alpha=0.28, linewidth=0
                    )
                    ax_mg.plot(x_g, g_mean, color=c_rtg, linewidth=2.0, label="mean ± std")
                    for xb in trial_bounds:
                        if xb < T_plot:
                            ax_mg.axvline(
                                xb, color="#bbbbbb", linestyle="--", linewidth=0.9, alpha=0.9
                            )
                    ax_mg.set_title("RTG token (before action), mean ± std across sessions")
                    ax_mg.set_ylabel("RTG token")
                    ax_mg.set_xlabel("env steps")
                    ax_mg.grid(True, axis="y", alpha=0.4)
                    ax_mg.legend(loc="upper right", fontsize=8, framealpha=0.92)
                    fig_m.suptitle(
                        f"Eval step {step} · mean ± std over N={N} session(s) · {K} trials stitched · "
                        f"rtg_scale={rtg_scale:g}",
                        fontsize=11,
                        y=1.03,
                        color="#1a1a1a",
                    )
                else:
                    fig_m, ax_mr = plt.subplots(1, 1, figsize=(9.0, 3.6), constrained_layout=True)
                    ax_mr.fill_between(
                        x_r, r_mean - r_std, r_mean + r_std, color=c_r, alpha=0.28, linewidth=0
                    )
                    ax_mr.plot(x_r, r_mean, color=c_r, linewidth=2.0, label="mean ± std")
                    for xb in trial_bounds:
                        if xb < T_r:
                            ax_mr.axvline(
                                xb, color="#bbbbbb", linestyle="--", linewidth=0.9, alpha=0.9
                            )
                    ax_mr.set_title("Reward / step (mean ± std across sessions)")
                    ax_mr.set_ylabel("env reward")
                    ax_mr.set_xlabel("env steps")
                    ax_mr.grid(True, axis="y", alpha=0.4)
                    ax_mr.legend(loc="upper right", fontsize=8, framealpha=0.92)
                    fig_m.suptitle(
                        f"Eval step {step} · mean ± std over N={N} session(s) · {K} trials stitched",
                        fontsize=11,
                        y=1.03,
                        color="#1a1a1a",
                    )

                fig_m.savefig(str(mean_path), dpi=120, bbox_inches="tight", facecolor=bg)
                plt.close(fig_m)
                out_paths.append(mean_path)
        return out_paths

    R = pad_ragged_1d(reward_rows)
    n_roll = int(R.shape[0])
    T = int(R.shape[1])
    x = np.arange(T, dtype=int)

    def _plot_reward(ax_r: Any, subtitle: str) -> None:
        if n_roll <= 1:
            ax_r.plot(x, R[0], color=c_r, linewidth=2.0, label="rollout")
            ax_r.set_title(subtitle)
        else:
            r_mean = np.nanmean(R, axis=0)
            r_std = np.nanstd(R, axis=0)
            r_std = np.nan_to_num(r_std, nan=0.0, posinf=0.0, neginf=0.0)
            ax_r.fill_between(x, r_mean - r_std, r_mean + r_std, color=c_r, alpha=0.28, linewidth=0)
            ax_r.plot(x, r_mean, color=c_r, linewidth=2.0, label="mean ± std")
            ax_r.set_title(subtitle)
        ax_r.set_ylabel("env reward")
        ax_r.grid(True, axis="y", alpha=0.4)
        ax_r.legend(loc="upper right", fontsize=8, framealpha=0.92)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": _serif,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": bg,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
        }
    ):
        if condition_rtg and rtg_rows and not all(np.asarray(t).size == 0 for t in rtg_rows):
            G = pad_ragged_1d(rtg_rows)
            share_x = int(G.shape[1]) == int(R.shape[1])
            fig, (ax_r, ax_g) = plt.subplots(
                2,
                1,
                figsize=(9, 5.4),
                sharex=share_x,
                constrained_layout=True,
                gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
            )
            r_sub = (
                "Reward per env step (single eval episode)"
                if n_roll <= 1
                else f"Reward per env step (mean ± std over {n_roll} eval episodes)"
            )
            _plot_reward(ax_r, r_sub)

            n_rtg = int(G.shape[0])
            if n_rtg <= 1:
                ax_g.plot(np.arange(G.shape[1]), G[0], color=c_rtg, linewidth=2.0, label="rollout")
                ax_g.set_title("RTG token (tail before each action)")
            else:
                g_mean = np.nanmean(G, axis=0)
                g_std = np.nanstd(G, axis=0)
                g_std = np.nan_to_num(g_std, nan=0.0, posinf=0.0, neginf=0.0)
                xg = np.arange(G.shape[1], dtype=int)
                ax_g.fill_between(
                    xg, g_mean - g_std, g_mean + g_std, color=c_rtg, alpha=0.28, linewidth=0
                )
                ax_g.plot(xg, g_mean, color=c_rtg, linewidth=2.0, label="mean ± std")
                ax_g.set_title(
                    f"RTG token (mean ± std, n={n_rtg}); update −r / rtg_scale (rtg_scale={rtg_scale:g})"
                )
            ax_g.set_ylabel("RTG token")
            ax_g.set_xlabel("environment step")
            ax_g.grid(True, axis="y", alpha=0.4)
            ax_g.legend(loc="upper right", fontsize=8, framealpha=0.92)
        else:
            fig, ax_r = plt.subplots(1, 1, figsize=(9, 3.4), constrained_layout=True)
            r_sub = (
                "Reward per env step (single eval episode); RTG conditioning off"
                if n_roll <= 1
                else f"Reward per env step (mean ± std over {n_roll} eval episodes); RTG conditioning off"
            )
            _plot_reward(ax_r, r_sub)
            ax_r.set_xlabel("environment step")

        fig.suptitle(f"Eval step {step} · rollout dynamics", fontsize=12, y=1.02, color="#1a1a1a")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=bg)
        plt.close(fig)
    out_paths.append(out_path)
    return out_paths
