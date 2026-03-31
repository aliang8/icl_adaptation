"""
Modular vision encoders for VLA-DT.
- patch: trainable per-patch (MultiViewVisionEncoder), pooled to 1 vector per view then concat.
- crossmae: ViT per-patch style; use attention pooling (ICRT-style) or mean-pool to (B, T, D).
- dinov2 / dinov3: 1 embedding per image (CLS) or patch tokens + attention pooling.
- paligemma: SigLIP vision tower (PaliGemma-style), pooled.

Attention pooling (ICRT-style, https://github.com/Max-Fu/icrt, https://arxiv.org/abs/2408.15980):
patch tokens are compressed by a learned query (or optional proprio query) attending over patches,
producing one state token per view per timestep. Set vision_encoder_attention_pool=true to use it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.vision import MultiViewVisionEncoder


class AttentionPooling(nn.Module):
    """
    ICRT-style attention pooling: compress patch tokens (B, T, N, D_patch) to (B, T, D_out)
    by using a learned query (or optional external query) to attend over patches.
    Ref: ICRT (Fu et al.), Set Transformer (Lee et al.).
    """

    def __init__(
        self,
        patch_dim: int,
        output_dim: int,
        query_dim: Optional[int] = None,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.output_dim = output_dim
        self.proj_k = nn.Linear(patch_dim, output_dim)
        self.proj_v = nn.Linear(patch_dim, output_dim)
        if query_dim is not None:
            self.proj_q = nn.Linear(query_dim, output_dim)
            self.learned_query = None
        else:
            self.proj_q = None
            self.learned_query = nn.Parameter(torch.zeros(1, 1, output_dim))
            nn.init.normal_(self.learned_query, std=0.02)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        patch_tokens: (B, T, N, D_patch) or (B, N, D_patch).
        query: optional (B, T, query_dim) or (B, query_dim). If None, use learned query.
        Returns: (B, T, output_dim) or (B, output_dim).
        """
        if patch_tokens.dim() == 3:
            patch_tokens = patch_tokens.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        B, T, N, D = patch_tokens.shape
        k = self.proj_k(patch_tokens)
        v = self.proj_v(patch_tokens)
        if query is not None:
            if query.dim() == 2:
                query = query.unsqueeze(1)
            q = self.proj_q(query)
        else:
            q = self.learned_query.expand(B, T, -1)
        scale = self.output_dim ** (-0.5)
        attn = torch.softmax((q.unsqueeze(2) @ k.transpose(-2, -1)) * scale, dim=-1)
        out = (attn @ v).squeeze(2)
        if squeeze:
            out = out.squeeze(1)
        return out


class VisionProprioAttentionFusion(nn.Module):
    """
    ICRT-style: treat view embeddings + proprio as tokens, attention-pool to one vector.
    If num_views=2: tokens = [view0, view1, proprio]. If num_views=1: tokens = [view0, proprio].
    Uses learned query to attend over tokens; output projected to hidden_size.
    """

    def __init__(
        self,
        vision_embed_dim: int,
        state_dim: int,
        num_views: int,
        hidden_size: int,
    ):
        super().__init__()
        self.num_views = num_views
        if vision_embed_dim % num_views != 0:
            raise ValueError(
                f"vision_embed_dim ({vision_embed_dim}) must be divisible by num_views ({num_views})."
            )
        self.d_per_view = vision_embed_dim // num_views
        self.state_proj = nn.Linear(state_dim, self.d_per_view)
        self.pool = AttentionPooling(
            patch_dim=self.d_per_view,
            output_dim=self.d_per_view,
            query_dim=None,
        )
        self.out_proj = nn.Linear(self.d_per_view, hidden_size)

    def forward(
        self,
        vision_emb: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        vision_emb: (B, T, vision_embed_dim) = [view0, view1, ...] concatenated.
        state: (B, T, state_dim) = proprio.
        Returns: (B, T, hidden_size).
        """
        B, T, D = vision_emb.shape
        num_views = self.num_views
        d = self.d_per_view
        if D != num_views * d:
            raise ValueError(
                f"VisionProprioAttentionFusion expected vision_emb last dim "
                f"{num_views * d} (num_views={num_views}, d_per_view={d}), got D={D}."
            )
        views = vision_emb.reshape(B, T, num_views, d)
        proprio = self.state_proj(state).unsqueeze(2)
        tokens = torch.cat([views, proprio], dim=2)
        pooled = self.pool(tokens)
        return self.out_proj(pooled)


def _ensure_bt(images: List[torch.Tensor]) -> Tuple[int, int]:
    """Infer B, T from list of tensors (each (B,T,C,H,W) or (B,C,H,W))."""
    im = images[0]
    if im.dim() == 5:
        return im.shape[0], im.shape[1]
    return im.shape[0], 1


class PooledVisionEncoderWrapper(nn.Module):
    """Wraps (B, T, N, D) encoder to (B, T, D) by mean-pooling over N."""

    def __init__(self, encoder: nn.Module, output_dim: int):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        x = self.encoder(images)
        if x.dim() == 4:
            return x.mean(dim=2)
        return x


def _build_patch_encoder(
    num_views: int = 2,
    embed_dim: int = 256,
    img_size: Tuple[int, int] = (224, 224),
    pool: bool = True,
    attention_pool: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Per-patch encoder. If pool: concat per-view (attention-pool or mean)."""
    enc = MultiViewVisionEncoder(
        num_views=num_views,
        embed_dim=embed_dim,
        img_size=img_size,
        **kwargs,
    )
    num_tokens_per_view = enc.encoders[0].num_output_tokens
    output_dim = num_views * embed_dim
    if pool and attention_pool:

        class _PatchAttentionPooled(nn.Module):
            def __init__(self, enc: nn.Module, n_views: int, n_tok: int, dim: int):
                super().__init__()
                self.encoder = enc
                self.num_views = n_views
                self.num_tokens_per_view = n_tok
                self.pool = nn.ModuleList([AttentionPooling(dim, dim) for _ in range(n_views)])
                self.output_dim = n_views * dim

            def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
                x = self.encoder(images)
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                B, T, N, D = x.shape
                out_list = []
                for v in range(self.num_views):
                    start = v * self.num_tokens_per_view
                    end = start + self.num_tokens_per_view
                    tok = x[:, :, start:end, :]
                    out_list.append(self.pool[v](tok))
                return torch.cat(out_list, dim=-1)

        return _PatchAttentionPooled(enc, num_views, num_tokens_per_view, embed_dim)
    if pool:

        class _PatchPooled(nn.Module):
            def __init__(self, enc: nn.Module, dim: int):
                super().__init__()
                self.encoder = enc
                self.output_dim = dim

            def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
                return self.encoder.forward_pooled(images)

        return _PatchPooled(enc, output_dim)
    return PooledVisionEncoderWrapper(enc, enc.num_tokens_total * embed_dim)


def _build_dinov2_encoder(
    num_views: int = 2,
    pool: bool = True,
    model_name: str = "facebook/dinov2-base",
    attention_pool: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """DINOv2/DINOv3: CLS token (default) or all patch tokens + attention pooling. Same API for both."""
    from transformers import AutoModel

    chunk_size = kwargs.pop("chunk_size", None)
    backbone = AutoModel.from_pretrained(model_name, **kwargs)
    hidden = backbone.config.hidden_size

    class DINOv2Encoder(nn.Module):
        def __init__(
            self,
            backbone: nn.Module,
            n_views: int,
            hidden_size: int,
            use_attention_pool: bool,
            chunk_size: Optional[int] = None,
        ):
            super().__init__()
            self.backbone = backbone
            self.num_views = n_views
            self.use_attention_pool = use_attention_pool
            self.chunk_size = chunk_size
            self.output_dim = n_views * hidden_size
            if use_attention_pool:
                self.pool = nn.ModuleList(
                    [AttentionPooling(hidden_size, hidden_size) for _ in range(n_views)]
                )

        def _backbone_forward(self, im: torch.Tensor) -> torch.Tensor:
            """Run backbone on im (N, C, H, W), optionally in chunks."""
            N = im.shape[0]
            if self.chunk_size is None or N <= self.chunk_size:
                return self.backbone(pixel_values=im).last_hidden_state
            chunks = []
            for start in tqdm(
                range(0, N, self.chunk_size),
                desc="Computing image embeddings",
                total=(N + self.chunk_size - 1) // self.chunk_size,
                unit="chunk",
                leave=False,
            ):
                end = min(start + self.chunk_size, N)
                chunk = im[start:end]
                h = self.backbone(pixel_values=chunk).last_hidden_state
                chunks.append(h)
            return torch.cat(chunks, dim=0)

        def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
            B, T = _ensure_bt(images)
            out_list = []
            for v in range(min(self.num_views, len(images))):
                im = images[v]
                if im.dim() == 5:
                    im = im.reshape(B * T, *im.shape[2:])
                h = self._backbone_forward(im)
                if self.use_attention_pool:
                    if T > 1:
                        h = h.reshape(B, T, h.shape[1], h.shape[2])
                    else:
                        h = h.unsqueeze(1)
                    out_list.append(self.pool[v](h))
                else:
                    cls_token = h[:, 0]
                    if T > 1:
                        cls_token = cls_token.reshape(B, T, -1)
                    else:
                        cls_token = cls_token.unsqueeze(1)
                    out_list.append(cls_token)
            while len(out_list) < self.num_views:
                out_list.append(out_list[-1])
            return torch.cat(out_list, dim=-1)

    return DINOv2Encoder(backbone, num_views, hidden, attention_pool, chunk_size)


def _build_paligemma_encoder(
    num_views: int = 2,
    pool: bool = True,
    model_name: str = "google/siglip-base-patch16-224",
    **kwargs: Any,
) -> nn.Module:
    """PaliGemma-style: SigLIP vision tower. Pooled = pooler or mean of patches."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name, **kwargs)
    hidden = model.config.hidden_size

    class SigLIPEncoder(nn.Module):
        def __init__(self, model: nn.Module, n_views: int, hidden_size: int, do_pool: bool):
            super().__init__()
            self.model = model
            self.num_views = n_views
            self.do_pool = do_pool
            self.output_dim = n_views * hidden_size

        def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
            B, T = _ensure_bt(images)
            out_list = []
            for v in range(min(self.num_views, len(images))):
                im = images[v]
                if im.dim() == 5:
                    im = im.reshape(B * T, *im.shape[2:])
                h = self.model(pixel_values=im)
                try:
                    po = h.pooler_output
                except AttributeError:
                    po = None
                if self.do_pool and po is not None:
                    tok = po
                else:
                    tok = h.last_hidden_state.mean(dim=1)
                if T > 1:
                    tok = tok.reshape(B, T, -1)
                else:
                    tok = tok.unsqueeze(1)
                out_list.append(tok)
            while len(out_list) < self.num_views:
                out_list.append(out_list[-1])
            return torch.cat(out_list, dim=-1)

    return SigLIPEncoder(model, num_views, hidden, pool)


def _build_crossmae_encoder(
    num_views: int = 2,
    embed_dim: int = 768,
    img_size: Tuple[int, int] = (224, 224),
    pool: bool = True,
    pretrained: Optional[str] = "google/vit-base-patch16-224",
    attention_pool: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """CrossMAE-style: ViT per-patch, then attention-pool (ICRT-style) or mean-pool."""
    from transformers import AutoModel

    vit = AutoModel.from_pretrained(pretrained or "google/vit-base-patch16-224", **kwargs)
    hidden = vit.config.hidden_size

    class ViTPatchEncoder(nn.Module):
        def __init__(
            self,
            vit: nn.Module,
            n_views: int,
            hidden_size: int,
            use_attention_pool: bool,
        ):
            super().__init__()
            self.vit = vit
            self.num_views = n_views
            self.use_attention_pool = use_attention_pool
            self.output_dim = n_views * hidden_size
            if use_attention_pool:
                self.pool = nn.ModuleList(
                    [AttentionPooling(hidden_size, hidden_size) for _ in range(n_views)]
                )

        def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
            B, T = _ensure_bt(images)
            out_list = []
            for v in range(min(self.num_views, len(images))):
                im = images[v]
                if im.dim() == 5:
                    im = im.reshape(B * T, *im.shape[2:])
                h = self.vit(pixel_values=im).last_hidden_state
                if self.use_attention_pool:
                    if T > 1:
                        h = h.reshape(B, T, h.shape[1], h.shape[2])
                    else:
                        h = h.unsqueeze(1)
                    tok = self.pool[v](h)
                else:
                    tok = h.mean(dim=1)
                    if T > 1:
                        tok = tok.reshape(B, T, -1)
                    else:
                        tok = tok.unsqueeze(1)
                out_list.append(tok)
            while len(out_list) < self.num_views:
                out_list.append(out_list[-1])
            return torch.cat(out_list, dim=-1)

    return ViTPatchEncoder(vit, num_views, hidden, attention_pool)


def build_vision_encoder(
    encoder_type: str = "patch",
    num_views: int = 2,
    embed_dim: int = 256,
    img_size: Tuple[int, int] = (224, 224),
    pool: bool = True,
    attention_pool: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory: returns encoder with .output_dim and .forward(images) -> (B, T, output_dim).
    attention_pool: use ICRT-style attention pooling over patch tokens (learned query) instead of mean.
    chunk_size: only used by dinov2/dinov3; popped for other encoders.
    """
    chunk_size = kwargs.pop("chunk_size", None)
    if encoder_type == "patch":
        return _build_patch_encoder(
            num_views=num_views,
            embed_dim=embed_dim,
            img_size=img_size,
            pool=pool,
            attention_pool=attention_pool,
            **kwargs,
        )
    if encoder_type == "dinov2":
        model_name = kwargs.pop("model_name", "facebook/dinov2-base")
        return _build_dinov2_encoder(
            num_views=num_views,
            pool=pool,
            model_name=model_name,
            attention_pool=attention_pool,
            chunk_size=chunk_size,
            **kwargs,
        )
    if encoder_type == "dinov3":
        model_name = kwargs.pop("model_name", "facebook/dinov3-vits16-pretrain-lvd1689m")
        return _build_dinov2_encoder(
            num_views=num_views,
            pool=pool,
            model_name=model_name,
            attention_pool=attention_pool,
            chunk_size=chunk_size,
            **kwargs,
        )
    if encoder_type == "paligemma":
        model_name = kwargs.pop("model_name", "google/siglip-base-patch16-224")
        return _build_paligemma_encoder(
            num_views=num_views,
            pool=pool,
            model_name=model_name,
            **kwargs,
        )
    if encoder_type == "crossmae":
        return _build_crossmae_encoder(
            num_views=num_views,
            embed_dim=embed_dim,
            img_size=img_size,
            pool=pool,
            attention_pool=attention_pool,
            **kwargs,
        )
    raise ValueError(
        f"Unknown vision_encoder_type: {encoder_type}. "
        "Use 'patch', 'crossmae', 'dinov2', 'dinov3', or 'paligemma'."
    )
