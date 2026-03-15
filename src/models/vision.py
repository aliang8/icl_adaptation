"""
Multi-view vision encoder for ICRT-style architectures.

Encodes multiple camera views (e.g. exterior + wrist) into a single embedding
or per-step features. Supports optional camera/modality positional embeddings.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Embed image patches with a small CNN (alternative to ViT for smaller scale)."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        patch_size: int = 16,
        img_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SingleViewEncoder(nn.Module):
    """Encode one image view to a fixed-size embedding (CNN backbone + optional pooling)."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        use_cls: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls = use_cls
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
        )
        num_patches = self.patch_embed.num_patches
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)
            self.num_output_tokens = num_patches + 1
        else:
            self.cls_token = None
            self.num_output_tokens = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_tokens, embed_dim)
        B = x.shape[0]
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls, x), dim=1)
        return x


class MultiViewVisionEncoder(nn.Module):
    """
    Encode multiple image views (e.g. exterior + wrist) and fuse to a single embedding.

    ICRT-style: separate adapter per camera, optional camera/modality positional embeddings.
    Output can be used as additional context tokens or fused with state.
    """

    def __init__(
        self,
        num_views: int = 2,
        in_channels: int = 3,
        embed_dim: int = 256,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        use_cls_per_view: bool = True,
        camera_pos_emb: bool = True,
        modality_pos_emb: bool = True,
        separate_encoders: bool = True,
    ):
        super().__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.camera_pos_emb = camera_pos_emb
        self.modality_pos_emb = modality_pos_emb

        if separate_encoders:
            self.encoders = nn.ModuleList(
                [
                    SingleViewEncoder(
                        in_channels=in_channels,
                        embed_dim=embed_dim,
                        img_size=img_size,
                        patch_size=patch_size,
                        use_cls=use_cls_per_view,
                    )
                    for _ in range(num_views)
                ]
            )
        else:
            shared = SingleViewEncoder(
                in_channels=in_channels,
                embed_dim=embed_dim,
                img_size=img_size,
                patch_size=patch_size,
                use_cls=use_cls_per_view,
            )
            self.encoders = nn.ModuleList([shared for _ in range(num_views)])

        num_tokens_per_view = self.encoders[0].num_output_tokens
        self.num_tokens_total = num_views * num_tokens_per_view

        if camera_pos_emb:
            self.camera_embed = nn.Embedding(num_views, embed_dim)
        else:
            self.camera_embed = None

        if modality_pos_emb:
            self.modality_embed = nn.Embedding(num_views, embed_dim)
        else:
            self.modality_embed = None

    def forward(
        self,
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        images: list of (B, T, C, H, W) or (B, C, H, W) per view.
        Returns: (B, T, num_tokens_total, embed_dim) or (B, num_tokens_total, embed_dim)
        """
        if len(images) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(images)}")

        view_tokens = []
        for i, im in enumerate(images):
            # im: (B, T, C, H, W) or (B, C, H, W)
            if im.dim() == 5:
                B, T, C, H, W = im.shape
                im = im.reshape(B * T, C, H, W)
                out = self.encoders[i](im)
                out = out.reshape(B, T, -1, self.embed_dim)
            else:
                out = self.encoders[i](im)
                out = out.unsqueeze(1)
            view_tokens.append(out)

        # Stack: list of (B, T, N, D) -> (B, T, num_views*N, D)
        x = torch.cat(view_tokens, dim=2)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B, T, N, D = x.shape
        if self.camera_embed is not None or self.modality_embed is not None:
            # Add camera/modality embeddings per view (repeat for each view's tokens)
            num_tokens_per_view = self.encoders[0].num_output_tokens
            pos_list = []
            for i in range(self.num_views):
                e = torch.zeros(B, T, num_tokens_per_view, D, device=x.device, dtype=x.dtype)
                if self.camera_embed is not None:
                    e = e + self.camera_embed.weight[i]
                if self.modality_embed is not None:
                    e = e + self.modality_embed.weight[i]
                pos_list.append(e)
            pos = torch.cat(pos_list, dim=2)
            x = x + pos

        return x

    def forward_pooled(
        self,
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode and pool to one vector per view per step, then concat: (B, T, num_views * embed_dim).
        """
        x = self.forward(images)
        # x: (B, T, N, D) -> take first token (CLS) per view or mean pool
        num_tokens_per_view = self.encoders[0].num_output_tokens
        pooled = []
        for i in range(self.num_views):
            start = i * num_tokens_per_view
            end = start + num_tokens_per_view
            tok = x[:, :, start:end, :]
            tok = tok.mean(dim=2)
            pooled.append(tok)
        return torch.cat(pooled, dim=-1)
