"""Multiplex Interactive Temporal Module (MXT), Section 3.4."""

import torch
from torch import nn

from .layers import (
    FeedForward, SingletonSafeBatchNorm1d, TransformerBlock, temporal_embedding,
)


class AdaptiveTCN(nn.Module):
    """Softmax-weighted 3/5/7 temporal convolutions with residual, Eqs. (17–18)."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim),
                SingletonSafeBatchNorm1d(dim),
                nn.GELU(),
                nn.Conv1d(dim, dim, 1),
                nn.Dropout(dropout),
            )
            for kernel in (3, 5, 7)
        ])
        self.fusion_logits = nn.Parameter(torch.zeros(3))
        self.fusion_norm = nn.LayerNorm(dim)

    def forward(self, x):
        channels_first = x.transpose(1, 2)
        weights = self.fusion_logits.softmax(dim=0)
        mixed = sum(
            weight * branch(channels_first)
            for weight, branch in zip(weights, self.branches)
        ).transpose(1, 2)
        return x + self.fusion_norm(mixed)


class TemporalEncodingBranch(nn.Module):
    """Always-on TEB: learned temporal position and Transformer, Eqs. (14–16)."""

    def __init__(self, dim, frames=27, depth=4, num_heads=8,
                 mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.temporal_position = nn.Parameter(torch.zeros(1, frames, dim))
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(x + temporal_embedding(self.temporal_position, x.shape[1]))


class KinematicRefinementBranch(nn.Module):
    """KRB: Adaptive TCN → motion-aware attention/FFN → Adaptive TCN."""

    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.pre_tcn = AdaptiveTCN(dim, dropout)
        self.velocity_encoder = FeedForward(dim, dim // 2, dropout)
        self.acceleration_encoder = FeedForward(dim, dim // 2, dropout)
        self.motion_norm = nn.LayerNorm(dim)
        self.motion_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout)
        self.post_tcn = AdaptiveTCN(dim, dropout)

    @staticmethod
    def finite_difference(x):
        """Backward difference with the first valid difference repeated at t=0.

        A single-frame sequence has no measurable motion and returns zeros.
        """
        if x.shape[1] == 1:
            return torch.zeros_like(x)
        difference = x[:, 1:] - x[:, :-1]
        return torch.cat((difference[:, :1], difference), dim=1)

    def forward(self, x):
        pre = self.pre_tcn(x)
        velocity = self.finite_difference(pre)
        acceleration = self.finite_difference(velocity)
        motion = self.velocity_encoder(velocity) + self.acceleration_encoder(acceleration)
        normalized = self.motion_norm(pre + motion)
        refined = pre + self.motion_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        refined = refined + self.ffn(self.ffn_norm(refined))
        return self.post_tcn(refined)


class MultiplexInteractiveTemporalModule(nn.Module):
    """MXT activates kinematic refinement only in PMSS stages 2 and 3."""

    def __init__(self, embed_dim=32, frames=27, depth=4, num_heads=8,
                 mlp_ratio=2.0, dropout=0.0, kinematic_dropout=0.1):
        super().__init__()
        dim = 17 * embed_dim
        self.teb = TemporalEncodingBranch(
            dim, frames, depth, num_heads, mlp_ratio, dropout
        )
        self.krb = KinematicRefinementBranch(dim, num_heads, mlp_ratio, kinematic_dropout)

    def forward(self, x, stage=1):
        if stage not in (1, 2, 3):
            raise ValueError("PMSS stage must be 1, 2, or 3")
        temporal = self.teb(x)
        return temporal if stage == 1 else self.krb(temporal)
