"""Small shared building blocks for MXPose."""

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )


class TransformerBlock(nn.Module):
    """Pre-normalized attention and FFN with separate residuals, Eqs. (15–16)."""

    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        normalized = self.attention_norm(x)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        x = x + attended
        return x + self.ffn(self.ffn_norm(x))


def temporal_embedding(position, frames):
    """Resize learned positions along time when evaluating a different window.

    Input shape is (..., T, C); the feature and optional view axes are preserved.
    """
    if position.shape[-2] == frames:
        return position
    leading_shape = position.shape[:-2]
    channels = position.shape[-1]
    flattened = position.reshape(-1, position.shape[-2], channels)
    resized = F.interpolate(
        flattened.transpose(1, 2), size=frames, mode="linear", align_corners=False
    ).transpose(1, 2)
    return resized.reshape(*leading_shape, frames, channels)


class SingletonSafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics for the degenerate B=T=1 training case."""

    def forward(self, x):
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, momentum=self.momentum, eps=self.eps,
            )
        return super().forward(x)
