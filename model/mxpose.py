"""MXPose: multiplex interactive learning for multi-view 3D human pose estimation."""

import torch
from torch import nn

from .fusion import CrossViewFusion
from .spatial import MultiplexInteractiveSpatialEncoder
from .temporal import MultiplexInteractiveTemporalModule


class MXPose(nn.Module):
    """Human3.6M model with four synchronized views and 17 joints.

    Args:
        frames: Length used to initialize the learned temporal positions.
        embed_dim: Per-joint spatial width C (32 in the paper).
        depth: Number of semantic spatial and temporal Transformer blocks.

    Input: normalized 2D poses of shape (B, T, 4, 17, 2).
    Output: root-relative 3D poses of shape (B, T, 17, 3).
    No camera parameters are needed by the forward pass.
    """

    def __init__(self, frames=27, embed_dim=32, depth=4, num_heads=8,
                 k_neighbors=3, mlp_ratio=2.0, dropout=0.0,
                 kinematic_dropout=0.1):
        super().__init__()
        if frames < 1 or embed_dim < 1 or depth < 1 or num_heads < 1:
            raise ValueError("frames, embed_dim, depth, and num_heads must be positive")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.frames = frames
        self.embed_dim = embed_dim
        self.mxs = nn.ModuleList([
            MultiplexInteractiveSpatialEncoder(
                embed_dim, depth, num_heads, k_neighbors, mlp_ratio, dropout
            )
            for _ in range(4)
        ])
        self.cvf = CrossViewFusion(frames, embed_dim)
        self.mxt = MultiplexInteractiveTemporalModule(
            embed_dim, frames, depth, num_heads, mlp_ratio, dropout, kinematic_dropout
        )
        self.regression = nn.Linear(17 * embed_dim, 17 * 3)

    def forward(self, input_2d, stage=1):
        if input_2d.ndim != 5 or tuple(input_2d.shape[2:]) != (4, 17, 2):
            raise ValueError("input_2d must have shape (B, T, 4, 17, 2)")
        if input_2d.shape[1] < 1:
            raise ValueError("input_2d must contain at least one frame")
        if stage not in (1, 2, 3):
            raise ValueError("PMSS stage must be 1, 2, or 3")
        batch, frames = input_2d.shape[:2]
        spatial_views, hop_views = [], []
        for view, encoder in enumerate(self.mxs):
            coordinates = input_2d[:, :, view].reshape(batch * frames, 17, 2)
            spatial, hop = encoder(coordinates)
            spatial_views.append(spatial.reshape(batch, frames, -1))
            hop_views.append(hop.reshape(batch, frames, -1))
        consensus = self.cvf(
            torch.stack(spatial_views, dim=1), torch.stack(hop_views, dim=1)
        )
        refined = self.mxt(consensus, stage)
        return self.regression(refined).reshape(batch, frames, 17, 3)
