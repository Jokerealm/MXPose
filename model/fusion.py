"""Cross-View Fusion (CVF), Section 3.3."""

import torch
from torch import nn

from .layers import temporal_embedding


class ViewConsensus(nn.Module):
    """7x7 convolution and view-sum residual followed by LN, Eq. (12)."""

    def __init__(self, num_views, feature_dim):
        super().__init__()
        self.view_norm = nn.BatchNorm2d(num_views)
        self.convolution = nn.Conv2d(num_views, 1, kernel_size=7, padding=3, bias=False)
        self.feature_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        consensus = self.convolution(self.view_norm(x)).squeeze(1)
        return self.feature_norm(consensus + x.sum(dim=1))


class CrossViewFusion(nn.Module):
    """Fuse spatial/hop features independently and apply structural gating.

    Inputs: (B, V, T, 17*C). Output: (B, T, 17*C).
    """

    def __init__(self, frames=27, embed_dim=32, num_views=4):
        super().__init__()
        feature_dim = 17 * embed_dim
        self.view_temporal_position = nn.Parameter(
            torch.zeros(1, num_views, frames, feature_dim)
        )
        self.spatial_fusion = ViewConsensus(num_views, feature_dim)
        self.hop_fusion = ViewConsensus(num_views, feature_dim)

    def forward(self, spatial, hop):
        position = temporal_embedding(self.view_temporal_position, spatial.shape[2])
        fused = self.spatial_fusion(spatial + position)
        geometric_gate = self.hop_fusion(hop + position)
        return fused * geometric_gate
