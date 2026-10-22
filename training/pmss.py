"""PMSS objectives from Section 3.6, Equations (27)-(31)."""

import torch
from torch import nn

from common.metrics import mpjpe
from common.skeleton import H36M_EDGES


def stage_for_epoch(epoch, stage1_end=35, stage2_end=45):
    """Epochs are one-based: 1-35 / 36-45 / 46-50 by default."""
    if epoch < 1 or not 1 <= stage1_end < stage2_end:
        raise ValueError("Require epoch >= 1 and 1 <= stage1_end < stage2_end")
    return 1 if epoch <= stage1_end else 2 if epoch <= stage2_end else 3


class ViewwiseProjection(nn.Module):
    """Training-only, view-specific 3D-to-2D consistency mappings.

    The paper leaves the internal form of Pi_v unspecified. Each view here
    uses a small shared-across-joints 3 -> hidden -> 2 MLP. Both sides are
    root-centered, consistent with the root-relative 3D training targets.
    No camera calibration or ground-truth 3D enters the projection mapping.
    """

    def __init__(self, num_views=4, hidden=32):
        super().__init__()
        self.views = nn.ModuleList([
            nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 2, bias=False))
            for _ in range(num_views)
        ])

    def forward(self, predicted):
        centered = predicted - predicted[:, :, :1]
        projected = torch.stack([view(centered) for view in self.views], dim=2)
        return projected - projected[:, :, :, :1]


class ProgressiveMultiStageSupervision(nn.Module):
    """Stage 1: MPJPE; Stage 2: MPJPE + bone std; Stage 3: projection.

    KRB activation follows the same stage in MXPose.forward. The projector
    participates in optimization only during Stage 3 and is unnecessary for
    inference. Pose losses use meters; projection uses normalized 2D units.
    """

    def __init__(self, bone_weight=0.9, pose_weight=0.01):
        super().__init__()
        if bone_weight < 0 or not 0 <= pose_weight <= 1:
            raise ValueError("Require bone_weight >= 0 and 0 <= pose_weight <= 1")
        self.bone_weight = bone_weight
        self.pose_weight = pose_weight
        self.projection = ViewwiseProjection()
        self.register_buffer("bones", torch.tensor(H36M_EDGES, dtype=torch.long))

    def bone_length_variation(self, predicted):
        """Mean per-bone temporal standard deviation, Eq. (28)."""
        vectors = predicted[:, :, self.bones[:, 0]] - predicted[:, :, self.bones[:, 1]]
        lengths = torch.linalg.vector_norm(vectors, dim=-1)
        return lengths.std(dim=1, unbiased=False).mean()

    def forward(self, predicted, target, input_2d, stage):
        if stage not in (1, 2, 3):
            raise ValueError("PMSS stage must be 1, 2 or 3")
        pose = mpjpe(predicted, target)
        components = {"mpjpe": pose}
        if stage == 1:
            loss = pose
        elif stage == 2:
            bone = self.bone_length_variation(predicted)
            loss = pose + self.bone_weight * bone
            components["bone_std"] = bone
        else:
            observed = input_2d - input_2d[:, :, :, :1]
            # L1 norm over x,y, then mean over batch, time, views and joints.
            projection = (self.projection(predicted) - observed).abs().sum(dim=-1).mean()
            loss = self.pose_weight * pose + (1 - self.pose_weight) * projection
            components["projection"] = projection
        return loss, components
