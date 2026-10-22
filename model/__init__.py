"""MXPose model and its three main modules."""

from .mxpose import MXPose
from .spatial import MultiplexInteractiveSpatialEncoder
from .fusion import CrossViewFusion
from .temporal import MultiplexInteractiveTemporalModule

__all__ = [
    "MXPose", "MultiplexInteractiveSpatialEncoder", "CrossViewFusion",
    "MultiplexInteractiveTemporalModule",
]
