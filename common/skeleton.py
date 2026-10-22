"""The 17-joint Human3.6M skeleton and its graph neighborhoods."""

import torch


H36M_JOINTS = (
    "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot",
    "Spine", "Thorax", "Neck", "Head", "LShoulder", "LElbow",
    "LWrist", "RShoulder", "RElbow", "RWrist",
)
H36M_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
H36M_EDGES = tuple((parent, joint) for joint, parent in enumerate(H36M_PARENTS) if parent >= 0)
H36M_KEPT_JOINTS = (0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27)
JOINTS_LEFT = (4, 5, 6, 11, 12, 13)
JOINTS_RIGHT = (1, 2, 3, 14, 15, 16)


def build_hop_matrices(max_hop=4):
    """Return row-normalized exact k-hop adjacency matrices, k=1,...,max_hop.

    Each matrix uses its own node degrees. Rows without a neighbor at the
    requested distance remain zero; the diagonal is excluded.
    """
    if not isinstance(max_hop, int) or max_hop < 1:
        raise ValueError("max_hop must be a positive integer")
    num_joints = len(H36M_PARENTS)
    distance = torch.full((num_joints, num_joints), float("inf"))
    distance.fill_diagonal_(0)
    for parent, joint in H36M_EDGES:
        distance[parent, joint] = distance[joint, parent] = 1
    for joint in range(num_joints):
        distance = torch.minimum(distance, distance[:, joint:joint + 1] + distance[joint:joint + 1, :])
    hops = torch.stack([(distance == hop).float() for hop in range(1, max_hop + 1)])
    return hops / hops.sum(dim=-1, keepdim=True).clamp_min(1)
