"""Multiplex Interactive Spatial Encoder (MXS), Section 3.2."""

import torch
from torch import nn
from torch.nn import functional as F

from common.skeleton import H36M_EDGES, build_hop_matrices
from .layers import FeedForward


class DynamicHypergraphConvolution(nn.Module):
    """Node–hyperedge–node propagation with feature-dependent incidence.

    Each joint centers one hyperedge containing itself and its nearest joints.
    Positive learned weights parameterize W_joint in Eq. (3). Vertex degrees
    include these weights, as in normalized spectral hypergraph convolution.
    """

    def __init__(self, dim, k_neighbors=3, skeletal_bias=0.5):
        super().__init__()
        if not 1 <= k_neighbors <= 17:
            raise ValueError("k_neighbors must be between 1 and 17")
        if not 0.0 <= skeletal_bias <= 1.0:
            raise ValueError("skeletal_bias must be between 0 and 1")
        self.k_neighbors = k_neighbors
        self.skeletal_bias = skeletal_bias
        adjacency = torch.zeros(17, 17)
        for parent, child in H36M_EDGES:
            adjacency[parent, child] = adjacency[child, parent] = 1
        self.register_buffer("skeletal_adjacency", adjacency)
        self.edge_logits = nn.Parameter(torch.full((17,), 0.54132485))
        self.projection = nn.Linear(dim, dim, bias=False)

    @torch.no_grad()
    def incidence(self, x):
        """Binary H[node, hyperedge] from squared biased distances, Eqs. (1–2)."""
        squared_norm = x.square().sum(dim=-1, keepdim=True)
        distance = (
            squared_norm + squared_norm.transpose(-1, -2)
            - 2 * torch.matmul(x, x.transpose(-1, -2))
        ).clamp_min(0)
        distance = distance * (1 - self.skeletal_bias * self.skeletal_adjacency)
        # Always retain the center, including when distinct nodes have equal features.
        distance.diagonal(dim1=-2, dim2=-1).fill_(-1)
        neighbors = distance.topk(self.k_neighbors, dim=-1, largest=False).indices
        edge_to_node = torch.zeros_like(distance).scatter_(-1, neighbors, 1)
        return edge_to_node.transpose(-1, -2)

    def forward(self, x):
        incidence = self.incidence(x)
        weights = F.softplus(self.edge_logits).to(x.dtype)
        vertex_degree = (incidence * weights).sum(dim=-1).clamp_min(1e-6)
        edge_degree = incidence.sum(dim=-2).clamp_min(1)
        vertex_scale = vertex_degree.rsqrt().unsqueeze(-1)
        nodes = self.projection(x) * vertex_scale
        edges = torch.matmul(incidence.transpose(-1, -2), nodes)
        edges = edges * (weights / edge_degree).unsqueeze(-1)
        nodes = torch.matmul(incidence, edges) * vertex_scale
        return F.gelu(nodes)


class MultiGranularityHypergraph(nn.Module):
    """Parallel joint, anatomical-part, and whole-body features, Eqs. (3–6)."""

    PARTS = (
        (9, 10),           # Head
        (0, 7, 8),        # Trunk
        (8, 11, 12, 13),  # Left arm
        (8, 14, 15, 16),  # Right arm
        (0, 4, 5, 6),     # Left leg
        (0, 1, 2, 3),     # Right leg
    )

    def __init__(self, dim, k_neighbors=3):
        super().__init__()
        self.joint = DynamicHypergraphConvolution(dim, k_neighbors)
        membership = torch.zeros(17, len(self.PARTS))
        for part, joints in enumerate(self.PARTS):
            membership[list(joints), part] = 1
        self.register_buffer("part_membership", membership)
        self.part_projection = nn.Linear(dim, dim)
        self.body_projection = nn.Linear(dim, dim)
        self.fusion = nn.Linear(3 * dim, dim)

    def part_features(self, x):
        membership = self.part_membership.to(x.dtype)
        means = torch.matmul(membership.transpose(0, 1), x)
        means = means / membership.sum(dim=0).unsqueeze(-1)
        transformed = self.part_projection(F.gelu(means))
        broadcast = torch.matmul(membership, transformed)
        # Shared pelvis/thorax joints receive the average of all their parts.
        return broadcast / membership.sum(dim=1).unsqueeze(-1)

    def forward(self, x):
        joint = self.joint(x)
        part = self.part_features(x)
        body = self.body_projection(x.mean(dim=-2, keepdim=True)).expand_as(x)
        return x + self.fusion(torch.cat((joint, part, body), dim=-1))


class SemanticSpatialBlock(nn.Module):
    """Joint-wise semantic attention followed by FFN and input residual, Eq. (8)."""

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
        return x + self.ffn(self.ffn_norm(attended))


class HopRelationEncoder(nn.Module):
    """Embed the original implementation's four-hop 2D geometric relations.

    Squared pairwise distances are row-normalized, restricted to each of the
    four skeletal hop neighborhoods, then concatenated into 17*4 features.
    The paper specifies this gating input without a separate encoding equation.
    """

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.register_buffer("hop_matrices", build_hop_matrices(max_hop=4))
        self.embedding = nn.Linear(17 * 4, dim)
        self.joint_position = nn.Parameter(torch.zeros(1, 17, dim))
        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 2, dropout)

    def forward(self, coordinates):
        differences = coordinates.unsqueeze(-2) - coordinates.unsqueeze(-3)
        distances = differences.square().sum(dim=-1)
        distances = distances / distances.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        relations = distances.unsqueeze(-3) * self.hop_matrices
        relations = relations.transpose(-3, -2).flatten(-2)
        features = self.embedding(relations) + self.joint_position
        return features + self.ffn(self.norm(features))


class MultiplexInteractiveSpatialEncoder(nn.Module):
    """MXS: two parallel streams from the same embedded 2D joint tokens.

    Input: (B*T, 17, 2). Outputs: spatial and hop features (B*T, 17, C).
    """

    def __init__(self, embed_dim=32, depth=4, num_heads=8, k_neighbors=3,
                 mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.embedding = nn.Linear(2, embed_dim)
        self.joint_position = nn.Parameter(torch.zeros(1, 17, embed_dim))
        self.structural = MultiGranularityHypergraph(embed_dim, k_neighbors)
        self.semantic = nn.Sequential(*[
            SemanticSpatialBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.hop_encoder = HopRelationEncoder(embed_dim, dropout)

    def forward(self, coordinates):
        embedded = self.embedding(coordinates) + self.joint_position
        # Both streams receive embedded, rather than feeding semantic features
        # through the hypergraph. Eq. (9) also preserves a direct input residual.
        spatial = self.structural(embedded) + self.semantic(embedded) + embedded
        return spatial, self.hop_encoder(coordinates)
