"""Regression checks for paper equations and stage-aware model execution."""

import unittest

import torch
from torch.nn import functional as F

from model import MXPose
from model.spatial import DynamicHypergraphConvolution, MultiGranularityHypergraph
from model.temporal import KinematicRefinementBranch


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_hypergraph_matches_spectral_equation(self):
        layer = DynamicHypergraphConvolution(8, k_neighbors=3)
        features = torch.randn(2, 17, 8)
        with torch.no_grad():
            layer.edge_logits.copy_(torch.linspace(-1, 1, 17))
        incidence = layer.incidence(features)
        self.assertTrue((incidence.sum(-2) == 3).all())
        self.assertTrue((incidence.diagonal(dim1=-2, dim2=-1) == 1).all())
        weights = F.softplus(layer.edge_logits)
        vertex_degree = torch.diag_embed((incidence * weights).sum(-1).rsqrt())
        edge_degree = torch.diag_embed(incidence.sum(-2).reciprocal())
        expected = F.gelu(
            vertex_degree @ incidence @ torch.diag(weights) @ edge_degree
            @ incidence.transpose(-1, -2) @ vertex_degree @ layer.projection(features)
        )
        torch.testing.assert_close(layer(features), expected)
        changed = layer.incidence(torch.randn_like(features))
        self.assertFalse(torch.equal(incidence, changed))

    def test_shared_joints_average_all_anatomical_parts(self):
        layer = MultiGranularityHypergraph(8)
        features = torch.randn(2, 17, 8)
        actual = layer.part_features(features)
        part_features = [
            layer.part_projection(F.gelu(features[:, joints].mean(1)))
            for joints in layer.PARTS
        ]
        for joint in range(17):
            expected = torch.stack([
                values for joints, values in zip(layer.PARTS, part_features)
                if joint in joints
            ]).mean(0)
            torch.testing.assert_close(actual[:, joint], expected)

    def test_all_stages_preserve_frames_and_activate_expected_parameters(self):
        model = MXPose(frames=9, embed_dim=8, depth=1, num_heads=2,
                       kinematic_dropout=0.0)
        for stage in (1, 2, 3):
            with self.subTest(stage=stage):
                model.zero_grad(set_to_none=True)
                inputs = torch.randn(2, 9, 4, 17, 2, requires_grad=True)
                output = model(inputs, stage=stage)
                self.assertEqual(output.shape, (2, 9, 17, 3))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                for name, parameter in model.named_parameters():
                    inactive = stage == 1 and name.startswith("mxt.krb.")
                    if inactive:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0, name)

    def test_single_frame_zero_pose_is_finite_in_training_and_evaluation(self):
        model = MXPose(frames=9, embed_dim=8, depth=1, num_heads=2)
        inputs = torch.zeros(1, 1, 4, 17, 2)
        for training in (True, False):
            model.train(training)
            for stage in (1, 3):
                with self.subTest(training=training, stage=stage):
                    output = model(inputs, stage=stage)
                    self.assertEqual(output.shape, (1, 1, 17, 3))
                    self.assertTrue(torch.isfinite(output).all())
        motion = KinematicRefinementBranch.finite_difference(torch.ones(1, 1, 8))
        self.assertTrue(torch.equal(motion, torch.zeros_like(motion)))


if __name__ == "__main__":
    unittest.main()
