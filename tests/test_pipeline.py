"""Data, evaluation, and PMSS regression checks using only synthetic poses."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from common.h36m_dataset import Human36mDataset
from common.metrics import ActionMetrics, p_mpjpe
from common.skeleton import H36M_KEPT_JOINTS, H36M_PARENTS, build_hop_matrices
from training.pmss import ProgressiveMultiStageSupervision, stage_for_epoch


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.poses = np.arange(5 * 32 * 3, dtype=np.float32).reshape(5, 32, 3) / 1000
        self.poses[:, 1:, 0] += np.arange(5)[:, None]
        self.views = [np.full((7, 17, 2), [100 + view, 200 + view], dtype=np.float32)
                      for view in range(4)]
        for view in self.views:
            view[:, :, 0] += np.arange(7)[:, None]
        np.savez(self.root / "3d.npz", positions_3d={"S1": {
            "Walking 1": self.poses, "Eating 1": self.poses}})
        np.savez(self.root / "2d.npz", positions_2d={"S1": {
            "Walking 1": self.views, "Eating 1": self.views}})

    def dataset(self, **kwargs):
        return Human36mDataset(self.root / "3d.npz", self.root / "2d.npz",
                               "S1", frames=3, actions="Walking", **kwargs)

    def test_synchronized_windows_and_joint_mapping(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset), 5)
        sample = dataset[2]
        self.assertEqual(sample["input_2d"].shape, (3, 4, 17, 2))
        self.assertEqual(sample["target_3d"].shape, (3, 17, 3))
        self.assertEqual(sample["action"], "Walking 1")
        target = self.poses[:, H36M_KEPT_JOINTS] - self.poses[:, :1]
        np.testing.assert_allclose(sample["target_3d"], target[1:4])
        np.testing.assert_array_equal(sample["target_3d"][:, 0], 0)
        expected_2d = self.views[0][2] * 0.002 - [1, 1.002]
        np.testing.assert_allclose(sample["input_2d"][1, 0], expected_2d, atol=1e-7)
        np.testing.assert_array_equal(dataset[0]["target_3d"][0], target[0])
        np.testing.assert_array_equal(dataset[0]["target_3d"][1], target[0])
        np.testing.assert_array_equal(dataset[-1]["target_3d"][-1], target[-1])

    def test_sampling_and_synchronized_reversal(self):
        dataset = self.dataset(downsample=2, stride=2)
        self.assertEqual(len(dataset), 2)
        target = self.poses[:, H36M_KEPT_JOINTS] - self.poses[:, :1]
        np.testing.assert_allclose(dataset[1]["target_3d"], target[[2, 4, 4]])
        augmented = self.dataset(reverse_augmentation=True)
        self.assertEqual(len(augmented), 10)
        for key in ("input_2d", "target_3d"):
            np.testing.assert_array_equal(augmented[4][key][::-1], augmented[5][key])

    def test_short_2d_sequence_is_rejected(self):
        np.savez(self.root / "2d.npz", positions_2d={"S1": {
            "Walking 1": [view[:2] for view in self.views]}})
        with self.assertRaisesRegex(ValueError, "shorter"):
            self.dataset()


class MetricTests(unittest.TestCase):
    def test_similarity_alignment_and_degenerate_poses(self):
        target = np.random.default_rng(1).normal(size=(2, 3, 17, 3))
        rotation = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])
        prediction = 2.3 * (target @ rotation) + [3, 4, 5]
        np.testing.assert_allclose(p_mpjpe(prediction, target), 0, atol=1e-12)
        np.testing.assert_array_equal(p_mpjpe(np.zeros_like(target), np.zeros_like(target)), 0)
        self.assertTrue(np.isfinite(p_mpjpe(np.zeros_like(target), target)).all())

    def test_mixed_actions_use_their_own_errors(self):
        target = torch.tensor(np.random.default_rng(2).normal(size=(2, 3, 17, 3))).float()
        prediction = target.clone()
        prediction[1, ..., 0] += 1
        metrics = ActionMetrics()
        metrics.update(prediction, target, ["Walking 1", "Eating 1"])
        result = metrics.compute()
        self.assertAlmostEqual(result["Walking"]["mpjpe"], 0)
        self.assertAlmostEqual(result["Eating"]["mpjpe"], 1000, places=3)
        self.assertAlmostEqual(result["Average"]["mpjpe"], 500, places=3)
        self.assertLess(result["Average"]["p_mpjpe"], 1e-3)

    def test_hop_matrices_have_independent_degree_normalization(self):
        hops = build_hop_matrices()
        self.assertEqual(tuple(hops.shape), (4, 17, 17))
        torch.testing.assert_close(hops.sum(-1), torch.ones(4, 17))
        self.assertTrue(torch.all(hops.diagonal(dim1=-2, dim2=-1) == 0))
        self.assertGreater(hops[0, 0, 1], 0)
        self.assertGreater(hops[1, 0, 2], 0)
        self.assertEqual(hops[0, 0, 2], 0)


class PMSSTests(unittest.TestCase):
    def test_stage_boundaries(self):
        self.assertEqual([stage_for_epoch(epoch) for epoch in (1, 35, 36, 45, 46, 50)],
                         [1, 1, 2, 2, 3, 3])
        with self.assertRaises(ValueError):
            stage_for_epoch(0)
        with self.assertRaises(ValueError):
            stage_for_epoch(1, 45, 35)

    def test_bone_objective_is_temporal_standard_deviation(self):
        depth = torch.zeros(17)
        for joint, parent in enumerate(H36M_PARENTS):
            if parent >= 0:
                depth[joint] = depth[parent] + 1
        prediction = torch.zeros(1, 2, 17, 3)
        prediction[0, :, :, 0] = torch.tensor([1., 5.])[:, None] * depth
        criterion = ProgressiveMultiStageSupervision()
        # Every bone has lengths [1, 5]: population std = 2, variance = 4.
        self.assertAlmostEqual(criterion.bone_length_variation(prediction).item(), 2)
        one_frame = prediction[:, :1].clone().requires_grad_()
        regularizer = criterion.bone_length_variation(one_frame)
        self.assertEqual(regularizer.item(), 0)
        regularizer.backward()
        self.assertTrue(torch.isfinite(one_frame.grad).all())
        inputs = torch.zeros(1, 2, 4, 17, 2)
        loss, _ = criterion(prediction, prediction.clone(), inputs, 2)
        self.assertAlmostEqual(loss.item(), 0.9 * 2, places=6)

    def test_stage3_uses_coordinate_l1_sum_and_paper_weights(self):
        class XYProjection(torch.nn.Module):
            def forward(self, poses):
                centered = poses[..., :2] - poses[:, :, :1, :2]
                return centered.unsqueeze(2).expand(-1, -1, 4, -1, -1)

        prediction = torch.zeros(1, 2, 17, 3)
        prediction[:, :, 1:, :2] = torch.tensor([1., 2.])
        target = prediction.clone()
        target[:, :, 1:, 2] = 4
        criterion = ProgressiveMultiStageSupervision()
        criterion.projection = XYProjection()
        inputs = torch.zeros(1, 2, 4, 17, 2)
        expected_pose = 4 * 16 / 17
        stage1, _ = criterion(prediction, target, inputs, 1)
        self.assertAlmostEqual(stage1.item(), expected_pose, places=6)
        stage3, components = criterion(prediction, target, inputs, 3)
        expected_projection = (1 + 2) * 16 / 17
        self.assertAlmostEqual(components["projection"].item(), expected_projection, places=6)
        self.assertAlmostEqual(stage3.item(), 0.01 * expected_pose + 0.99 * expected_projection, places=6)

    def test_projection_receives_gradients_only_in_stage3(self):
        torch.manual_seed(7)
        criterion = ProgressiveMultiStageSupervision(pose_weight=0)
        prediction = torch.randn(2, 3, 17, 3, requires_grad=True)
        target = torch.zeros_like(prediction)
        inputs = torch.randn(2, 3, 4, 17, 2)
        for stage in (1, 2):
            criterion.zero_grad(set_to_none=True)
            criterion(prediction, target, inputs, stage)[0].backward()
            self.assertTrue(all(parameter.grad is None for parameter in criterion.projection.parameters()))
        prediction.grad = None
        criterion(prediction, target, inputs, 3)[0].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        for view in criterion.projection.views:
            self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in view.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
