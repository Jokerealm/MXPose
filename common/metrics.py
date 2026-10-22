"""Human3.6M Protocol 1 (MPJPE) and Protocol 2 (P-MPJPE)."""

from collections import defaultdict

import numpy as np
import torch


def mpjpe(predicted, target):
    """Mean Euclidean joint error, in the same units as the input tensors."""
    if predicted.shape != target.shape:
        raise ValueError(f"Pose shapes differ: {predicted.shape} and {target.shape}")
    return torch.linalg.vector_norm(predicted - target, dim=-1).mean()


def p_mpjpe(predicted, target):
    """Per-pose error after similarity alignment (translation, rotation, scale).

    Accept arrays of shape (..., J, 3); return an array with shape (...,).
    Reflections are excluded. Degenerate zero-extent predictions are handled
    without producing NaNs.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim < 2 or predicted.shape[-1] != 3:
        raise ValueError("Predicted and target poses must have matching (..., J, 3) shapes")
    leading_shape = predicted.shape[:-2]
    predicted = predicted.reshape(-1, predicted.shape[-2], 3)
    target = target.reshape(-1, target.shape[-2], 3)
    mu_target = target.mean(axis=1, keepdims=True)
    mu_predicted = predicted.mean(axis=1, keepdims=True)
    centered_target = target - mu_target
    centered_predicted = predicted - mu_predicted
    target_norm = np.linalg.norm(centered_target, axis=(1, 2), keepdims=True)
    predicted_norm = np.linalg.norm(centered_predicted, axis=(1, 2), keepdims=True)
    epsilon = np.finfo(np.float64).eps
    unit_target = centered_target / np.maximum(target_norm, epsilon)
    unit_predicted = centered_predicted / np.maximum(predicted_norm, epsilon)
    covariance = unit_target.transpose(0, 2, 1) @ unit_predicted
    left, singular_values, right_t = np.linalg.svd(covariance)
    right = right_t.transpose(0, 2, 1)
    rotation = right @ left.transpose(0, 2, 1)
    sign = np.where(np.linalg.det(rotation) < 0, -1.0, 1.0)
    right[:, :, -1] *= sign[:, None]
    singular_values[:, -1] *= sign
    rotation = right @ left.transpose(0, 2, 1)
    scale = singular_values.sum(axis=1)[:, None, None] * target_norm / np.maximum(predicted_norm, epsilon)
    aligned = scale * (centered_predicted @ rotation) + mu_target
    errors = np.linalg.norm(aligned - target, axis=-1).mean(axis=-1)
    return errors.reshape(leading_shape)


class ActionMetrics:
    """Accumulate per-frame errors and report action-averaged results in mm."""

    def __init__(self):
        self._sums = defaultdict(lambda: np.zeros(3, dtype=np.float64))

    def update(self, predicted, target, actions):
        predicted = predicted.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        if predicted.shape != target.shape or predicted.ndim not in (3, 4):
            raise ValueError("Expected matching (B, J, 3) or (B, T, J, 3) tensors")
        if len(actions) != len(predicted):
            raise ValueError("Each predicted sample must have an action label")
        p1 = np.linalg.norm(predicted - target, axis=-1).mean(axis=-1)
        p2 = p_mpjpe(predicted, target)
        for index, action in enumerate(actions):
            # Merge the two trials of each action for standard H36M reporting.
            name = action.split(" ")[0]
            self._sums[name] += (np.sum(p1[index]), np.sum(p2[index]), np.size(p1[index]))

    def compute(self):
        """Return {action: {'mpjpe': mm, 'p_mpjpe': mm}}, plus 'Average'."""
        if not self._sums:
            raise ValueError("No evaluation samples have been accumulated")
        results = {
            action: {"mpjpe": float(p1 / count * 1000), "p_mpjpe": float(p2 / count * 1000)}
            for action, (p1, p2, count) in sorted(self._sums.items())
        }
        results["Average"] = {
            metric: float(np.mean([values[metric] for values in results.values()]))
            for metric in ("mpjpe", "p_mpjpe")
        }
        return results
