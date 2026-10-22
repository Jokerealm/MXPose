"""Synchronized Human3.6M windows from the standard preprocessed NPZ files.

The 3D archive stores world-coordinate poses in meters. The 2D archive
contains four pixel-coordinate detections per sequence, in camera order
54138969, 55011271, 58860488, 60457274.
"""

from bisect import bisect_right
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from common.skeleton import H36M_KEPT_JOINTS


TRAIN_SUBJECTS = ("S1", "S5", "S6", "S7", "S8")
TEST_SUBJECTS = ("S9", "S11")
CAMERA_RESOLUTIONS = ((1000, 1002), (1000, 1000), (1000, 1000), (1000, 1002))


def normalize_screen_coordinates(points, width, height):
    """Normalize pixel coordinates while preserving the image aspect ratio."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[-1] != 2:
        raise ValueError("2D points must have two coordinates")
    return points * np.float32(2 / width) - np.array([1, height / width], dtype=np.float32)


def _read_positions(path, key):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Human3.6M archive does not exist: {path}")
    # Standard Human3.6M NPZ archives contain nested Python dictionaries.
    # Only load archives obtained from a trusted preprocessing pipeline.
    with np.load(path, allow_pickle=True) as archive:
        if key not in archive:
            raise ValueError(f"{path} is missing the '{key}' dictionary")
        positions = archive[key].item()
    if not isinstance(positions, dict):
        raise ValueError(f"'{key}' in {path} must be a subject/action dictionary")
    return positions


class Human36mDataset(Dataset):
    """Return synchronized, centered pose windows with replicated boundaries.

    Every item contains ``input_2d`` (T, 4, 17, 2), ``target_3d``
    (T, 17, 3), ``action`` and ``subject``. Targets are root-relative world
    coordinates in meters. ``stride`` controls the spacing of window centers;
    ``downsample`` subsamples each complete sequence before windowing.

    Optional reversal duplicates each window with its time-reversed copy.
    Both the 2D inputs and the 3D targets are reversed together.
    """

    def __init__(self, path_3d, path_2d, subjects, frames=27, stride=1,
                 downsample=1, actions=None, reverse_augmentation=False):
        if not isinstance(frames, int) or frames < 1 or frames % 2 != 1:
            raise ValueError("frames must be a positive odd integer")
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")
        if not isinstance(downsample, int) or downsample < 1:
            raise ValueError("downsample must be a positive integer")
        if isinstance(subjects, str):
            subjects = [subject.strip() for subject in subjects.split(",") if subject.strip()]
        subjects = list(dict.fromkeys(subjects))
        if not subjects:
            raise ValueError("At least one subject is required")
        if isinstance(actions, str):
            actions = None if actions in ("*", "all", "All") else actions.split(",")
        action_filter = {action.strip() for action in actions} if actions is not None else None

        self.frames = frames
        self.stride = stride
        self.reverse_augmentation = bool(reverse_augmentation)
        self.sequences = []
        self._ends = []
        self.num_frames = 0
        poses_3d = _read_positions(path_3d, "positions_3d")
        poses_2d = _read_positions(path_2d, "positions_2d")

        for subject in subjects:
            if subject not in poses_3d or subject not in poses_2d:
                raise ValueError(f"Subject {subject} must be present in both archives")
            for action, raw_3d in poses_3d[subject].items():
                action_name = action.split(" ")[0]
                if action_filter is not None and action not in action_filter and action_name not in action_filter:
                    continue
                if action not in poses_2d[subject]:
                    raise ValueError(f"Missing 2D detections for {subject}/{action}")
                target = np.asarray(raw_3d, dtype=np.float32)
                if target.ndim != 3 or target.shape[-1] != 3 or target.shape[1] not in (17, 32):
                    raise ValueError(f"Expected (F, 17 or 32, 3) poses for {subject}/{action}, got {target.shape}")
                length = target.shape[0]
                if length == 0:
                    raise ValueError(f"Empty sequence: {subject}/{action}")
                if target.shape[1] == 32:
                    target = target[:, H36M_KEPT_JOINTS]
                target = target[::downsample].copy()
                target -= target[:, :1].copy()
                if not np.isfinite(target).all():
                    raise ValueError(f"Non-finite 3D coordinates in {subject}/{action}")

                views = poses_2d[subject][action]
                if len(views) != 4:
                    raise ValueError(f"Expected four synchronized camera views for {subject}/{action}")
                normalized_views = []
                for camera, (view, (width, height)) in enumerate(zip(views, CAMERA_RESOLUTIONS)):
                    view = np.asarray(view, dtype=np.float32)
                    if view.ndim != 3 or view.shape[1] != 17 or view.shape[2] < 2:
                        raise ValueError(f"Expected (F, 17, 2 or more) detections for {subject}/{action}, camera {camera}")
                    if len(view) < length:
                        raise ValueError(f"2D sequence is shorter than 3D for {subject}/{action}, camera {camera}")
                    # Detectors can retain extra frames beyond the motion-capture sequence.
                    view = view[:length:downsample, :, :2]
                    if not np.isfinite(view).all():
                        raise ValueError(f"Non-finite 2D coordinates in {subject}/{action}, camera {camera}")
                    normalized_views.append(normalize_screen_coordinates(view, width, height))
                inputs = np.stack(normalized_views, axis=1)
                self.sequences.append({
                    "input_2d": inputs,
                    "target_3d": target,
                    "subject": subject,
                    "action": action,
                })
                self.num_frames += len(target)
                count = (len(target) + stride - 1) // stride
                self._ends.append((self._ends[-1] if self._ends else 0) + count)

        if not self.sequences:
            raise ValueError("No Human3.6M sequences match the selected subjects and actions")
        self._offsets = np.arange(frames) - frames // 2

    def __len__(self):
        return self._ends[-1] * (2 if self.reverse_augmentation else 1)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        reverse = self.reverse_augmentation and index % 2 == 1
        window_index = index // 2 if self.reverse_augmentation else index
        sequence_index = bisect_right(self._ends, window_index)
        previous_end = self._ends[sequence_index - 1] if sequence_index else 0
        center = (window_index - previous_end) * self.stride
        sequence = self.sequences[sequence_index]
        frame_indices = np.clip(center + self._offsets, 0, len(sequence["target_3d"]) - 1)
        if reverse:
            frame_indices = frame_indices[::-1]
        return {
            "input_2d": sequence["input_2d"][frame_indices].copy(),
            "target_3d": sequence["target_3d"][frame_indices].copy(),
            "action": sequence["action"],
            "subject": sequence["subject"],
        }
