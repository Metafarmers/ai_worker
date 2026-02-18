# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# End-pose relative action inference data config for FFW SG2 Rev1 robot.
#
# This config supports inference with the endpose_grip_672x376_abs_state_rel_action algorithm:
# - State (18D, absolute): end_pose_left(7) + gripper_l(1) + end_pose_right(7) + gripper_r(1) + head(2)
# - Action (16D, relative + absolute): rel_end_pose_left(7) + gripper_l(1) + rel_end_pose_right(7) + gripper_r(1)
#
# Uses 3 cameras:
# - video.ego_view (head camera, 672x376)
# - video.cam_wrist_left (left wrist, 424x240)
# - video.cam_wrist_right (right wrist, 424x240)

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any, Dict
import cv2
import numpy as np

# Add Isaac-GR00T to path if not already there
ISAAC_GROOT_PATH = Path("/workspace/Isaac-GR00T")
if ISAAC_GROOT_PATH.exists() and str(ISAAC_GROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_PATH))

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.model.transforms import GR00TTransform
import torch
from gr00t.experiment.data_config import BaseDataConfig


class VideoToTensorSimple(ModalityTransform):
    """Simple video to tensor conversion WITHOUT resolution checks."""

    apply_to: List[str]

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            video = data[key]
            if not isinstance(video, np.ndarray):
                continue
            tensor = torch.from_numpy(video.copy())
            data[key] = tensor
        return data


class VideoToNumpySimple(ModalityTransform):
    """Simple tensor to numpy conversion WITHOUT resolution checks."""

    apply_to: List[str]

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            video = data[key]
            if isinstance(video, torch.Tensor):
                data[key] = video.numpy()
        return data


class VideoResizeIndependent(ModalityTransform):
    """Resize each video key independently to the same resolution.

    This transform resizes each camera view BEFORE other video transforms
    concatenate them together. This allows using multiple cameras with
    different original resolutions.
    """

    apply_to: List[str]
    height: int
    width: int
    interpolation: str = "linear"

    _INTERPOLATION_MAP = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        interp = self._INTERPOLATION_MAP.get(self.interpolation, cv2.INTER_LINEAR)

        for key in self.apply_to:
            if key not in data:
                continue

            video = data[key]
            original_dtype = video.dtype

            if video.ndim == 5:
                resized = []
                for b in range(video.shape[0]):
                    batch_frames = []
                    for t in range(video.shape[1]):
                        frame = video[b, t]
                        resized_frame = cv2.resize(frame, (self.width, self.height), interpolation=interp)
                        batch_frames.append(resized_frame)
                    resized.append(np.stack(batch_frames, axis=0))
                result = np.stack(resized, axis=0)
            elif video.ndim == 4:
                resized_frames = []
                for t in range(video.shape[0]):
                    frame = video[t]
                    resized_frame = cv2.resize(frame, (self.width, self.height), interpolation=interp)
                    resized_frames.append(resized_frame)
                result = np.stack(resized_frames, axis=0)
            elif video.ndim == 3:
                resized = cv2.resize(video, (self.width, self.height), interpolation=interp)
                result = resized[np.newaxis, ...]
            else:
                raise ValueError(f"Unsupported video shape {video.shape} for key {key}")

            result = np.ascontiguousarray(result, dtype=original_dtype)
            data[key] = result

        return data


@dataclass
class FFWSG2EndPoseRelActionInferenceConfig(BaseDataConfig):
    """End-pose relative action inference config for FFW SG2 Rev1 dual-arm robot.

    This config is designed for the endpose_grip_672x376_abs_state_rel_action algorithm:
    - State: absolute end-effector poses + joints
    - Action: relative end-effector pose deltas + absolute grippers

    Camera Configuration (3 cameras):
    - video.ego_view: 672x376 (head camera)
    - video.cam_wrist_left: 424x240 (left wrist)
    - video.cam_wrist_right: 424x240 (right wrist)

    State Configuration (18D total, absolute):
    - state.end_pose_left: 7D (x, y, z, qx, qy, qz, qw)
    - state.gripper_l: 1D (gripper_l_joint1)
    - state.end_pose_right: 7D (x, y, z, qx, qy, qz, qw)
    - state.gripper_r: 1D (gripper_r_joint1)
    - state.head: 2D (head_joint1, head_joint2)

    Action Configuration (16D total, relative poses + absolute grippers):
    - action.rel_end_pose_left: 7D (dx, dy, dz, dqx, dqy, dqz, dqw) - relative delta
    - action.gripper_l: 1D (gripper_l_joint1) - absolute
    - action.rel_end_pose_right: 7D (dx, dy, dz, dqx, dqy, dqz, dqw) - relative delta
    - action.gripper_r: 1D (gripper_r_joint1) - absolute
    """

    # Camera keys - 3 cameras for multi-cam inference
    video_keys: List[str] = field(default_factory=lambda: [
        "video.ego_view",
        "video.cam_wrist_left",
        "video.cam_wrist_right"
    ])

    # State keys (18D total, absolute)
    state_keys: List[str] = field(default_factory=lambda: [
        "state.end_pose_left",   # 7D absolute
        "state.gripper_l",       # 1D absolute
        "state.end_pose_right",  # 7D absolute
        "state.gripper_r",       # 1D absolute
        "state.head",            # 2D absolute
    ])

    # Action keys (16D total, relative poses + absolute grippers)
    action_keys: List[str] = field(default_factory=lambda: [
        "action.rel_end_pose_left",   # 7D relative delta
        "action.gripper_l",           # 1D absolute
        "action.rel_end_pose_right",  # 7D relative delta
        "action.gripper_r",           # 1D absolute
    ])

    # Language key for task instruction
    language_keys: List[str] = field(default_factory=lambda: [
        "annotation.human.action.task_description"
    ])

    # Observation and action horizons
    observation_indices: List[int] = field(default_factory=lambda: [0])
    action_indices: List[int] = field(default_factory=lambda: list(range(16)))

    # Target resolution for all cameras
    target_height: int = 224
    target_width: int = 224

    def modality_config(self) -> dict:
        """Return modality configuration for the model."""
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        """Return the modality transform for end-pose relative action inference."""
        transforms = [
            # Resize each camera independently to 224x224
            VideoResizeIndependent(
                apply_to=self.video_keys,
                height=self.target_height,
                width=self.target_width,
                interpolation="linear",
            ),
            VideoToTensorSimple(apply_to=self.video_keys),
            VideoToNumpySimple(apply_to=self.video_keys),
            # State transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # Action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # Concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # Model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


@dataclass
class FFWSG2EndPoseRelActionSingleCamInferenceConfig(FFWSG2EndPoseRelActionInferenceConfig):
    """Single-camera inference config - backward compatible with single ego_view."""

    video_keys: List[str] = field(default_factory=lambda: ["video.ego_view"])
