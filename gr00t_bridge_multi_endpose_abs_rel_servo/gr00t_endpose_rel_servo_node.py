#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# GR00T End-Pose Relative Action Inference ROS2 Node — MoveIt Servo version
#
# This node replaces the blocking /compute_ik approach with MoveIt Servo's
# PoseCommand mode for real-time dual-arm control with built-in:
#   - IK solving (via Jacobian or kinematics plugin)
#   - Motion smoothing (Butterworth / AccelerationLimited / Ruckig)
#   - Singularity detection and velocity scaling
#   - Collision checking and deceleration
#   - Joint position/velocity limit enforcement
#
# Camera Inputs:
#   - video.ego_view: Head camera
#   - video.cam_wrist_left: Left wrist camera
#   - video.cam_wrist_right: Right wrist camera
#
# State Inputs (18D, absolute):
#   - /end_pose_left (PoseStamped): 7D (x, y, z, qx, qy, qz, qw)
#   - /joint_states gripper_l_joint1: 1D
#   - /end_pose_right (PoseStamped): 7D
#   - /joint_states gripper_r_joint1: 1D
#   - /joint_states head_joint1, head_joint2: 2D
#
# Action Pipeline:
#   model output (relative delta 7D) → current + delta = absolute PoseStamped
#   → publish to MoveIt Servo pose_command_in_topic
#   → Servo handles IK + smoothing + safety → JointTrajectory to controller
#   Grippers: absolute values published as separate JointTrajectory

import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import cv2
from dataclasses import dataclass
from collections import deque
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header, Float64
from builtin_interfaces.msg import Duration
import tf2_ros

from moveit_msgs.srv import ServoCommandType
from moveit_msgs.msg import ServoStatus

ISAAC_GROOT_PATH = Path("/workspace/Isaac-GR00T")
if ISAAC_GROOT_PATH.exists() and str(ISAAC_GROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_PATH))


# =============================================================================
# Quaternion helpers
# =============================================================================

def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute the conjugate of a quaternion [qx, qy, qz, qw]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [qx, qy, qz, qw] using Hamilton product."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def apply_relative_action(current_pose: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    """Convert a relative pose delta to an absolute target pose.

    target_pos  = current_pos + delta_pos
    target_quat = delta_quat * current_quat  (data: delta = q_curr * q_prev⁻¹ → q_next = delta * q_tracking)
    """
    target_pos = current_pose[:3] + delta_pose[:3]

    current_quat = current_pose[3:7]
    delta_quat = delta_pose[3:7]
    target_quat = quaternion_multiply(delta_quat, current_quat)

    quat_norm = np.linalg.norm(target_quat)
    if quat_norm > 1e-6:
        target_quat = target_quat / quat_norm
    else:
        target_quat = current_quat.copy()

    return np.concatenate([target_pos, target_quat])


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class JointConfig:
    LEFT_GRIPPER_JOINTS = ["gripper_l_joint1"]
    RIGHT_GRIPPER_JOINTS = ["gripper_r_joint1"]
    HEAD_JOINTS = ["head_joint1", "head_joint2"]


@dataclass
class CameraConfig:
    EGO_VIEW_TOPIC = "/zed/zed_node/left/image_rect_color/compressed"
    WRIST_LEFT_TOPIC = "/camera_left/camera_left/color/image_rect_raw/compressed"
    WRIST_RIGHT_TOPIC = "/camera_right/camera_right/color/image_rect_raw/compressed"


# =============================================================================
# Action output dataclass
# =============================================================================

@dataclass
class EndPoseRelActionOutput:
    """Structured output from GR00T relative end-pose inference."""
    rel_end_pose_left: np.ndarray
    gripper_l: np.ndarray
    rel_end_pose_right: np.ndarray
    gripper_r: np.ndarray
    raw_dict: Dict[str, np.ndarray]

    def get_first_timestep(self) -> Dict[str, np.ndarray]:
        return {
            "rel_end_pose_left": self.rel_end_pose_left[0] if len(self.rel_end_pose_left.shape) > 1 else self.rel_end_pose_left,
            "gripper_l": self.gripper_l[0] if len(self.gripper_l.shape) > 1 else self.gripper_l,
            "rel_end_pose_right": self.rel_end_pose_right[0] if len(self.rel_end_pose_right.shape) > 1 else self.rel_end_pose_right,
            "gripper_r": self.gripper_r[0] if len(self.gripper_r.shape) > 1 else self.gripper_r,
        }

    def get_concatenated_action(self, timestep: int = 0) -> np.ndarray:
        first = self.get_first_timestep() if timestep == 0 else {
            "rel_end_pose_left": self.rel_end_pose_left[timestep],
            "gripper_l": self.gripper_l[timestep],
            "rel_end_pose_right": self.rel_end_pose_right[timestep],
            "gripper_r": self.gripper_r[timestep],
        }
        return np.concatenate([
            first["rel_end_pose_left"].flatten(),
            first["gripper_l"].flatten(),
            first["rel_end_pose_right"].flatten(),
            first["gripper_r"].flatten(),
        ])


# =============================================================================
# Model wrapper
# =============================================================================

class Gr00tEndPoseRelInferenceWrapper:
    """Wrapper for Isaac-GR00T model inference with relative end-pose action output."""

    STATE_DIMS = {
        "end_pose_left": 7,
        "gripper_l": 1,
        "end_pose_right": 7,
        "gripper_r": 1,
        "head": 2,
    }
    DEFAULT_VIDEO_KEYS = [
        "video.ego_view",
        "video.cam_wrist_left",
        "video.cam_wrist_right",
    ]
    TARGET_HEIGHT = 224
    TARGET_WIDTH = 224

    def __init__(
        self,
        checkpoint_path: str,
        embodiment_tag: str = "new_embodiment",
        device: str = "cuda",
        data_config: Optional[Any] = None,
        denoising_steps: Optional[int] = None,
        video_keys: Optional[List[str]] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.embodiment_tag = embodiment_tag
        self.device = device
        self.denoising_steps = denoising_steps

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        if data_config is None:
            from .ffw_sg2_endpose_rel_action_inference_config import FFWSG2EndPoseRelActionInferenceConfig
            self.data_config = FFWSG2EndPoseRelActionInferenceConfig()
        else:
            self.data_config = data_config

        if video_keys is not None:
            self.data_config.video_keys = video_keys

        self._video_keys = self.data_config.video_keys
        self._load_model()

        print(f"[Gr00tEndPoseRelWrapper] Model loaded successfully!")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Embodiment: {self.embodiment_tag}")
        print(f"  Device: {self.device}")
        print(f"  Action type: RELATIVE end-pose deltas + absolute grippers")
        print(f"  Output method: MoveIt Servo PoseCommand")

    def _load_model(self):
        from gr00t.model.policy import Gr00tPolicy
        import warnings

        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.policy = Gr00tPolicy(
                model_path=str(self.checkpoint_path),
                embodiment_tag=self.embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=self.denoising_steps,
                device=self.device,
            )

    def preprocess_image(self, image: np.ndarray, camera_key: str = "ego_view") -> Dict[str, np.ndarray]:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")
        if len(image.shape) != 3:
            raise ValueError(f"Expected (H, W, C) image, got shape {image.shape}")
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        image = image[np.newaxis, ...]
        return {f"video.{camera_key}": image}

    def preprocess_state(self, state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        observation = {}
        for key, expected_dim in self.STATE_DIMS.items():
            if key not in state:
                raise ValueError(f"Missing required state key: {key}")
            value = state[key]
            if not isinstance(value, np.ndarray):
                value = np.array(value, dtype=np.float64)
            else:
                value = value.astype(np.float64)
            value = value.flatten()
            if len(value) != expected_dim:
                raise ValueError(f"State {key} has {len(value)} dims, expected {expected_dim}")
            value = value[np.newaxis, ...]
            observation[f"state.{key}"] = value
        return observation

    def preprocess(
        self,
        images: Dict[str, np.ndarray],
        state: Dict[str, np.ndarray],
        task_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        required_cameras = [k.replace("video.", "") for k in self._video_keys]
        missing = set(required_cameras) - set(images.keys())
        if missing:
            raise ValueError(f"Missing required camera images: {missing}")

        observation = {}
        for camera_key, image in images.items():
            obs_image = self.preprocess_image(image, camera_key)
            observation.update(obs_image)

        obs_state = self.preprocess_state(state)
        observation.update(obs_state)

        if task_instruction is not None:
            observation["annotation.human.action.task_description"] = [task_instruction]

        return observation

    def predict(
        self,
        images: Dict[str, np.ndarray],
        state: Dict[str, np.ndarray],
        task_instruction: Optional[str] = None,
    ) -> EndPoseRelActionOutput:
        observation = self.preprocess(images, state, task_instruction)
        action_dict = self.policy.get_action(observation)
        return self._parse_action_output(action_dict)

    def _parse_action_output(self, action_dict: Dict[str, np.ndarray]) -> EndPoseRelActionOutput:
        return EndPoseRelActionOutput(
            rel_end_pose_left=action_dict.get("action.rel_end_pose_left", np.zeros((16, 7))),
            gripper_l=action_dict.get("action.gripper_l", np.zeros((16, 1))),
            rel_end_pose_right=action_dict.get("action.rel_end_pose_right", np.zeros((16, 7))),
            gripper_r=action_dict.get("action.gripper_r", np.zeros((16, 1))),
            raw_dict=action_dict,
        )


def prepare_observation_endpose_rel(
    images: Dict[str, np.ndarray],
    state: Dict[str, np.ndarray],
    task_instruction: str,
    wrapper: Gr00tEndPoseRelInferenceWrapper,
) -> Dict[str, Any]:
    observation = wrapper.preprocess(images, state, task_instruction)
    processed_obs = {}
    for key, value in observation.items():
        if isinstance(value, str):
            processed_obs[key] = value
        elif isinstance(value, list):
            processed_obs[key] = np.array(value)
        elif isinstance(value, np.ndarray):
            processed_obs[key] = value
        else:
            processed_obs[key] = np.array(value)
    return processed_obs


# =============================================================================
# Main ROS2 inference node — MoveIt Servo version
# =============================================================================

class Gr00tEndPoseRelServoNode(Node):
    """ROS2 node for GR00T end-pose RELATIVE action inference with MoveIt Servo.

    Instead of calling /compute_ik and manually publishing JointTrajectory,
    this node publishes PoseStamped to MoveIt Servo's pose_command_in_topic.
    Servo handles IK, smoothing, singularity, and collision internally.

    Grippers are published as separate JointTrajectory messages since Servo
    only controls the arm planning group joints.

    Topics Published (to Servo):
      - /servo_left/pose_target_cmds  (PoseStamped)
      - /servo_right/pose_target_cmds (PoseStamped)

    Topics Published (gripper, direct to controller):
      - /leader/joint_trajectory_command_broadcaster_left/joint_trajectory
      - /leader/joint_trajectory_command_broadcaster_right/joint_trajectory

    Services Called (once at startup):
      - /servo_left/switch_command_type  (ServoCommandType → POSE=2)
      - /servo_right/switch_command_type (ServoCommandType → POSE=2)
    """

    SERVO_CMD_POSE = 2  # ServoCommandType.Request constant for POSE mode

    def __init__(self, parameter_overrides: Optional[list] = None):
        super().__init__(
            "gr00t_endpose_rel_servo_node",
            parameter_overrides=parameter_overrides,
        )

        self._declare_parameters()

        self.checkpoint_path = self.get_parameter("checkpoint_path").value
        self.embodiment_tag = self.get_parameter("embodiment_tag").value
        self.device = self.get_parameter("device").value
        self.inference_rate = self.get_parameter("inference_rate").value
        self.dry_run = self.get_parameter("dry_run").value
        self.task_instruction = self.get_parameter("task_instruction").value

        self.ego_view_topic = self.get_parameter("ego_view_topic").value
        self.wrist_left_topic = self.get_parameter("wrist_left_topic").value
        self.wrist_right_topic = self.get_parameter("wrist_right_topic").value
        self.end_pose_left_topic = self.get_parameter("end_pose_left_topic").value
        self.end_pose_right_topic = self.get_parameter("end_pose_right_topic").value
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.print_rate = self.get_parameter("print_rate").value

        self.servo_left_ns = self.get_parameter("servo_left_ns").value
        self.servo_right_ns = self.get_parameter("servo_right_ns").value

        self.action_horizon = self.get_parameter("action_horizon").value
        self.action_horizon_execute = self.get_parameter("action_horizon_execute").value
        self.action_horizon_step_delay = self.get_parameter("action_horizon_step_delay").value

        self.gripper_step_duration_sec = self.get_parameter("gripper_step_duration_sec").value
        self.step_by_step = self.get_parameter("step_by_step").value

        # Step-by-step gate: blocks inference until user presses Enter
        self._step_approved = not self.step_by_step  # auto-approve if disabled
        self._step_lock = threading.Lock()
        if self.step_by_step:
            self._start_step_input_thread()

        # State buffers
        self.latest_images: Dict[str, Optional[np.ndarray]] = {
            "ego_view": None,
            "cam_wrist_left": None,
            "cam_wrist_right": None,
        }
        self.latest_image_stamps: Dict[str, Optional[Time]] = {
            "ego_view": None,
            "cam_wrist_left": None,
            "cam_wrist_right": None,
        }
        self.latest_end_pose_left: Optional[PoseStamped] = None
        self.latest_end_pose_right: Optional[PoseStamped] = None
        self.latest_joint_state: Optional[JointState] = None

        self.inference_count = 0
        self.last_print_time = time.time()
        self.inference_times: deque = deque(maxlen=100)

        self.last_status_print = time.time()
        self.status_print_interval = 5.0

        self.joint_config = JointConfig()

        self.action_buffer: Optional[EndPoseRelActionOutput] = None
        self.action_buffer_index = 0
        self.step_count = 0
        self.tracking_pose_left: Optional[np.ndarray] = None
        self.tracking_pose_right: Optional[np.ndarray] = None

        # Servo status tracking
        self._servo_status_left = None
        self._servo_status_right = None

        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._init_subscribers()
        self._init_servo_publishers()
        self._init_servo_service_clients()
        self._init_tf_broadcaster()

        if not self.dry_run:
            self._init_gripper_publishers()
        else:
            self.get_logger().info("Dry-run mode: Actions will NOT be published")

        self._load_model()

        self.inference_timer = self.create_timer(
            1.0 / self.inference_rate,
            self._inference_callback,
        )

        self._print_startup_info()

    # =========================================================================
    # Parameter declaration
    # =========================================================================

    def _declare_parameters(self):
        self.declare_parameter(
            "checkpoint_path",
            "/workspace/Isaac-GR00T/results/gr00t_endpose_rel_action/checkpoint-100000",
        )
        self.declare_parameter("embodiment_tag", "new_embodiment")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("inference_rate", 20.0)
        self.declare_parameter("dry_run", True)
        self.declare_parameter("task_instruction", "Execute the task")

        self.declare_parameter("ego_view_topic", CameraConfig.EGO_VIEW_TOPIC)
        self.declare_parameter("wrist_left_topic", CameraConfig.WRIST_LEFT_TOPIC)
        self.declare_parameter("wrist_right_topic", CameraConfig.WRIST_RIGHT_TOPIC)
        self.declare_parameter("end_pose_left_topic", "/end_pose_left")
        self.declare_parameter("end_pose_right_topic", "/end_pose_right")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("print_rate", 1.0)

        self.declare_parameter("servo_left_ns", "servo_left/servo_node")
        self.declare_parameter("servo_right_ns", "servo_right/servo_node")

        self.declare_parameter("action_horizon", 16)
        self.declare_parameter("action_horizon_execute", 4)
        self.declare_parameter("action_horizon_step_delay", 0.02)
        self.declare_parameter("gripper_step_duration_sec", 0.05)
        self.declare_parameter("step_by_step", False)

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_subscribers(self):
        self.ego_view_sub = self.create_subscription(
            CompressedImage, self.ego_view_topic,
            lambda msg: self._image_callback(msg, "ego_view"),
            self.sensor_qos,
        )
        self.wrist_left_sub = self.create_subscription(
            CompressedImage, self.wrist_left_topic,
            lambda msg: self._image_callback(msg, "cam_wrist_left"),
            self.sensor_qos,
        )
        self.wrist_right_sub = self.create_subscription(
            CompressedImage, self.wrist_right_topic,
            lambda msg: self._image_callback(msg, "cam_wrist_right"),
            self.sensor_qos,
        )
        self.end_pose_left_sub = self.create_subscription(
            PoseStamped, self.end_pose_left_topic,
            self._end_pose_left_callback, self.sensor_qos,
        )
        self.end_pose_right_sub = self.create_subscription(
            PoseStamped, self.end_pose_right_topic,
            self._end_pose_right_callback, self.sensor_qos,
        )
        self.joint_state_sub = self.create_subscription(
            JointState, self.joint_state_topic,
            self._joint_state_callback, self.sensor_qos,
        )

    def _init_servo_publishers(self):
        """Create PoseStamped publishers that feed into MoveIt Servo."""
        self.pose_left_pub = self.create_publisher(
            PoseStamped,
            f"/{self.servo_left_ns}/pose_target_cmds",
            10,
        )
        self.pose_right_pub = self.create_publisher(
            PoseStamped,
            f"/{self.servo_right_ns}/pose_target_cmds",
            10,
        )
        self.get_logger().info("Servo PoseStamped publishers initialized:")
        self.get_logger().info(f"  Left:  /{self.servo_left_ns}/pose_target_cmds")
        self.get_logger().info(f"  Right: /{self.servo_right_ns}/pose_target_cmds")

        # Subscribe to Servo status for monitoring
        self.servo_status_left_sub = self.create_subscription(
            ServoStatus,
            f"/{self.servo_left_ns}/status",
            lambda msg: self._servo_status_callback(msg, "left"),
            10,
        )
        self.servo_status_right_sub = self.create_subscription(
            ServoStatus,
            f"/{self.servo_right_ns}/status",
            lambda msg: self._servo_status_callback(msg, "right"),
            10,
        )

    def _init_servo_service_clients(self):
        """Create service clients to switch Servo to POSE command mode."""
        self.switch_left_client = self.create_client(
            ServoCommandType,
            f"/{self.servo_left_ns}/switch_command_type",
        )
        self.switch_right_client = self.create_client(
            ServoCommandType,
            f"/{self.servo_right_ns}/switch_command_type",
        )

        self.get_logger().info("Waiting for Servo switch_command_type services...")
        left_ready = self.switch_left_client.wait_for_service(timeout_sec=10.0)
        right_ready = self.switch_right_client.wait_for_service(timeout_sec=10.0)

        if not left_ready or not right_ready:
            self.get_logger().warn(
                "Servo services not available! Make sure to run: "
                "ros2 launch gr00t_bridge_multi_endpose_abs_rel_servo servo_dual_arm.launch.py"
            )
        else:
            self._switch_servo_to_pose_mode()

    def _switch_servo_to_pose_mode(self):
        """Switch both Servo nodes to POSE command mode (command_type=2)."""
        request = ServoCommandType.Request()
        request.command_type = self.SERVO_CMD_POSE

        future_left = self.switch_left_client.call_async(request)
        future_right = self.switch_right_client.call_async(request)

        future_left.add_done_callback(
            lambda f: self.get_logger().info(
                f"Servo LEFT switched to POSE mode: success={f.result().success}"
                if f.result() else "Servo LEFT switch failed"
            )
        )
        future_right.add_done_callback(
            lambda f: self.get_logger().info(
                f"Servo RIGHT switched to POSE mode: success={f.result().success}"
                if f.result() else "Servo RIGHT switch failed"
            )
        )

    def _init_gripper_publishers(self):
        """Gripper commands are published separately from Servo (arm-only)."""
        self.gripper_left_pub = self.create_publisher(
            JointTrajectory,
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            10,
        )
        self.gripper_right_pub = self.create_publisher(
            JointTrajectory,
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            10,
        )
        self.get_logger().info("Gripper publishers initialized (separate from Servo)")

    def _init_tf_broadcaster(self):
        """Initialize TF broadcaster for target poses."""
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.get_logger().info("TF broadcaster initialized for target poses")

    def _load_model(self):
        try:
            self.model = Gr00tEndPoseRelInferenceWrapper(
                checkpoint_path=self.checkpoint_path,
                embodiment_tag=self.embodiment_tag,
                device=self.device,
            )
            self.get_logger().info("GR00T end-pose relative action model loaded!")
        except Exception as e:
            self.get_logger().error(f"Failed to load GR00T model: {e}")
            raise

    def _print_startup_info(self):
        self.get_logger().info("=" * 70)
        self.get_logger().info("GR00T End-Pose Relative Action + MoveIt Servo Node")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"  Checkpoint: {self.checkpoint_path}")
        self.get_logger().info(f"  Embodiment: {self.embodiment_tag}")
        self.get_logger().info(f"  Device: {self.device}")
        self.get_logger().info(f"  Inference rate: {self.inference_rate} Hz")
        self.get_logger().info(f"  Dry run: {self.dry_run}")
        self.get_logger().info("-" * 70)
        self.get_logger().info("Output: MoveIt Servo PoseCommand (non-blocking)")
        self.get_logger().info("  Model → relative delta → absolute PoseStamped → Servo")
        self.get_logger().info("  Servo handles: IK + smoothing + singularity + collision")
        self.get_logger().info("-" * 70)
        self.get_logger().info(f"  Servo left ns:  {self.servo_left_ns}")
        self.get_logger().info(f"  Servo right ns: {self.servo_right_ns}")
        self.get_logger().info("=" * 70)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _image_callback(self, msg: CompressedImage, camera_key: str):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.latest_images[camera_key] = image
                self.latest_image_stamps[camera_key] = Time.from_msg(msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f"Error processing {camera_key} image: {e}")

    def _end_pose_left_callback(self, msg: PoseStamped):
        self.latest_end_pose_left = msg

    def _end_pose_right_callback(self, msg: PoseStamped):
        self.latest_end_pose_right = msg

    def _joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def _servo_status_callback(self, msg: ServoStatus, side: str):
        if side == "left":
            self._servo_status_left = msg
        else:
            self._servo_status_right = msg

    # =========================================================================
    # Helpers
    # =========================================================================

    def _extract_pose_array(self, pose_stamped: PoseStamped) -> np.ndarray:
        pose = pose_stamped.pose
        return np.array([
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ])

    def _extract_state(self) -> Dict[str, np.ndarray]:
        end_pose_left = self._extract_pose_array(self.latest_end_pose_left)
        end_pose_right = self._extract_pose_array(self.latest_end_pose_right)

        joint_dict = {
            name: pos
            for name, pos in zip(
                self.latest_joint_state.name, self.latest_joint_state.position
            )
        }
        gripper_l = np.array([joint_dict.get("gripper_l_joint1", 0.0)])
        gripper_r = np.array([joint_dict.get("gripper_r_joint1", 0.0)])
        head = np.array([
            joint_dict.get("head_joint1", 0.0),
            joint_dict.get("head_joint2", 0.0),
        ])
        return {
            "end_pose_left": end_pose_left,
            "gripper_l": gripper_l,
            "end_pose_right": end_pose_right,
            "gripper_r": gripper_r,
            "head": head,
        }

    def _pose_array_to_pose_msg(self, pose_array: np.ndarray) -> Pose:
        pose = Pose()
        pose.position.x = float(pose_array[0])
        pose.position.y = float(pose_array[1])
        pose.position.z = float(pose_array[2])
        pose.orientation.x = float(pose_array[3])
        pose.orientation.y = float(pose_array[4])
        pose.orientation.z = float(pose_array[5])
        pose.orientation.w = float(pose_array[6])
        return pose

    def _check_data_ready(self) -> bool:
        missing_cameras = [k for k, v in self.latest_images.items() if v is None]
        has_end_pose_left = self.latest_end_pose_left is not None
        has_end_pose_right = self.latest_end_pose_right is not None
        has_joint_state = self.latest_joint_state is not None

        current_time = time.time()
        if current_time - self.last_status_print >= self.status_print_interval:
            self._print_data_status(
                missing_cameras, has_end_pose_left, has_end_pose_right, has_joint_state
            )
            self.last_status_print = current_time

        return (
            len(missing_cameras) == 0
            and has_end_pose_left
            and has_end_pose_right
            and has_joint_state
        )

    def _print_data_status(
        self, missing_cameras, has_end_pose_left, has_end_pose_right, has_joint_state
    ):
        self.get_logger().info("-" * 50)
        self.get_logger().info("Data Status:")
        for camera_key in self.latest_images:
            img = self.latest_images[camera_key]
            status = "OK" if img is not None else "MISSING"
            shape = f" {img.shape}" if img is not None else ""
            self.get_logger().info(f"  {camera_key}: {status}{shape}")
        self.get_logger().info(f"  end_pose_left: {'OK' if has_end_pose_left else 'MISSING'}")
        self.get_logger().info(f"  end_pose_right: {'OK' if has_end_pose_right else 'MISSING'}")
        self.get_logger().info(f"  joint_states: {'OK' if has_joint_state else 'MISSING'}")

    # =========================================================================
    # Servo publish helpers
    # =========================================================================

    def _publish_pose_to_servo(self, pose_array: np.ndarray, side: str):
        """Create PoseStamped from pose_array and publish to Servo.
        header.stamp = 0 so Servo does not treat the command as 'TF old' when clocks differ (e.g. cross-container).
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.header.frame_id = self.latest_end_pose_left.header.frame_id if side == "left" else self.latest_end_pose_right.header.frame_id
        msg.pose = self._pose_array_to_pose_msg(pose_array)

        if side == "left":
            self.pose_left_pub.publish(msg)
        else:
            self.pose_right_pub.publish(msg)

    def _publish_gripper_command(self, gripper_value: np.ndarray, side: str):
        """Publish gripper-only JointTrajectory (separate from Servo arm control).
        Use current time for header.stamp so the controller does not reject as 'trajectory ends in the past'.
        """
        msg = JointTrajectory()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""

        if side == "left":
            msg.joint_names = self.joint_config.LEFT_GRIPPER_JOINTS
        else:
            msg.joint_names = self.joint_config.RIGHT_GRIPPER_JOINTS

        point = JointTrajectoryPoint()
        point.positions = gripper_value.flatten().tolist()
        duration_ns = int(self.gripper_step_duration_sec * 1_000_000_000)
        point.time_from_start = Duration(sec=0, nanosec=duration_ns)
        msg.points = [point]

        if side == "left":
            self.gripper_left_pub.publish(msg)
        else:
            self.gripper_right_pub.publish(msg)

    def _broadcast_target_pose_tf(self, pose_array: np.ndarray, side: str, parent_frame_id: str):
        """Broadcast target pose as TF transform for visualization in RViz."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame_id
        t.child_frame_id = f"target_pose_{side}"

        t.transform.translation.x = float(pose_array[0])
        t.transform.translation.y = float(pose_array[1])
        t.transform.translation.z = float(pose_array[2])
        t.transform.rotation.x = float(pose_array[3])
        t.transform.rotation.y = float(pose_array[4])
        t.transform.rotation.z = float(pose_array[5])
        t.transform.rotation.w = float(pose_array[6])

        self.tf_broadcaster.sendTransform(t)

    # =========================================================================
    # Step-by-step control
    # =========================================================================

    def _start_step_input_thread(self):
        """Background thread that waits for Enter key to approve each inference step."""
        def _input_loop():
            while rclpy.ok():
                try:
                    input(
                        "\n[STEP-BY-STEP] Press Enter to execute next inference "
                        "(or 'q' + Enter to switch to continuous)... "
                    )
                except EOFError:
                    break
                with self._step_lock:
                    if not self._step_approved:
                        self._step_approved = True

        t = threading.Thread(target=_input_loop, daemon=True)
        t.start()

    # =========================================================================
    # Inference loop
    # =========================================================================

    def _inference_callback(self):
        if not self._check_data_ready():
            return

        self.step_count += 1

        if self.step_count % self.action_horizon != 0:
            if self.action_buffer is not None:
                self._execute_action_from_buffer()
            return

        # Step-by-step gate: skip this inference cycle until user approves
        if self.step_by_step:
            with self._step_lock:
                if not self._step_approved:
                    return
                self._step_approved = False

        try:
            inference_start = time.time()

            state = self._extract_state()
            observation = prepare_observation_endpose_rel(
                self.latest_images.copy(),
                state,
                self.task_instruction,
                self.model,
            )

            action_dict = self.model.policy.get_action(observation)
            action = self.model._parse_action_output(action_dict)

            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            self.inference_count += 1

            self.action_buffer = action
            self.action_buffer_index = 0

            self.tracking_pose_left = self._extract_pose_array(
                self.latest_end_pose_left
            ).copy()
            self.tracking_pose_right = self._extract_pose_array(
                self.latest_end_pose_right
            ).copy()
            
            self._print_action(action, inference_time)

            num_execute = min(self.action_horizon_execute, 16)
            for i in range(num_execute):
                self._execute_action_from_buffer()
                if self.action_horizon_step_delay > 0 and i < num_execute - 1:
                    time.sleep(self.action_horizon_step_delay)

            if self.step_by_step:
                self.get_logger().info(
                    f"[STEP] Executed {num_execute} actions. "
                    "Press Enter for next inference..."
                )

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _execute_action_from_buffer(self):
        """Pop one action from the buffer, convert to absolute pose, publish to Servo.

        1-frame shift fix (see analysis_gr00t_endpose_rel_servo_inference.md §11.3):
        Data stores action[i] = pose[i] - pose[i-1] (delta that reached frame i).
        For control we need delta_curr_to_next = pose[i+1] - pose[i], which is
        stored at index i+1. So we use idx = action_buffer_index + 1.
        """
        if self.action_buffer is None:
            return
        if self.tracking_pose_left is None or self.tracking_pose_right is None:
            self.get_logger().warn("Tracking poses not initialized, skipping")
            return

        actual_chunk_size = (
            self.action_buffer.rel_end_pose_left.shape[0]
            if len(self.action_buffer.rel_end_pose_left.shape) > 1
            else 1
        )
        # Use idx+1 so we apply delta_curr_to_next (action[i+1] = pose[i+1]-pose[i])
        idx = self.action_buffer_index + 1
        if idx >= actual_chunk_size:
            return

        delta_left = self.action_buffer.rel_end_pose_left[idx].flatten()
        gripper_l = self.action_buffer.gripper_l[idx].flatten()
        delta_right = self.action_buffer.rel_end_pose_right[idx].flatten()
        gripper_r = self.action_buffer.gripper_r[idx].flatten()

        target_pose_left = apply_relative_action(self.tracking_pose_left, delta_left)
        target_pose_right = apply_relative_action(self.tracking_pose_right, delta_right)

        self.tracking_pose_left = target_pose_left.copy()
        self.tracking_pose_right = target_pose_right.copy()

        # Broadcast TF for target poses (for RViz visualization)
        if self.latest_end_pose_left is not None and self.latest_end_pose_right is not None:
            self._broadcast_target_pose_tf(
                target_pose_left, "left", self.latest_end_pose_left.header.frame_id
            )
            self._broadcast_target_pose_tf(
                target_pose_right, "right", self.latest_end_pose_right.header.frame_id
            )

        if not self.dry_run:
            self._publish_pose_to_servo(target_pose_left, "left")
            self._publish_pose_to_servo(target_pose_right, "right")
            self._publish_gripper_command(gripper_l, "left")
            self._publish_gripper_command(gripper_r, "right")

        self.action_buffer_index += 1

    # =========================================================================
    # Logging
    # =========================================================================

    def _print_action(self, action: EndPoseRelActionOutput, inference_time: float):
        first = action.get_first_timestep()
        concat = action.get_concatenated_action()
        avg_inference = np.mean(self.inference_times) if self.inference_times else 0.0

        self.get_logger().info("-" * 60)
        self.get_logger().info(
            f"[Inference #{self.inference_count}] Step: {self.step_count}"
        )
        self.get_logger().info(
            f"  Inference: {inference_time*1000:.1f}ms (avg: {avg_inference*1000:.1f}ms)"
        )
        self.get_logger().info("  Output: MoveIt Servo PoseCommand (non-blocking)")

        if self.latest_end_pose_left is not None:
            cur = self._extract_pose_array(self.latest_end_pose_left)
            self.get_logger().info(
                f"  Current left:  pos=[{cur[0]:.4f}, {cur[1]:.4f}, {cur[2]:.4f}]"
            )
        if self.latest_end_pose_right is not None:
            cur = self._extract_pose_array(self.latest_end_pose_right)
            self.get_logger().info(
                f"  Current right: pos=[{cur[0]:.4f}, {cur[1]:.4f}, {cur[2]:.4f}]"
            )

        self.get_logger().info("  Predicted RELATIVE Deltas (first timestep):")
        dl = first["rel_end_pose_left"]
        dr = first["rel_end_pose_right"]
        self.get_logger().info(
            f"    Left delta:  [{', '.join(f'{v:.6f}' for v in dl[:3])}] (dpos)"
        )
        self.get_logger().info(
            f"    Left grip:   [{first['gripper_l'][0]:.4f}] (absolute)"
        )
        self.get_logger().info(
            f"    Right delta: [{', '.join(f'{v:.6f}' for v in dr[:3])}] (dpos)"
        )
        self.get_logger().info(
            f"    Right grip:  [{first['gripper_r'][0]:.4f}] (absolute)"
        )
        self.get_logger().info(
            f"  Concatenated: {concat[:4]}... (len={len(concat)})"
        )


def main(args=None):
    rclpy.init(args=args)
    try:
        node = Gr00tEndPoseRelServoNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
