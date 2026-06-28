#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# obstacle_node (/obstacle_markers) -> MoveIt planning scene + cuRobo HTTP /plan -> ExecuteTrajectory
#
# Service/action futures from subscription callbacks must not call spin_until_future_complete
# while MultiThreadedExecutor is spinning (Jazzy: action client wait-set corruption / crash).
# Poll futures with short sleeps instead; see pathplanning_node.py.
#
# Prerequisites (typical sim stack):
#   1. ros2 launch ffw_bringup ffw_sg2_follower_ai_gazebo.launch.py
#   2. ros2 launch ffw_moveit_config moveit.launch.py use_sim:=true
#   3. ros2 run ffw_moveit_config obstacle_node.py --ros-args -p use_sim_time:=true
#   4. cuRobo Docker: python curobo_plan_server.py  (bind 0.0.0.0:8000, publish -p 8000:8000)
#   5. FFW Docker (this node):
#        export CUROBO_URL=http://<curobo_host>:8000/plan
#        ros2 launch ffw_moveit_config curobo_planning.launch.py use_sim:=true
#
# cuRobo and FFW in separate containers: do NOT use 127.0.0.1 — use host IP, docker network
# name (e.g. http://curobo:8000/plan), or host.docker.internal with --add-host=host-gateway.
#
# RViz: obstacles appear via planning scene; target pose on /curobo_target_markers (MarkerArray)
#   and /curobo_target_pose (PoseStamped). Planned path on display_planned_path_topic.
#   With confirm_before_execute:=true, check RViz then Enter in the launch terminal, or:
#     ros2 topic pub --once /curobo/confirm_execute std_msgs/msg/Empty {}
# Return to robot initial pose via cuRobo joint plan (same confirm flow):
#     ros2 topic pub --once /curobo/plan_home std_msgs/msg/Empty {}
#     ros2 topic pub --once /curobo/confirm_execute std_msgs/msg/Empty {}
# Return to initial pose via direct FollowJointTrajectory (no cuRobo / no confirm):
#     ros2 topic pub --once /curobo/joint_home std_msgs/msg/Empty {}
# Manual goals (geometry_msgs/PoseStamped):
#   left:  goal_pose_topic (default /target_pose)
#   right: goal_pose_r_topic (default /target_pose_r) when planning_mode is right or dual
# Dual arm: planning_mode:=dual plans ffw_sg2_arm_l.yml + ffw_sg2_arm_r.yml, merges trajectories.

from __future__ import annotations

import copy
import math
import os
import random
import secrets
import sys
from pathlib import Path

# Sibling import when installed under lib/ffw_moveit_config/
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))
import threading
import time
from typing import Dict, List, Optional

import yaml
import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from std_msgs.msg import ColorRGBA, Empty, Header
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
  BoundingVolume,
  Constraints,
  DisplayTrajectory,
  MoveItErrorCodes,
  MotionPlanRequest,
  OrientationConstraint,
  PlanningOptions,
  PositionConstraint,
  RobotState,
  RobotTrajectory,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

try:
  import requests
except ImportError as e:
  print('Install requests: pip install requests', file=sys.stderr)
  raise SystemExit(1) from e

from curobo_obstacle_utils import (
  DEFAULT_PLANNING_FRAME,
  TARGET_BACK_WALL_ID,
  append_ee_axes_markers,
  curobo_cuboids_to_marker_array,
  curobo_obstacle_debug_delete_all,
  filter_curobo_obstacles_for_goal,
  marker_array_signature,
  markers_to_planning_cuboids,
  planning_scene_from_cuboids,
  quat_toward_robot_y_up_arm,
)

DEFAULT_EE_LINK_L = 'end_effector_l_link'
DEFAULT_EE_LINK_R = 'end_effector_r_link'
ARM_L_JOINTS = [
  'arm_l_joint1',
  'arm_l_joint2',
  'arm_l_joint3',
  'arm_l_joint4',
  'arm_l_joint5',
  'arm_l_joint6',
  'arm_l_joint7',
]
ARM_R_JOINTS = [f'arm_r_joint{i}' for i in range(1, 8)]

# ExecuteTrajectory joint groups (lift owned by left plan in dual mode).
TRAJECTORY_JOINTS_L = ['lift_joint', *ARM_L_JOINTS]
TRAJECTORY_JOINTS_R = list(ARM_R_JOINTS)
TRAJECTORY_JOINTS_R_LIFT = ['lift_joint', *ARM_R_JOINTS]
# Dual pose planning: do not move lift (right IK is planned with lift fixed at start).
TRAJECTORY_JOINTS_L_DUAL = list(ARM_L_JOINTS)
TRAJECTORY_JOINTS_DUAL = ['lift_joint', *ARM_L_JOINTS, *ARM_R_JOINTS]
# Back-compat alias
TRAJECTORY_JOINTS = TRAJECTORY_JOINTS_L
DEFAULT_CUROBO_ROBOT_FILE_L = 'ffw_sg2_arm_l.yml'
DEFAULT_CUROBO_ROBOT_FILE_R = 'ffw_sg2_arm_r.yml'

_MOVEIT_ERROR_HINTS = {
  -4: 'CONTROL_FAILED (controller / start mismatch / overlapping goals)',
  -3: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
  -10: 'START_STATE_IN_COLLISION',
  -12: 'GOAL_IN_COLLISION',
}

# Match cuRobo ffw_sg2_arm_l.yml cspace; used for start-state FK/collision.
# Wheels omitted (not on /joint_states in sim); cuRobo server fills them from YAML defaults.
CUROBO_STATE_JOINTS = [
  'lift_joint',
  *ARM_L_JOINTS,
  'head_joint1',
  'head_joint2',
  *[f'arm_r_joint{i}' for i in range(1, 8)],
  'gripper_l_joint1',
  'gripper_r_joint1',
]

# Nominal collision-free pose from ffw_sg2_arm_l.yml default_joint_position.
CUROBO_PASSIVE_JOINT_DEFAULTS: Dict[str, float] = {
  'head_joint1': 0.23170000314712524,
  'head_joint2': 0.0,
  'arm_r_joint1': -1.5700000524520874,
  'arm_r_joint2': 0.0,
  'arm_r_joint3': -0.9287499785423279,
  'arm_r_joint4': 0.0,
  'arm_r_joint5': 0.0,
  'arm_r_joint6': 0.11984997987747192,
  'arm_r_joint7': 0.550000011920929,
  'gripper_l_joint1': 0.550000011920929,
  'gripper_r_joint1': 0.0,
}

# ffw_sg2_follower_initial_positions.yaml + cuRobo ffw_sg2_arm_l.yml default_joint_position.
ROBOT_INITIAL_JOINTS: Dict[str, float] = {
  'lift_joint': -0.25,
  'arm_l_joint1': 1.5700000524520874,
  'arm_l_joint2': 0.0,
  'arm_l_joint3': -0.9287499785423279,
  'arm_l_joint4': 0.0,
  'arm_l_joint5': 0.0,
  'arm_l_joint6': -0.11984997987747192,
  'arm_l_joint7': 0.550000011920929,
  'arm_r_joint1': -1.5700000524520874,
  'arm_r_joint2': 0.0,
  'arm_r_joint3': -0.9287499785423279,
  'arm_r_joint4': 0.0,
  'arm_r_joint5': 0.0,
  'arm_r_joint6': 0.11984997987747192,
  'arm_r_joint7': 0.550000011920929,
}

# Fallback when robot_joint_home.yaml is missing from install (e.g. before colcon build).
DEFAULT_JOINT_HOME_GROUPS: List[dict] = [
  {
    'name': 'arm_l',
    'joint_names': [
      'arm_l_joint1', 'arm_l_joint2', 'arm_l_joint3', 'arm_l_joint4',
      'arm_l_joint5', 'arm_l_joint6', 'arm_l_joint7', 'gripper_l_joint1',
    ],
    'positions': [
      -0.085627, 0.042807, 0.162697, -0.613939,
      -0.151097, -0.247222, -0.350348, 0.175741,
    ],
    'action_topic': '/arm_l_controller/follow_joint_trajectory',
    'duration': 5.0,
  },
  {
    'name': 'arm_r',
    'joint_names': [
      'arm_r_joint1', 'arm_r_joint2', 'arm_r_joint3', 'arm_r_joint4',
      'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7', 'gripper_r_joint1',
    ],
    'positions': [
      -0.039092, -0.015699, -0.296765, -0.713672,
      0.178756, -0.054947, 0.445770, 0.156432,
    ],
    'action_topic': '/arm_r_controller/follow_joint_trajectory',
    'duration': 5.0,
  },
  {
    'name': 'head',
    'joint_names': ['head_joint1', 'head_joint2'],
    'positions': [0.66, -0.11],
    'action_topic': '/head_controller/follow_joint_trajectory',
    'duration': 5.0,
  },
  {
    'name': 'lift',
    'joint_names': ['lift_joint'],
    'positions': [0.0],
    'action_topic': '/lift_controller/follow_joint_trajectory',
    'duration': 10.0,
  },
  {
    'name': 'swerve',
    'joint_names': ['left_wheel_steer', 'right_wheel_steer', 'rear_wheel_steer'],
    'positions': [0.0, 0.0, 0.0],
    'action_topic': '/swerve_steering_initial_position_controller/follow_joint_trajectory',
    'duration': 10.0,
  },
]

TARGET_MARKER_NS_L = 'curobo_target_l'
TARGET_MARKER_NS_R = 'curobo_target_r'
TARGET_CONTACT_NS_L = 'curobo_contact_l'
TARGET_CONTACT_NS_R = 'curobo_contact_r'
TARGET_MARKER_ID_L_AXES = 0
TARGET_MARKER_ID_L_SPHERE = 1
TARGET_MARKER_ID_R_AXES = 2
TARGET_MARKER_ID_R_SPHERE = 3
TARGET_MARKER_ID_L_CONTACT = 0
TARGET_MARKER_ID_L_CONTACT_LINE = 1
TARGET_MARKER_ID_R_CONTACT = 0
TARGET_MARKER_ID_R_CONTACT_LINE = 1
GOAL_POSITION_EPS_M = 1e-4


class CuRoboPathPlanningNode(Node):
  def __init__(self) -> None:
    super().__init__('curobo_pathplanning_node')
    # Planning/obstacles block inside this group; confirm must use a separate group so
    # MultiThreadedExecutor can run _on_execute_confirm while waiting on event.wait().
    self._cb_group = ReentrantCallbackGroup()
    self._confirm_cb_group = MutuallyExclusiveCallbackGroup()

    self.declare_parameter('obstacle_topic', '/obstacle_markers')
    self.declare_parameter('apply_planning_scene_service', '')
    self.declare_parameter('move_group_namespace', 'root')
    _default_curobo_url = os.environ.get('CUROBO_URL', 'http://127.0.0.1:8000/plan')
    self.declare_parameter('curobo_url', _default_curobo_url)
    self.declare_parameter('curobo_health_url', '')
    self.declare_parameter('joint_state_topic', '/joint_states')
    self.declare_parameter('planning_mode', 'dual')
    # plan_merged: plan left+lift, plan right at left goal, merge, one confirm, one execute
    # parallel: plan both from same start (lift fixed), merge, one execute
    # left_then_right: plan+execute left+lift, then plan+execute right (two executes)
    # lift_midpoint: probe L+lift & R+lift → shared lift → replan L/R @ shared lift
    # right_lift_first: alias for lift_midpoint (legacy name)
    self.declare_parameter('dual_sequence', 'plan_merged')
    self.declare_parameter('dual_sequence_settle_sec', 1.0)
    self.declare_parameter('dual_sequence_joint_wait_sec', 5.0)
    # false: plan+execute right automatically after left+lift (no 2nd confirm)
    self.declare_parameter('dual_require_right_confirm', False)
    # true: left dual plan includes lift_joint so EE can reach target z; false: lift fixed at start.
    self.declare_parameter('dual_plan_lift', True)
    self.declare_parameter(
      'require_back_wall_for_dual',
      True,
    )
    self.declare_parameter('plan_debounce_sec', 0.35)
    self.declare_parameter('goal_pose_topic', '/target_pose')
    self.declare_parameter('goal_pose_r_topic', '/target_pose_r')
    self.declare_parameter('goal_pose_contact_topic', '/target_pose_contact')
    self.declare_parameter('goal_pose_contact_r_topic', '/target_pose_contact_r')
    self.declare_parameter('enable_final_tool_approach', True)
    self.declare_parameter('final_approach_use_move_group', True)
    self.declare_parameter('final_approach_tool_m', 0.08)
    self.declare_parameter('final_approach_tool_axis', 'z')
    self.declare_parameter('final_approach_steps', 4)
    self.declare_parameter('final_approach_max_step', 0.005)
    self.declare_parameter('final_approach_velocity_scale', 0.15)
    self.declare_parameter('final_approach_goal_tolerance_m', 0.008)
    self.declare_parameter('final_approach_planning_time', 10.0)
    self.declare_parameter('final_approach_planning_attempts', 8)
    # Max EE→contact distance to attempt STAGE 2 (cuRobo may stop short of pre-grasp).
    self.declare_parameter('final_approach_max_distance_m', 0.35)
    self.declare_parameter('final_approach_settle_sec', 1.0)
    self.declare_parameter('final_approach_tf_timeout_sec', 0.5)
    self.declare_parameter('curobo_robot_file_l', DEFAULT_CUROBO_ROBOT_FILE_L)
    self.declare_parameter('curobo_robot_file_r', DEFAULT_CUROBO_ROBOT_FILE_R)
    self.declare_parameter('planning_frame', DEFAULT_PLANNING_FRAME)
    self.declare_parameter('auto_plan', True)
    self.declare_parameter('min_plan_interval_sec', 0.5)
    self.declare_parameter('goal_pose_epsilon_m', GOAL_POSITION_EPS_M)
    self.declare_parameter('scene_settle_sec', 0.05)
    self.declare_parameter('max_attempts', 15)
    self.declare_parameter('goal_seed', -1)
    self.declare_parameter('goal_pose_x_min', 0.38)
    self.declare_parameter('goal_pose_x_max', 0.52)
    self.declare_parameter('goal_pose_y_min', 0.15)
    self.declare_parameter('goal_pose_y_max', 0.35)
    self.declare_parameter('goal_pose_z_min', 1.15)
    self.declare_parameter('goal_pose_z_max', 1.42)
    self.declare_parameter('robot_base_x', 0.0)
    self.declare_parameter('robot_base_y', 0.0)
    # free: position only (EE rotation chosen by IK/planner)
    # from_message: use orientation from /target_pose PoseStamped
    # toward_robot: EE +X→base, +Y↑ (legacy default hint, soft-tracked on server)
    # fixed: enforce target_pose quaternion fully
    self.declare_parameter('goal_orientation_mode', 'toward_robot')
    self.declare_parameter('orientation_weight', 0.2)
    self.declare_parameter('home', False)
    # false: wait for first /target_pose before planning; true: legacy random goal at startup
    self.declare_parameter('seed_goal_on_start', False)
    self.declare_parameter('sync_moveit_scene', True)
    self.declare_parameter('send_obstacles_to_curobo', True)
    self.declare_parameter('curobo_obstacle_dim_scale', 1.0)
    self.declare_parameter(
      'curobo_goal_obstacle_clearance_m',
      0.05,
    )
    self.declare_parameter('sphere_obstacles_as_boxes_in_scene', True)
    self.declare_parameter('publish_curobo_obstacle_debug', False)
    self.declare_parameter('curobo_obstacle_debug_topic', '/curobo_obstacle_cuboids')
    self.declare_parameter('use_passive_defaults_for_start', True)
    self.declare_parameter('skip_start_feasibility_check', False)
    self.declare_parameter('publish_target_visualization', True)
    self.declare_parameter('target_marker_topic', '/curobo_target_markers')
    self.declare_parameter('target_pose_viz_topic', '/curobo_target_pose')
    self.declare_parameter('target_contact_pose_viz_topic', '/curobo_target_pose_contact')
    self.declare_parameter('target_contact_pose_viz_r_topic', '/curobo_target_pose_contact_r')
    self.declare_parameter('target_marker_publish_period_sec', 1.0)
    self.declare_parameter('target_arrow_length', 0.12)
    self.declare_parameter('target_show_ee_axes', True)
    # free plan mode still draws toward_robot axes when true (plan uses position only).
    self.declare_parameter('viz_toward_robot_in_free_mode', True)
    self.declare_parameter('right_arm_roll_rad', math.pi)
    self.declare_parameter('target_sphere_diameter', 0.06)
    # Fraction in (0,1] OR percent in (1,100]: 20 -> 20%%, 0.2 -> 20%%.
    self.declare_parameter('velocity_scale', 0.2)
    self.declare_parameter('min_trajectory_duration_sec', 8.0)
    self.declare_parameter('confirm_before_execute', True)
    self.declare_parameter('execute_confirm_topic', '/curobo/confirm_execute')
    self.declare_parameter('plan_home_topic', '/curobo/plan_home')
    self.declare_parameter('joint_home_topic', '/curobo/joint_home')
    self.declare_parameter('joint_home_config', '')
    self.declare_parameter('display_planned_path_topic', '/display_planned_path')
    self.declare_parameter('robot_model_id', 'ffw')
    self.declare_parameter('display_path_publish_count', 3)
    self.declare_parameter('wait_for_joint_states_sec', 0.0)
    self.declare_parameter('require_joint_states_at_startup', False)
    self.declare_parameter('joint_state_best_effort_qos', True)
    self.declare_parameter('joint_state_stale_sec', 0.5)
    self.declare_parameter('joint_state_stamp_tolerance_sec', 2.0)

    self._curobo_url = str(self.get_parameter('curobo_url').value).strip()
    if self._curobo_url.endswith('/plan'):
      self._curobo_plan_joint_url = self._curobo_url[:-5] + '/plan_joint'
    else:
      self._curobo_plan_joint_url = self._curobo_url.rstrip('/') + '/plan_joint'
    health_param = str(self.get_parameter('curobo_health_url').value).strip()
    if health_param:
      self._health_url = health_param
    elif self._curobo_url.endswith('/plan'):
      self._health_url = self._curobo_url[:-5] + '/health'
    else:
      self._health_url = self._curobo_url.rstrip('/') + '/health'
    self.get_logger().info(f'cuRobo plan URL: {self._curobo_url}')
    self.get_logger().info(f'cuRobo health URL: {self._health_url}')
    self._planning_frame = str(self.get_parameter('planning_frame').value)
    self._planning_mode = str(self.get_parameter('planning_mode').value).strip().lower()
    if self._planning_mode not in ('left', 'right', 'dual'):
      self.get_logger().warn(
        f'Unknown planning_mode={self._planning_mode!r}; using left'
      )
      self._planning_mode = 'left'
    self._curobo_robot_file_l = str(self.get_parameter('curobo_robot_file_l').value).strip()
    self._curobo_robot_file_r = str(self.get_parameter('curobo_robot_file_r').value).strip()
    self.get_logger().info(
      f'goal_orientation_mode={self._goal_orientation_mode()} '
      f'(cuRobo orientation_mode={self._server_orientation_mode()}, '
      f'weight={float(self.get_parameter("orientation_weight").value)})'
    )
    self._arm_joints = self._required_arm_joints()
    self._trajectory_joints = self._trajectory_joints_for_mode()
    self._curobo_state_joints = list(CUROBO_STATE_JOINTS)
    self._latest_joint_state: Optional[JointState] = None
    self._latest_joint_state_mono: float = -1e10
    self._last_joint_state_stamp_warn_mono: float = -1e10
    self._last_obstacle_cuboids: List[dict] = []
    self._last_obstacle_msg: Optional[MarkerArray] = None
    self._moveit_scene_arm_timer = None
    self._last_plan_mono = -1e10
    self._last_marker_sig: tuple | None = None
    self._obstacle_lock = threading.Lock()
    self._plan_lock = threading.Lock()
    self._joint_home_lock = threading.Lock()
    self._execute_confirm_event = threading.Event()
    self._awaiting_confirm = False
    self._confirm_session_id = 0
    self._goal_l_ready = False
    self._goal_r_ready = False
    self._cached_goal_l_xyz: Optional[tuple[float, float, float]] = None
    self._cached_goal_l_quat = None
    self._cached_goal_r_xyz: Optional[tuple[float, float, float]] = None
    self._cached_goal_r_quat = None
    self._cached_contact_l_xyz: Optional[tuple[float, float, float]] = None
    self._cached_contact_l_quat = None
    self._cached_contact_r_xyz: Optional[tuple[float, float, float]] = None
    self._cached_contact_r_quat = None
    self._plan_debounce_timer = None
    self._pending_plan_reason: Optional[str] = None
    self._pending_plan_fire_mono: float = 0.0
    self._planning_armed = False
    self._readiness_poll_timer = None
    self._readiness_was_ready = False
    self._tf_buffer = tf2_ros.Buffer()
    self._tf_listener = tf2_ros.TransformListener(
      self._tf_buffer, self, spin_thread=True
    )

    srv = self._resolve_apply_scene_service()
    self._apply_scene_cli = self.create_client(
      ApplyPlanningScene, srv, callback_group=self._cb_group
    )
    self.get_logger().info(f'ApplyPlanningScene service: {srv}')
    cartesian_srv = self._resolve_cartesian_path_service()
    self._cartesian_path_cli = self.create_client(
      GetCartesianPath, cartesian_srv, callback_group=self._cb_group
    )
    self.get_logger().info(f'GetCartesianPath service: {cartesian_srv}')

    raw = str(self.get_parameter('move_group_namespace').value).strip()
    low = raw.lower()
    if low in ('', 'root', '/'):
      exec_name = '/execute_trajectory'
      move_action_name = '/move_action'
    else:
      ns = raw if raw.startswith('/') else '/' + raw
      ns = ns.rstrip('/') or '/move_group'
      exec_name = f'{ns}/execute_trajectory'
      move_action_name = f'{ns}/move_action'

    self._execute = ActionClient(
      self, ExecuteTrajectory, exec_name, callback_group=self._cb_group
    )
    self._move_group = ActionClient(
      self, MoveGroup, move_action_name, callback_group=self._cb_group
    )
    self.get_logger().info(f'ExecuteTrajectory action: {exec_name}')
    self.get_logger().info(f'MoveGroup action: {move_action_name}')

    self._qos = QoSProfile(
      depth=10,
      durability=DurabilityPolicy.VOLATILE,
      reliability=ReliabilityPolicy.RELIABLE,
    )
    # Match perception_click_planning_node latched target/obstacle publishers.
    self._planning_input_qos = QoSProfile(
      depth=10,
      durability=DurabilityPolicy.TRANSIENT_LOCAL,
      reliability=ReliabilityPolicy.RELIABLE,
    )
    if bool(self.get_parameter('joint_state_best_effort_qos').value):
      self._js_qos = qos_profile_sensor_data
      js_qos_label = 'BEST_EFFORT (sensor_data)'
    else:
      self._js_qos = QoSProfile(
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
        reliability=ReliabilityPolicy.RELIABLE,
      )
      js_qos_label = 'RELIABLE, VOLATILE'

    js_topic = str(self.get_parameter('joint_state_topic').value)
    self.create_subscription(
      JointState,
      js_topic,
      self._on_joint_states,
      self._js_qos,
      callback_group=self._cb_group,
    )
    self.get_logger().info(f'Subscribed to {js_topic} ({js_qos_label})')

    display_topic = str(self.get_parameter('display_planned_path_topic').value).strip()
    self._display_path_pub = self.create_publisher(
      DisplayTrajectory, display_topic, self._qos
    )
    self.get_logger().info(f'Planned path RViz topic: {display_topic}')

    confirm_topic = str(self.get_parameter('execute_confirm_topic').value).strip()
    if confirm_topic:
      self.create_subscription(
        Empty,
        confirm_topic,
        self._on_execute_confirm,
        self._qos,
        callback_group=self._confirm_cb_group,
      )
      self.get_logger().info(f'Execute confirm topic: {confirm_topic}')

    plan_home_topic = str(self.get_parameter('plan_home_topic').value).strip()
    if plan_home_topic:
      self.create_subscription(
        Empty,
        plan_home_topic,
        self._on_plan_home,
        self._qos,
        callback_group=self._confirm_cb_group,
      )
      self.get_logger().info(
        f'Plan home topic: {plan_home_topic} '
        f'(joint goal via {self._curobo_plan_joint_url})'
      )

    joint_home_topic = str(self.get_parameter('joint_home_topic').value).strip()
    if joint_home_topic:
      self.create_subscription(
        Empty,
        joint_home_topic,
        self._on_joint_home,
        self._qos,
        callback_group=self._confirm_cb_group,
      )
      groups = self._load_joint_home_groups()
      self.get_logger().info(
        f'Joint home topic: {joint_home_topic} '
        f'({len(groups)} controller groups, direct FollowJointTrajectory)'
      )

    vscale = self._resolve_velocity_scale()
    confirm = bool(self.get_parameter('confirm_before_execute').value)
    self.get_logger().info(
      f'effective velocity={vscale * 100:.0f}% (param={self.get_parameter("velocity_scale").value}), '
      f'min_trajectory_duration_sec={float(self.get_parameter("min_trajectory_duration_sec").value):.1f}, '
      f'confirm_before_execute={confirm}'
    )

    self.create_obstacle_subscription()

    goal_topic = str(self.get_parameter('goal_pose_topic').value).strip()
    if goal_topic:
      self.create_subscription(
        PoseStamped,
        goal_topic,
        self._on_goal_pose_l,
        self._planning_input_qos,
        callback_group=self._cb_group,
      )
      self.get_logger().info(f'Left-arm goals on {goal_topic}')

    if self._planning_mode in ('right', 'dual'):
      goal_r_topic = str(self.get_parameter('goal_pose_r_topic').value).strip()
      if goal_r_topic:
        self.create_subscription(
          PoseStamped,
          goal_r_topic,
          self._on_goal_pose_r,
          self._planning_input_qos,
          callback_group=self._cb_group,
        )
        self.get_logger().info(f'Right-arm goals on {goal_r_topic}')

    contact_topic = str(self.get_parameter('goal_pose_contact_topic').value).strip()
    if contact_topic and self._planning_mode in ('left', 'dual'):
      self.create_subscription(
        PoseStamped,
        contact_topic,
        self._on_goal_contact_l,
        self._planning_input_qos,
        callback_group=self._cb_group,
      )
      self.get_logger().info(f'Left contact on {contact_topic}')
    if self._planning_mode in ('right', 'dual'):
      contact_r_topic = str(
        self.get_parameter('goal_pose_contact_r_topic').value
      ).strip()
      if contact_r_topic:
        self.create_subscription(
          PoseStamped,
          contact_r_topic,
          self._on_goal_contact_r,
          self._planning_input_qos,
          callback_group=self._cb_group,
        )
        self.get_logger().info(f'Right contact on {contact_r_topic}')

    if bool(self.get_parameter('enable_final_tool_approach').value):
      max_dist = float(self.get_parameter('final_approach_max_distance_m').value)
      use_mg = bool(self.get_parameter('final_approach_use_move_group').value)
      settle = float(self.get_parameter('final_approach_settle_sec').value)
      self.get_logger().info(
        '[MoveIt] STAGE 2: plan to clicked contact pose from current EE '
        f'(MoveGroup={"yes" if use_mg else "Cartesian only"}, '
        f'max EE→contact={max_dist:.2f}m, pause {settle:.1f}s after cuRobo)'
      )

    dual_seq = self._dual_sequence_mode() if self._planning_mode == 'dual' else 'n/a'
    self.get_logger().info(
      f'planning_mode={self._planning_mode}, dual_sequence={dual_seq}, '
      f'robot_files=({self._curobo_robot_file_l}, {self._curobo_robot_file_r})'
    )
    if self._planning_mode == 'dual':
      lift_note = (
        'L+lift'
        if self._dual_plan_lift_enabled()
        else 'L only (lift fixed at start)'
      )
      if dual_seq == 'plan_merged':
        self.get_logger().warn(
          f'dual_sequence=plan_merged: plan {lift_note} → plan R at L goal → merge → '
          'ONE confirm_execute (expect logs: dual plan 1/2, dual plan 2/2, dual plan merged)'
        )
      elif dual_seq == 'left_then_right':
        self.get_logger().warn(
          f'dual_sequence=left_then_right: {lift_note} then right arm '
          '(two execute steps; use plan_merged for one RViz path)'
        )
      elif dual_seq in ('lift_midpoint', 'right_lift_first'):
        self.get_logger().warn(
          'dual_sequence=lift_midpoint: probe L+lift & R+lift → lift midpoint → '
          'replan L/R @ shared lift → ONE confirm → execute lift → left → right'
        )

    self._init_goal_cache()
    self._setup_target_visualization()

  def _required_arm_joints(self) -> List[str]:
    if self._planning_mode == 'right':
      return list(ARM_R_JOINTS)
    if self._planning_mode == 'dual':
      return list(ARM_L_JOINTS) + list(ARM_R_JOINTS)
    return list(ARM_L_JOINTS)

  def _trajectory_joints_for_mode(self) -> List[str]:
    if self._planning_mode == 'right':
      return list(TRAJECTORY_JOINTS_R)
    if self._planning_mode == 'dual':
      return list(TRAJECTORY_JOINTS_DUAL)
    return list(TRAJECTORY_JOINTS_L)

  def _goals_ready(self) -> bool:
    if self._planning_mode == 'left':
      return self._goal_l_ready
    if self._planning_mode == 'right':
      return self._goal_r_ready
    return self._goal_l_ready and self._goal_r_ready

  def _missing_goal_hint(self) -> str:
    if self._planning_mode == 'dual':
      missing = []
      if not self._goal_l_ready:
        missing.append(str(self.get_parameter('goal_pose_topic').value))
      if not self._goal_r_ready:
        missing.append(str(self.get_parameter('goal_pose_r_topic').value))
      return ' and '.join(missing) if missing else ''
    if self._planning_mode == 'right':
      return str(self.get_parameter('goal_pose_r_topic').value)
    return str(self.get_parameter('goal_pose_topic').value)

  def _resolve_apply_scene_service(self) -> str:
    custom = str(self.get_parameter('apply_planning_scene_service').value).strip()
    if custom:
      return custom if custom.startswith('/') else '/' + custom
    raw = str(self.get_parameter('move_group_namespace').value).strip()
    low = raw.lower()
    if low in ('', 'root', '/'):
      return '/apply_planning_scene'
    ns = raw if raw.startswith('/') else '/' + raw
    return f'{ns.rstrip("/")}/apply_planning_scene'

  def _resolve_cartesian_path_service(self) -> str:
    raw = str(self.get_parameter('move_group_namespace').value).strip()
    low = raw.lower()
    if low in ('', 'root', '/'):
      return '/compute_cartesian_path'
    ns = raw if raw.startswith('/') else '/' + raw
    return f'{ns.rstrip("/")}/compute_cartesian_path'

  def _goal_quat_for_position(
    self, xyz: tuple[float, float, float], arm: str = 'l'
  ) -> Quaternion:
    bx = float(self.get_parameter('robot_base_x').value)
    by = float(self.get_parameter('robot_base_y').value)
    roll_r = float(self.get_parameter('right_arm_roll_rad').value)
    return quat_toward_robot_y_up_arm(
      xyz, robot_base_xy=(bx, by), arm=arm, right_roll_rad=roll_r
    )

  def _goal_orientation_mode(self) -> str:
    return str(self.get_parameter('goal_orientation_mode').value).strip().lower()

  def _server_orientation_mode(self) -> str:
    mode = self._goal_orientation_mode()
    if mode == 'free':
      return 'free'
    if mode == 'fixed':
      return 'fixed'
    return 'soft'

  def _resolve_goal_orientation(
    self,
    xyz: tuple[float, float, float],
    msg_quat_wxyz: tuple[float, float, float, float] | None = None,
    arm: str = 'l',
  ) -> Quaternion:
    mode = self._goal_orientation_mode()
    if mode == 'from_message' and msg_quat_wxyz is not None:
      w, x, y, z = msg_quat_wxyz
      q = Quaternion()
      q.w, q.x, q.y, q.z = float(w), float(x), float(y), float(z)
      return q
    if mode == 'free':
      q = Quaternion()
      q.w = 1.0
      return q
    if mode == 'fixed' and msg_quat_wxyz is not None:
      w, x, y, z = msg_quat_wxyz
      q = Quaternion()
      q.w, q.x, q.y, q.z = float(w), float(x), float(y), float(z)
      return q
    if mode in ('soft', 'toward_robot', 'fixed'):
      return self._goal_quat_for_position(xyz, arm)
    self.get_logger().warn(
      f'Unknown goal_orientation_mode={mode!r}; using toward_robot hint'
    )
    return self._goal_quat_for_position(xyz, arm)

  def _quat_for_viz(
    self,
    xyz: tuple[float, float, float],
    plan_quat: Quaternion,
    arm: str = 'l',
  ) -> Quaternion:
    """RViz axes: show toward_robot frame even when plan mode is free."""
    mode = self._goal_orientation_mode()
    if mode == 'free' and bool(self.get_parameter('viz_toward_robot_in_free_mode').value):
      return self._goal_quat_for_position(xyz, arm)
    return plan_quat

  def _orientation_mode_log_label(self) -> str:
    mode = self._goal_orientation_mode()
    if mode == 'free':
      return 'position only (rotation free)'
    if mode == 'from_message':
      return 'orientation from /target_pose'
    if mode == 'fixed':
      return 'fixed orientation'
    w = float(self.get_parameter('orientation_weight').value)
    return f'soft (+X→robot, +Y↑, weight={w:.2f}; right arm +roll π)'

  def _init_goal_cache(self) -> None:
    seed = self.get_parameter('seed_goal_on_start').value
    seed_on = seed if isinstance(seed, bool) else str(seed).strip().lower() in ('true', '1', 'yes')
    if not seed_on:
      self._goal_l_ready = False
      self._goal_r_ready = False
      self._cached_goal_l_xyz = None
      self._cached_goal_l_quat = None
      self._cached_goal_r_xyz = None
      self._cached_goal_r_quat = None
      self.get_logger().info(
        f'seed_goal_on_start=false: waiting for goal(s); mode={self._planning_mode}, '
        f'need {self._missing_goal_hint() or "targets"}'
      )
      return

    self._goal_l_ready = self._planning_mode in ('left', 'dual')
    self._goal_r_ready = self._planning_mode in ('right', 'dual')
    home = self.get_parameter('home').value
    home_on = home if isinstance(home, bool) else str(home).strip().lower() in ('true', '1', 'yes')
    if home_on:
      if self._goal_l_ready:
        self._cached_goal_l_xyz = (0.45, 0.25, 1.35)
        self._cached_goal_l_quat = self._resolve_goal_orientation(
          self._cached_goal_l_xyz, arm='l'
        )
      if self._goal_r_ready:
        self._cached_goal_r_xyz = (0.45, -0.25, 1.35)
        self._cached_goal_r_quat = self._resolve_goal_orientation(
          self._cached_goal_r_xyz, arm='r'
        )
      self.get_logger().info('home=true: using fixed bilateral reach poses.')
      return

    raw_gs = self.get_parameter('goal_seed').value
    try:
      goal_seed_param = int(raw_gs)
    except (TypeError, ValueError):
      goal_seed_param = -1

    if goal_seed_param >= 0:
      rng = random.Random(goal_seed_param)
    else:
      rng = random.Random(secrets.randbelow(2**63))

    def _u(name: str) -> float:
      a = float(self.get_parameter(f'goal_pose_{name}_min').value)
      b = float(self.get_parameter(f'goal_pose_{name}_max').value)
      return rng.uniform(min(a, b), max(a, b))

    if self._goal_l_ready:
      x, y, z = _u('x'), _u('y'), _u('z')
      self._cached_goal_l_xyz = (x, y, z)
      self._cached_goal_l_quat = self._resolve_goal_orientation(
        self._cached_goal_l_xyz, arm='l'
      )
      self.get_logger().info(
        f'Cached left goal xyz=({x:.3f},{y:.3f},{z:.3f}) ({self._orientation_mode_log_label()})'
      )
    if self._goal_r_ready:
      x, y, z = _u('x'), -_u('y'), _u('z')
      self._cached_goal_r_xyz = (x, y, z)
      self._cached_goal_r_quat = self._resolve_goal_orientation(
        self._cached_goal_r_xyz, arm='r'
      )
      self.get_logger().info(
        f'Cached right goal xyz=({x:.3f},{y:.3f},{z:.3f}) ({self._orientation_mode_log_label()})'
      )

  def _setup_target_visualization(self) -> None:
    if not bool(self.get_parameter('publish_target_visualization').value):
      return
    marker_topic = str(self.get_parameter('target_marker_topic').value).strip()
    pose_topic = str(self.get_parameter('target_pose_viz_topic').value).strip()
    contact_pose_topic = str(
      self.get_parameter('target_contact_pose_viz_topic').value
    ).strip()
    contact_pose_r_topic = str(
      self.get_parameter('target_contact_pose_viz_r_topic').value
    ).strip()
    self._target_marker_pub = self.create_publisher(MarkerArray, marker_topic, self._qos)
    self._target_pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
    self._target_contact_pose_pub = None
    self._target_contact_pose_r_pub = None
    if contact_pose_topic:
      self._target_contact_pose_pub = self.create_publisher(
        PoseStamped, contact_pose_topic, 10
      )
    if contact_pose_r_topic:
      self._target_contact_pose_r_pub = self.create_publisher(
        PoseStamped, contact_pose_r_topic, 10
      )
    period = float(self.get_parameter('target_marker_publish_period_sec').value)
    self._publish_target_visualization()
    if period > 0.0:
      self.create_timer(period, self._publish_target_visualization)
    contact_note = ''
    if contact_pose_topic:
      contact_note = f', contact PoseStamped "{contact_pose_topic}"'
    if contact_pose_r_topic:
      contact_note += f' / "{contact_pose_r_topic}"'
    self.get_logger().info(
      f'Target RViz: MarkerArray "{marker_topic}" '
      f'(cuRobo pre-grasp + contact), approach PoseStamped "{pose_topic}"'
      f'{contact_note} (frame={self._planning_frame})'
    )

  @staticmethod
  def _pose_from_goal(
    xyz: tuple[float, float, float], quat
  ) -> Pose:
    x, y, z = xyz
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.x = float(quat.x)
    pose.orientation.y = float(quat.y)
    pose.orientation.z = float(quat.z)
    pose.orientation.w = float(quat.w)
    return pose

  def _append_target_markers(
    self,
    arr: MarkerArray,
    stamp,
    frame: str,
    pose: Pose,
    *,
    ns: str,
    arrow_id: int,
    sphere_id: int,
    arrow_color: ColorRGBA,
    sphere_color: ColorRGBA,
  ) -> None:
    axis_len = float(self.get_parameter('target_arrow_length').value)
    sphere_d = float(self.get_parameter('target_sphere_diameter').value)
    if bool(self.get_parameter('target_show_ee_axes').value):
      append_ee_axes_markers(
        arr,
        stamp,
        frame,
        pose,
        ns=ns,
        marker_id=arrow_id,
        axis_length=axis_len,
      )
    sphere = Marker()
    sphere.header = Header(stamp=stamp, frame_id=frame)
    sphere.ns = ns
    sphere.id = sphere_id
    sphere.type = Marker.SPHERE
    sphere.action = Marker.ADD
    sphere.pose = pose
    sphere.scale.x = sphere_d
    sphere.scale.y = sphere_d
    sphere.scale.z = sphere_d
    sphere.color = sphere_color
    arr.markers.append(sphere)

  def _append_contact_target_markers(
    self,
    arr: MarkerArray,
    stamp,
    frame: str,
    approach_xyz: tuple[float, float, float],
    contact_xyz: tuple[float, float, float],
    contact_quat,
    *,
    ns: str,
    contact_id: int,
    line_id: int,
    contact_color: ColorRGBA,
  ) -> None:
    sphere_d = float(self.get_parameter('target_sphere_diameter').value)
    contact_pose = self._pose_from_goal(contact_xyz, contact_quat)

    contact_sphere = Marker()
    contact_sphere.header = Header(stamp=stamp, frame_id=frame)
    contact_sphere.ns = ns
    contact_sphere.id = contact_id
    contact_sphere.type = Marker.SPHERE
    contact_sphere.action = Marker.ADD
    contact_sphere.pose = contact_pose
    d = sphere_d * 0.7
    contact_sphere.scale.x = d
    contact_sphere.scale.y = d
    contact_sphere.scale.z = d
    contact_sphere.color = contact_color
    arr.markers.append(contact_sphere)

    ax, ay, az = approach_xyz
    cx, cy, cz = contact_xyz
    line = Marker()
    line.header = Header(stamp=stamp, frame_id=frame)
    line.ns = ns
    line.id = line_id
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.scale.x = max(sphere_d * 0.08, 0.003)
    line.color = ColorRGBA(
      r=contact_color.r,
      g=contact_color.g,
      b=contact_color.b,
      a=min(contact_color.a, 0.75),
    )
    p0 = Point()
    p0.x, p0.y, p0.z = float(ax), float(ay), float(az)
    p1 = Point()
    p1.x, p1.y, p1.z = float(cx), float(cy), float(cz)
    line.points = [p0, p1]
    arr.markers.append(line)

  def _make_target_markers(self, stamp) -> MarkerArray:
    frame = self._planning_frame
    arr = MarkerArray()
    if self._goal_l_ready and self._cached_goal_l_xyz and self._cached_goal_l_quat:
      self._append_target_markers(
        arr,
        stamp,
        frame,
        self._pose_from_goal(
          self._cached_goal_l_xyz,
          self._quat_for_viz(self._cached_goal_l_xyz, self._cached_goal_l_quat, 'l'),
        ),
        ns=TARGET_MARKER_NS_L,
        arrow_id=TARGET_MARKER_ID_L_AXES,
        sphere_id=TARGET_MARKER_ID_L_SPHERE,
        arrow_color=ColorRGBA(r=0.1, g=0.85, b=0.25, a=0.95),
        sphere_color=ColorRGBA(r=0.1, g=0.55, b=1.0, a=0.75),
      )
      if self._cached_contact_l_xyz and self._cached_contact_l_quat:
        self._append_contact_target_markers(
          arr,
          stamp,
          frame,
          self._cached_goal_l_xyz,
          self._cached_contact_l_xyz,
          self._cached_contact_l_quat,
          ns=TARGET_CONTACT_NS_L,
          contact_id=TARGET_MARKER_ID_L_CONTACT,
          line_id=TARGET_MARKER_ID_L_CONTACT_LINE,
          contact_color=ColorRGBA(r=0.95, g=0.95, b=0.2, a=0.95),
        )
    if self._goal_r_ready and self._cached_goal_r_xyz and self._cached_goal_r_quat:
      self._append_target_markers(
        arr,
        stamp,
        frame,
        self._pose_from_goal(
          self._cached_goal_r_xyz,
          self._quat_for_viz(self._cached_goal_r_xyz, self._cached_goal_r_quat, 'r'),
        ),
        ns=TARGET_MARKER_NS_R,
        arrow_id=TARGET_MARKER_ID_R_AXES,
        sphere_id=TARGET_MARKER_ID_R_SPHERE,
        arrow_color=ColorRGBA(r=0.95, g=0.45, b=0.1, a=0.95),
        sphere_color=ColorRGBA(r=1.0, g=0.55, b=0.1, a=0.75),
      )
      if self._cached_contact_r_xyz and self._cached_contact_r_quat:
        self._append_contact_target_markers(
          arr,
          stamp,
          frame,
          self._cached_goal_r_xyz,
          self._cached_contact_r_xyz,
          self._cached_contact_r_quat,
          ns=TARGET_CONTACT_NS_R,
          contact_id=TARGET_MARKER_ID_R_CONTACT,
          line_id=TARGET_MARKER_ID_R_CONTACT_LINE,
          contact_color=ColorRGBA(r=0.95, g=0.85, b=0.2, a=0.95),
        )
    return arr

  def _publish_target_visualization(self) -> None:
    if not bool(self.get_parameter('publish_target_visualization').value):
      return
    if not self._goals_ready() and not (self._goal_l_ready or self._goal_r_ready):
      return
    if not hasattr(self, '_target_marker_pub'):
      return
    stamp = self.get_clock().now().to_msg()
    self._target_marker_pub.publish(self._make_target_markers(stamp))
    if self._goal_l_ready and self._cached_goal_l_xyz and self._cached_goal_l_quat:
      pose_msg = PoseStamped()
      pose_msg.header.stamp = stamp
      pose_msg.header.frame_id = self._planning_frame
      pose_msg.pose = self._pose_from_goal(
        self._cached_goal_l_xyz,
        self._quat_for_viz(self._cached_goal_l_xyz, self._cached_goal_l_quat, 'l'),
      )
      self._target_pose_pub.publish(pose_msg)
    if (
      self._target_contact_pose_pub is not None
      and self._cached_contact_l_xyz is not None
      and self._cached_contact_l_quat is not None
    ):
      contact_msg = PoseStamped()
      contact_msg.header.stamp = stamp
      contact_msg.header.frame_id = self._planning_frame
      contact_msg.pose = self._pose_from_goal(
        self._cached_contact_l_xyz,
        self._quat_for_viz(
          self._cached_contact_l_xyz, self._cached_contact_l_quat, 'l'
        ),
      )
      self._target_contact_pose_pub.publish(contact_msg)
    if (
      self._target_contact_pose_r_pub is not None
      and self._cached_contact_r_xyz is not None
      and self._cached_contact_r_quat is not None
    ):
      contact_r_msg = PoseStamped()
      contact_r_msg.header.stamp = stamp
      contact_r_msg.header.frame_id = self._planning_frame
      contact_r_msg.pose = self._pose_from_goal(
        self._cached_contact_r_xyz,
        self._quat_for_viz(
          self._cached_contact_r_xyz, self._cached_contact_r_quat, 'r'
        ),
      )
      self._target_contact_pose_r_pub.publish(contact_r_msg)

  def _wait_curobo(self, timeout_sec: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
      try:
        r = requests.get(self._health_url, timeout=2.0)
        if r.status_code == 200 and r.json().get('ok'):
          self.get_logger().info(
            f'cuRobo server OK (robot_file={r.json().get("robot_file")})'
          )
          return True
      except requests.RequestException:
        pass
      time.sleep(0.5)
    self.get_logger().error(f'cuRobo server not ready at {self._health_url}')
    return False

  def _wait_service(self, timeout_sec: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
      if self._apply_scene_cli.service_is_ready():
        return True
      rclpy.spin_once(self, timeout_sec=0.05)
    self.get_logger().error('Timeout: apply_planning_scene not available')
    return False

  def _wait_action_server(self, client: ActionClient, name: str, timeout_sec: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
      if client.server_is_ready():
        return True
      rclpy.spin_once(self, timeout_sec=0.05)
    self.get_logger().error(f'Timeout: {name}')
    return False

  def _wait_future_done(self, fut, timeout_sec: float = 120.0) -> bool:
    """Block without calling executor.spin* (safe under MultiThreadedExecutor)."""
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok():
      if fut.done():
        return True
      if time.monotonic() >= deadline:
        self.get_logger().error('Timed out waiting for async future (service/action)')
        return False
      time.sleep(0.002)
    return False

  def ready_for_planning(self) -> bool:
    if not self._wait_curobo():
      return False
    if bool(self.get_parameter('sync_moveit_scene').value) and not self._wait_service():
      return False
    if not self._wait_action_server(self._execute, 'ExecuteTrajectory'):
      return False
    if (
      bool(self.get_parameter('enable_final_tool_approach').value)
      and bool(self.get_parameter('final_approach_use_move_group').value)
      and not self._wait_action_server(self._move_group, 'MoveGroup')
    ):
      return False
    js_timeout = max(float(self.get_parameter('wait_for_joint_states_sec').value), 0.0)
    if js_timeout > 0.0 and not self._wait_for_joint_states(js_timeout):
      msg = (
        'No /joint_states received yet (check follower bringup, use_sim_time, '
        'and joint_state QoS; try joint_state_best_effort_qos:=true).'
      )
      if bool(self.get_parameter('require_joint_states_at_startup').value):
        self.get_logger().error(msg)
        return False
      self.get_logger().warn(
        f'{msg} Node keeps running — planning waits for /joint_states.'
      )
    return True

  def _wait_for_joint_states(self, timeout_sec: float) -> bool:
    if self._latest_joint_state is not None:
      return True
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
      if self._latest_joint_state is not None:
        self.get_logger().info('Received /joint_states.')
        return True
      rclpy.spin_once(self, timeout_sec=0.05)
    return self._latest_joint_state is not None

  def arm_planning(self) -> None:
    """Enable auto-plan after cuRobo, MoveIt, and joint_states are ready."""
    self._planning_armed = True
    ready, detail = self._plan_readiness_status()
    if ready:
      self.get_logger().info(
        'Planning armed — prerequisites met; scheduling plan.'
      )
      self._schedule_plan('planning_armed', delay_sec=0.1)
    else:
      self.get_logger().info(f'Planning armed — waiting for: {detail}')
    self._start_readiness_poll_timer()
    self._schedule_deferred_moveit_scene_sync()

  def _sync_moveit_scene_from_cuboids(self, cuboids) -> bool:
    if not bool(self.get_parameter('sync_moveit_scene').value):
      return True
    if not cuboids:
      return True
    stamp = self.get_clock().now().to_msg()
    scene = planning_scene_from_cuboids(
      cuboids, stamp, frame_id=self._planning_frame
    )
    if scene is None:
      return True
    return self._call_apply_planning_scene(scene)

  def _sync_moveit_scene_from_obstacles(self, msg: MarkerArray) -> bool:
    obs_scale = float(self.get_parameter('curobo_obstacle_dim_scale').value)
    cuboids = markers_to_planning_cuboids(msg, dim_scale=obs_scale)
    return self._sync_moveit_scene_from_cuboids(cuboids)

  def _on_arm_moveit_scene_timer(self) -> None:
    if self._moveit_scene_arm_timer is not None:
      self._moveit_scene_arm_timer.cancel()
      self.destroy_timer(self._moveit_scene_arm_timer)
      self._moveit_scene_arm_timer = None
    msg = self._last_obstacle_msg
    cuboids = self._last_obstacle_cuboids
    if cuboids:
      self._sync_moveit_scene_from_cuboids(cuboids)
    elif msg is not None:
      self._sync_moveit_scene_from_obstacles(msg)

  def _schedule_deferred_moveit_scene_sync(self) -> None:
    if not bool(self.get_parameter('sync_moveit_scene').value):
      return
    if self._last_obstacle_msg is None:
      return
    if self._moveit_scene_arm_timer is not None:
      self._moveit_scene_arm_timer.cancel()
      self.destroy_timer(self._moveit_scene_arm_timer)
    self._moveit_scene_arm_timer = self.create_timer(
      0.05,
      self._on_arm_moveit_scene_timer,
      callback_group=self._cb_group,
    )

  def create_obstacle_subscription(self) -> None:
    topic = str(self.get_parameter('obstacle_topic').value)
    self.create_subscription(
      MarkerArray,
      topic,
      self._on_obstacles,
      self._planning_input_qos,
      callback_group=self._cb_group,
    )
    self.get_logger().info(f'Subscribed to {topic} (MarkerArray)')
    debug_topic = str(self.get_parameter('curobo_obstacle_debug_topic').value)
    self._curobo_obstacle_debug_pub = self.create_publisher(
      MarkerArray, debug_topic, self._qos
    )
    if bool(self.get_parameter('publish_curobo_obstacle_debug').value):
      self.get_logger().info(
        f'Planning obstacle viz (MoveIt+cuRobo geometry): {debug_topic}'
      )
    else:
      self._clear_curobo_obstacle_debug_markers()
      self.get_logger().info(
        f'Planning obstacle viz OFF — cleared {debug_topic}'
      )

  def _publish_planning_obstacle_markers(self, cuboids) -> None:
    if not bool(self.get_parameter('publish_curobo_obstacle_debug').value):
      self._clear_curobo_obstacle_debug_markers()
      return
    if not hasattr(self, '_curobo_obstacle_debug_pub'):
      return
    stamp = self.get_clock().now().to_msg()
    self._curobo_obstacle_debug_pub.publish(
      curobo_cuboids_to_marker_array(
        cuboids, stamp, frame_id=self._planning_frame
      )
    )

  def _clear_curobo_obstacle_debug_markers(self) -> None:
    if not hasattr(self, '_curobo_obstacle_debug_pub'):
      return
    stamp = self.get_clock().now().to_msg()
    self._curobo_obstacle_debug_pub.publish(
      curobo_obstacle_debug_delete_all(stamp, frame_id=self._planning_frame)
    )

  def _on_joint_states(self, msg: JointState) -> None:
    first = self._latest_joint_state is None
    self._latest_joint_state = msg
    self._latest_joint_state_mono = time.monotonic()
    if first:
      self.get_logger().info(
        f'Received /joint_states ({len(msg.name)} joints).'
      )
    # Detect ROS-time mismatch (e.g., use_sim_time mismatch or stale timestamps).
    stamp = msg.header.stamp
    stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if stamp_sec > 0.0:
      now_msg = self.get_clock().now().to_msg()
      now_sec = float(now_msg.sec) + float(now_msg.nanosec) * 1e-9
      diff = abs(now_sec - stamp_sec)
      tol = max(float(self.get_parameter('joint_state_stamp_tolerance_sec').value), 0.1)
      if diff > tol:
        mono = time.monotonic()
        if mono - self._last_joint_state_stamp_warn_mono > 5.0:
          self._last_joint_state_stamp_warn_mono = mono
          use_sim = (
            self.has_parameter('use_sim_time')
            and bool(self.get_parameter('use_sim_time').value)
          )
          if use_sim and now_sec < 10.0 and stamp_sec > 1.0e6:
            hint = (
              'This node use_sim_time=true but /clock is missing or /joint_states '
              'uses wall time. Real robot: launch with use_sim:=false.'
            )
          elif not use_sim and stamp_sec < 1.0e4 and now_sec > 1.0e6:
            hint = (
              'This node uses wall time but /joint_states stamp looks like sim time. '
              'Match use_sim_time on all nodes or fix the joint_state publisher.'
            )
          else:
            hint = 'Check use_sim_time and /clock sync on every node.'
          self.get_logger().warn(
            f'/joint_states stamp drift: |now-stamp|={diff:.3f}s '
            f'(now={now_sec:.3f}, stamp={stamp_sec:.3f}, use_sim_time={use_sim}, '
            f'tolerance={tol:.3f}s). {hint}'
          )

  def _joint_state_is_fresh(self, stale_sec: Optional[float] = None) -> bool:
    if self._latest_joint_state is None:
      return False
    if stale_sec is None:
      stale_sec = max(float(self.get_parameter('joint_state_stale_sec').value), 0.0)
    if stale_sec <= 0.0:
      return True
    age = time.monotonic() - self._latest_joint_state_mono
    return age <= stale_sec

  def _wait_joint_states_after_move(self, label: str) -> bool:
    settle = max(float(self.get_parameter('dual_sequence_settle_sec').value), 0.0)
    wait_extra = max(float(self.get_parameter('dual_sequence_joint_wait_sec').value), 0.0)
    relaxed = max(
      float(self.get_parameter('joint_state_stale_sec').value) * 4.0,
      2.0,
    )
    if settle > 0.0:
      self._plan_progress('dual sequence settle', f'{label}: waiting {settle:.1f}s')
      time.sleep(settle)
    deadline = time.monotonic() + wait_extra
    while time.monotonic() < deadline:
      if self._latest_joint_state is not None and self._joint_state_is_fresh(relaxed):
        return True
      time.sleep(0.05)
    if self._latest_joint_state is not None:
      age = time.monotonic() - self._latest_joint_state_mono
      self._plan_progress(
        'dual sequence warn',
        f'{label}: /joint_states age={age:.2f}s — continuing with last sample',
      )
      return True
    return False

  def _current_curobo_joint_state(self) -> tuple[List[str], List[float]]:
    """Full-body start state for cuRobo FK; trajectory still uses arm_l only."""
    if self._latest_joint_state is None:
      raise RuntimeError('No /joint_states received yet')
    name_to_pos: Dict[str, float] = dict(
      zip(self._latest_joint_state.name, self._latest_joint_state.position)
    )
    use_passive = bool(self.get_parameter('use_passive_defaults_for_start').value)
    passive_defaults = CUROBO_PASSIVE_JOINT_DEFAULTS if use_passive else {}
    names: List[str] = []
    positions: List[float] = []
    missing: List[str] = []
    for jn in self._curobo_state_joints:
      if jn in name_to_pos:
        names.append(jn)
        positions.append(float(name_to_pos[jn]))
      elif use_passive and jn in passive_defaults:
        names.append(jn)
        positions.append(passive_defaults[jn])
      else:
        missing.append(jn)
    if not names:
      raise RuntimeError('No cuRobo state joints found in /joint_states')
    for jn in self._arm_joints:
      if jn not in name_to_pos:
        raise RuntimeError(f'Joint {jn} missing from /joint_states')
    if self._planning_mode in ('left', 'dual') and 'lift_joint' not in name_to_pos:
      raise RuntimeError('Joint lift_joint missing from /joint_states')
    if missing:
      self.get_logger().warn(
        f'cuRobo start state: {len(missing)} joints missing from /joint_states '
        f'(cuRobo server YAML defaults): {missing}',
        throttle_duration_sec=30.0,
      )
    return names, positions

  def _resolve_skip_start_feasibility(self) -> bool:
    """Sim /joint_states home often fails cuRobo self-collision sphere model."""
    if bool(self.get_parameter('skip_start_feasibility_check').value):
      return True
    if self.has_parameter('use_sim_time') and bool(self.get_parameter('use_sim_time').value):
      return True
    return False

  def _goal_from_pose_stamped(self, msg: PoseStamped) -> tuple:
    if msg.header.frame_id and msg.header.frame_id != self._planning_frame:
      self.get_logger().warn(
        f'Goal frame {msg.header.frame_id} != {self._planning_frame}; '
        'assuming pose is already in planning frame.'
      )
    p = msg.pose.position
    q = msg.pose.orientation
    return (float(p.x), float(p.y), float(p.z)), (
      float(q.w),
      float(q.x),
      float(q.y),
      float(q.z),
    )

  def _goal_pose_epsilon(self) -> float:
    return max(float(self.get_parameter('goal_pose_epsilon_m').value), 1e-6)

  def _goal_pose_changed(
    self,
    xyz: tuple[float, float, float],
    quat: Quaternion,
    arm: str,
  ) -> bool:
    if arm == 'l':
      prev_xyz, prev_quat = self._cached_goal_l_xyz, self._cached_goal_l_quat
    else:
      prev_xyz, prev_quat = self._cached_goal_r_xyz, self._cached_goal_r_quat
    if prev_xyz is None or prev_quat is None:
      return True
    eps = self._goal_pose_epsilon()
    if math.dist(xyz, prev_xyz) > eps:
      return True
    if self._goal_orientation_mode() in ('from_message', 'fixed'):
      dq = (
        abs(float(prev_quat.w) - float(quat.w))
        + abs(float(prev_quat.x) - float(quat.x))
        + abs(float(prev_quat.y) - float(quat.y))
        + abs(float(prev_quat.z) - float(quat.z))
      )
      return dq > 1e-3
    return False

  def _contact_pose_changed(
    self,
    xyz: tuple[float, float, float],
    quat: Quaternion,
    arm: str,
  ) -> bool:
    if arm == 'l':
      prev_xyz, prev_quat = self._cached_contact_l_xyz, self._cached_contact_l_quat
    else:
      prev_xyz, prev_quat = self._cached_contact_r_xyz, self._cached_contact_r_quat
    if prev_xyz is None or prev_quat is None:
      return True
    eps = self._goal_pose_epsilon()
    if math.dist(xyz, prev_xyz) > eps:
      return True
    dq = (
      abs(float(prev_quat.w) - float(quat.w))
      + abs(float(prev_quat.x) - float(quat.x))
      + abs(float(prev_quat.y) - float(quat.y))
      + abs(float(prev_quat.z) - float(quat.z))
    )
    return dq > 1e-3

  def _update_goal_from_pose(
    self, msg: PoseStamped, arm: str, reason_suffix: str
  ) -> None:
    xyz, msg_quat = self._goal_from_pose_stamped(msg)
    quat = self._resolve_goal_orientation(xyz, msg_quat, arm)
    if not self._goal_pose_changed(xyz, quat, arm):
      return

    if arm == 'l':
      first = not self._goal_l_ready
      self._cached_goal_l_xyz = xyz
      self._cached_goal_l_quat = quat
      self._goal_l_ready = True
      label = 'Left'
    else:
      first = not self._goal_r_ready
      self._cached_goal_r_xyz = xyz
      self._cached_goal_r_quat = quat
      self._goal_r_ready = True
      label = 'Right'
    self._motion_log(
      'cuRobo',
      arm,
      f'goal updated (pre-grasp /target_pose): xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})',
    )
    self._publish_target_visualization()
    if self._planning_mode == 'dual':
      if not self._goals_ready():
        self.get_logger().info(
          f'{label} goal cached; dual plan waits for: {self._missing_goal_hint()}',
          throttle_duration_sec=3.0,
        )
        return
      if not self._goals_ready_for_plan():
        _, detail = self._plan_readiness_status()
        self.get_logger().info(
          f'{label} goal cached; plan waits for: {detail}',
          throttle_duration_sec=3.0,
        )
        return
      self._schedule_plan('dual_goals_ready')
      return
    self._schedule_plan(
      f'{reason_suffix}' if not first else f'first_{reason_suffix}',
      delay_sec=0.2,
    )

  def _on_goal_pose_l(self, msg: PoseStamped) -> None:
    self._update_goal_from_pose(msg, 'l', 'goal_pose_l')

  def _on_goal_pose_r(self, msg: PoseStamped) -> None:
    self._update_goal_from_pose(msg, 'r', 'goal_pose_r')

  def _update_contact_from_pose(self, msg: PoseStamped, arm: str) -> None:
    xyz, quat_wxyz = self._goal_from_pose_stamped(msg)
    quat = self._quat_from_wxyz_tuple(quat_wxyz)
    if not self._contact_pose_changed(xyz, quat, arm):
      return
    if arm == 'l':
      self._cached_contact_l_xyz = xyz
      self._cached_contact_l_quat = quat
    else:
      self._cached_contact_r_xyz = xyz
      self._cached_contact_r_quat = quat
    approach_xyz = self._goal_xyz_for_arm(arm)
    approach_note = ''
    if approach_xyz is not None:
      dist = math.dist(approach_xyz, xyz)
      approach_note = (
        f', offset from cuRobo goal={dist:.3f}m '
        f'(approach=({approach_xyz[0]:.3f},{approach_xyz[1]:.3f},{approach_xyz[2]:.3f}))'
      )
    self._motion_log(
      'MoveIt',
      arm,
      f'contact goal updated (/target_pose_contact): '
      f'xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}){approach_note}',
    )
    self._publish_target_visualization()

  def _on_goal_contact_l(self, msg: PoseStamped) -> None:
    self._update_contact_from_pose(msg, 'l')

  def _on_goal_contact_r(self, msg: PoseStamped) -> None:
    self._update_contact_from_pose(msg, 'r')

  def _call_apply_planning_scene(self, scene) -> bool:
    req = ApplyPlanningScene.Request()
    req.scene = scene
    fut = self._apply_scene_cli.call_async(req)
    if not self._wait_future_done(fut):
      return False
    resp = fut.result()
    if resp is None or not resp.success:
      self.get_logger().error('ApplyPlanningScene failed')
      return False
    return True

  def _on_obstacles(self, msg: MarkerArray) -> None:
    obs_scale = float(self.get_parameter('curobo_obstacle_dim_scale').value)
    with self._obstacle_lock:
      sig = marker_array_signature(msg)
      if self._last_marker_sig is not None and sig == self._last_marker_sig:
        return
      self._last_obstacle_cuboids = markers_to_planning_cuboids(msg, obs_scale)
      self._last_marker_sig = sig
      self._last_obstacle_msg = msg

    cuboid_lines = ', '.join(
      f"{o['name']} dims={[round(d, 3) for d in o['dims']]} @"
      f"({o['pose'][0]:.2f},{o['pose'][1]:.2f},{o['pose'][2]:.2f})"
      for o in self._last_obstacle_cuboids[:4]
    )
    self.get_logger().info(
      f'Planning obstacles (MoveIt+cuRobo): {len(self._last_obstacle_cuboids)} cuboid(s)'
      f' (dim_scale={obs_scale:.2f}; {cuboid_lines})'
    )
    self._publish_planning_obstacle_markers(self._last_obstacle_cuboids)

    if self._planning_armed and bool(self.get_parameter('sync_moveit_scene').value):
      if not self._sync_moveit_scene_from_cuboids(self._last_obstacle_cuboids):
        self.get_logger().warn(
          'ApplyPlanningScene failed; cuRobo cuboids still updated.'
        )

    if not self._planning_armed:
      return

    settle = float(self.get_parameter('scene_settle_sec').value)
    if settle > 0.0:
      time.sleep(settle)

    if not bool(self.get_parameter('auto_plan').value):
      self.get_logger().info('auto_plan=false; scene updated only.')
      return

    if not self._goals_ready_for_plan():
      _, detail = self._plan_readiness_status()
      self.get_logger().info(
        f'Obstacle update received; plan waits for: {detail}',
        throttle_duration_sec=5.0,
      )
      return

    now = time.monotonic()
    min_dt = float(self.get_parameter('min_plan_interval_sec').value)
    if now - self._last_plan_mono < min_dt:
      return

    self._schedule_plan('obstacle_update', delay_sec=0.05)
    self.get_logger().info(
      'Obstacle scene updated — scheduled dual plan (goals + back wall ready)',
    )

  def _dual_sequence_mode(self) -> str:
    mode = str(self.get_parameter('dual_sequence').value).strip().lower()
    if mode == 'right_lift_first':
      self.get_logger().warn(
        'dual_sequence=right_lift_first is deprecated; using lift_midpoint'
      )
      return 'lift_midpoint'
    if mode in ('plan_merged', 'parallel', 'left_then_right', 'lift_midpoint'):
      return mode
    self.get_logger().warn(f'Unknown dual_sequence={mode!r}; using plan_merged')
    return 'plan_merged'

  def _dual_plan_lift_enabled(self) -> bool:
    return bool(self.get_parameter('dual_plan_lift').value)

  def _dual_left_trajectory_joints(self) -> List[str]:
    if self._dual_plan_lift_enabled():
      return list(TRAJECTORY_JOINTS_L)
    return list(TRAJECTORY_JOINTS_L_DUAL)

  def _resolve_velocity_scale(self) -> float:
    """Map velocity_scale param to fraction in (0,1].

    Values in (0,1] are fractions (0.2 = 20% speed).
    Values in (1,100] are percents (20 = 20% speed).
    Values >100 were a common mistake and are treated as percent (clamped to 100%).
    """
    raw = float(self.get_parameter('velocity_scale').value)
    if raw > 1.0:
      effective = raw / 100.0
      if raw > 100.0:
        self.get_logger().warn(
          f'velocity_scale={raw} > 100; clamping to 100% (use 20 for 20%%, not 100+).'
        )
        effective = 1.0
    else:
      effective = raw
    return max(min(effective, 1.0), 0.01)

  def _plan_progress(self, stage: str, detail: str = '') -> None:
    msg = f'[plan] {stage}'
    if detail:
      msg += f' | {detail}'
    self.get_logger().info(msg)

  @staticmethod
  def _arm_tag(arm: str) -> str:
    return 'LEFT' if str(arm).lower() == 'l' else 'RIGHT'

  def _motion_log(self, channel: str, arm: Optional[str], message: str, *, level: str = 'info') -> None:
    """Structured motion log: channel=cuRobo|MoveIt|RViz, arm=l|r|None."""
    tag = f'[{channel}'
    if arm is not None:
      tag += f'/{self._arm_tag(arm)}'
    tag += ']'
    full = f'{tag} {message}'
    if level == 'warn':
      self.get_logger().warn(full)
    elif level == 'error':
      self.get_logger().error(full)
    else:
      self.get_logger().info(full)

  def _resolve_final_approach_arm(self, step_label: str) -> Optional[str]:
    """Arm for MoveIt STAGE 2 after cuRobo coarse execute."""
    low = step_label.lower()
    if 'lift' in low and 'left' not in low and 'right' not in low:
      return None
    if 'left' in low:
      return 'l'
    if 'right' in low:
      return 'r'
    if self._planning_mode == 'left':
      return 'l'
    if self._planning_mode == 'right':
      return 'r'
    return None

  def _goal_xyz_for_arm(self, arm: str) -> Optional[tuple[float, float, float]]:
    if arm == 'l':
      return self._cached_goal_l_xyz
    return self._cached_goal_r_xyz

  @staticmethod
  def _plan_response_summary(plan: dict) -> str:
    points = plan.get('points') or []
    if not points:
      return '0 points'
    duration = float(points[-1].get('time_from_start', 0.0))
    joints = plan.get('joint_names') or []
    return f'{len(points)} pts, {duration:.2f}s, joints={len(joints)}'

  def _start_plan_worker(self, compute_fn, *args, exec_label: str, skip_label: str) -> None:
    if not self._plan_lock.acquire(blocking=False):
      return

    def _run() -> None:
      traj: Optional[RobotTrajectory] = None
      t0 = time.monotonic()
      self._plan_progress('started', f'label={exec_label}, trigger={skip_label}')
      try:
        traj = compute_fn(*args)
      finally:
        self._plan_lock.release()
      elapsed = time.monotonic() - t0
      if traj is not None:
        jt = traj.joint_trajectory
        dur = 0.0
        if jt.points:
          dur = self._traj_point_time_sec(jt.points[-1])
        self._plan_progress(
          'trajectory ready',
          f'{len(jt.joint_names)} joints, {len(jt.points)} pts, '
          f'duration={dur:.2f}s, compute={elapsed:.2f}s',
        )
        self._confirm_and_execute_trajectory(traj, exec_label)
      else:
        self._plan_progress('failed', f'no trajectory ({elapsed:.2f}s)')

    threading.Thread(target=_run, daemon=True).start()

  def _is_goal_plan_reason(self, reason: str) -> bool:
    return (
      reason.startswith('goal_pose')
      or reason.startswith('first_goal_pose')
      or reason == 'dual_goals_ready'
    )

  def _plan_and_execute(self, reason: str) -> None:
    if not self._goals_ready_for_plan():
      self.get_logger().info(
        f'Skip plan ({reason}): goal(s) not ready — publish {self._missing_goal_hint()}',
        throttle_duration_sec=5.0,
      )
      return
    if self._awaiting_confirm:
      if reason == 'obstacle_update' or self._is_goal_plan_reason(reason):
        return
    if self._is_goal_plan_reason(reason):
      min_dt = float(self.get_parameter('min_plan_interval_sec').value)
      if time.monotonic() - self._last_plan_mono < min_dt:
        return
    if self._plan_lock.locked():
      return
    dual_seq = self._dual_sequence_mode() if self._planning_mode == 'dual' else ''
    self._plan_progress(
      'triggered',
      f'reason={reason}, mode={self._planning_mode}'
      + (f', dual_sequence={dual_seq}' if dual_seq else ''),
    )
    if self._planning_mode == 'dual' and dual_seq == 'left_then_right':
      self._start_dual_sequence_worker(reason)
      return
    if self._planning_mode == 'dual' and dual_seq == 'lift_midpoint':
      self._start_dual_lift_midpoint_worker(reason)
      return
    self._start_plan_worker(
      self._compute_pose_plan_trajectory,
      reason,
      exec_label='cuRobo',
      skip_label=reason,
    )

  def _start_dual_sequence_worker(self, reason: str) -> None:
    if not self._plan_lock.acquire(blocking=False):
      return

    def _run() -> None:
      t0 = time.monotonic()
      self._plan_progress('started', f'label=dual left_then_right, trigger={reason}')
      try:
        self._run_dual_left_then_right(reason)
      finally:
        self._plan_lock.release()
        self._plan_progress(
          'dual sequence finished', f'elapsed={time.monotonic() - t0:.2f}s'
        )

    threading.Thread(target=_run, daemon=True).start()

  def _start_dual_lift_midpoint_worker(self, reason: str) -> None:
    if not self._plan_lock.acquire(blocking=False):
      return

    def _run() -> None:
      t0 = time.monotonic()
      self._plan_progress('started', f'label=dual lift_midpoint, trigger={reason}')
      try:
        self._run_dual_lift_midpoint(reason)
      finally:
        self._plan_lock.release()
        self._plan_progress(
          'dual sequence finished', f'elapsed={time.monotonic() - t0:.2f}s'
        )

    threading.Thread(target=_run, daemon=True).start()

  def _back_wall_cuboid(self) -> Optional[dict]:
    suffix = f'_{TARGET_BACK_WALL_ID}'
    with self._obstacle_lock:
      for o in self._last_obstacle_cuboids:
        if str(o.get('name', '')).endswith(suffix):
          return o
    return None

  def _plan_readiness_status(self) -> tuple[bool, str]:
    """Return (ready, human-readable missing prerequisites for auto-plan)."""
    missing: List[str] = []
    if self._planning_mode == 'left' and not self._goal_l_ready:
      missing.append(str(self.get_parameter('goal_pose_topic').value))
    elif self._planning_mode == 'right' and not self._goal_r_ready:
      missing.append(str(self.get_parameter('goal_pose_r_topic').value))
    elif self._planning_mode == 'dual':
      if not self._goal_l_ready:
        missing.append(str(self.get_parameter('goal_pose_topic').value))
      if not self._goal_r_ready:
        missing.append(str(self.get_parameter('goal_pose_r_topic').value))
      if bool(self.get_parameter('require_back_wall_for_dual').value):
        if self._back_wall_cuboid() is None:
          missing.append(
            'back wall on /obstacle_markers (click L or R target in perception_click_planning)'
          )
    if not bool(self.get_parameter('auto_plan').value):
      missing.append('auto_plan:=true')
    js_topic = str(self.get_parameter('joint_state_topic').value)
    if self._latest_joint_state is None:
      missing.append(js_topic)
    elif not self._joint_state_is_fresh():
      age = time.monotonic() - self._latest_joint_state_mono
      missing.append(f'{js_topic} (stale {age:.1f}s)')
    if missing:
      return False, ', '.join(missing)
    return True, 'ready'

  def _start_readiness_poll_timer(self) -> None:
    if self._readiness_poll_timer is not None:
      return
    self._readiness_poll_timer = self.create_timer(
      2.0,
      self._on_readiness_poll,
      callback_group=self._cb_group,
    )

  def _on_readiness_poll(self) -> None:
    if not self._planning_armed or self._awaiting_confirm:
      return
    ready, detail = self._plan_readiness_status()
    if not ready:
      self._readiness_was_ready = False
      self.get_logger().info(
        f'Plan wait: {detail}',
        throttle_duration_sec=5.0,
      )
      return
    if self._readiness_was_ready:
      return
    self._readiness_was_ready = True
    self.get_logger().info('Plan prerequisites met — scheduling plan.')
    self._schedule_plan('readiness_poll', delay_sec=0.1)

  def _goals_ready_for_plan(self) -> bool:
    if not self._goals_ready():
      return False
    if self._planning_mode != 'dual':
      return True
    if not bool(self.get_parameter('require_back_wall_for_dual').value):
      return True
    if self._back_wall_cuboid() is not None:
      return True
    self.get_logger().warn(
      'Dual plan waiting: back wall missing from /obstacle_markers '
      '(run perception_click_planning and click L or R target)',
      throttle_duration_sec=5.0,
    )
    return False

  @staticmethod
  def _curobo_server_dim_scale() -> float:
    # Cuboid dims are already scaled in markers_to_curobo_cuboids(); server must not shrink again.
    return 1.0

  def _schedule_plan(self, reason: str, delay_sec: Optional[float] = None) -> None:
    if not self._planning_armed:
      return
    if delay_sec is None:
      delay_sec = float(self.get_parameter('plan_debounce_sec').value)
    self._pending_plan_reason = reason
    self._pending_plan_fire_mono = time.monotonic() + max(float(delay_sec), 0.05)
    if self._plan_debounce_timer is None:
      self._plan_debounce_timer = self.create_timer(0.05, self._on_plan_debounce_tick)

  def _on_plan_debounce_tick(self) -> None:
    if time.monotonic() < self._pending_plan_fire_mono:
      return
    if self._plan_debounce_timer is not None:
      self._plan_debounce_timer.cancel()
      self.destroy_timer(self._plan_debounce_timer)
      self._plan_debounce_timer = None
    reason = self._pending_plan_reason or 'debounced'
    self._pending_plan_reason = None
    if not self._goals_ready_for_plan():
      return
    self._plan_and_execute(reason=reason)

  def _log_plan_obstacles(self, arm_label: str) -> None:
    with self._obstacle_lock:
      obs = list(self._last_obstacle_cuboids)
    wall = [o for o in obs if str(o.get('name', '')).endswith(f'_{TARGET_BACK_WALL_ID}')]
    detail = f'count={len(obs)}'
    if wall:
      w = wall[0]
      dims = [round(float(d), 3) for d in w['dims']]
      half_x = float(dims[0]) * 0.5 if dims else 0.0
      near_x = float(w['pose'][0]) - half_x
      detail += (
        f', back_wall=yes near_x={near_x:.3f} center_x={float(w["pose"][0]):.3f}'
        f' dims={dims}'
      )
    else:
      detail += ', back_wall=MISSING (set left target first in click node)'
    self._plan_progress(f'obstacles ({arm_label})', detail)

  def _plan_pose_arm_trajectory(
    self,
    arm_label: str,
    state_names: List[str],
    state_positions: List[float],
    goal_xyz: tuple[float, float, float],
    goal_quat,
    trajectory_joint_names: List[str],
    robot_file: str,
  ) -> Optional[RobotTrajectory]:
    self._log_plan_obstacles(arm_label)
    payload = self._build_plan_payload(
      state_names,
      state_positions,
      goal_xyz,
      goal_quat,
      trajectory_joint_names,
      robot_file,
      plan_label=arm_label,
    )
    plan = self._call_curobo_plan(payload, arm_label)
    if plan is None:
      return None
    traj = self._plan_dict_to_trajectory(plan)
    if traj is not None:
      final = self._trajectory_final_positions(traj)
      goal_xyz = payload.get('target_pose', {}).get('position', [])
      preview = ', '.join(f'{k}={v:.3f}' for k, v in list(final.items())[:5])
      self._plan_progress(
        f'planned ({arm_label})',
        f'goal={[round(float(x), 3) for x in goal_xyz]}, final: {preview}',
      )
    return traj

  def _run_dual_left_then_right(self, reason: str) -> None:
    if not self._goals_ready():
      self._plan_progress('failed', f'goals not ready ({reason})')
      return
    if not self._joint_state_is_fresh():
      age = time.monotonic() - self._latest_joint_state_mono
      self._plan_progress('failed', f'/joint_states stale ({age:.2f}s)')
      return
    if (
      self._cached_goal_l_xyz is None
      or self._cached_goal_l_quat is None
      or self._cached_goal_r_xyz is None
      or self._cached_goal_r_quat is None
    ):
      self._plan_progress('failed', 'missing left or right goal')
      return
    try:
      state_names, state_positions = self._current_curobo_joint_state()
    except RuntimeError as e:
      self.get_logger().error(str(e))
      return

    left_joints = self._dual_left_trajectory_joints()
    left_label = 'plan left arm + lift' if self._dual_plan_lift_enabled() else (
      'plan left arm (lift fixed at start)'
    )
    self._plan_progress('dual sequence 1/2', left_label)
    traj_l = self._plan_pose_arm_trajectory(
      'left',
      state_names,
      state_positions,
      self._cached_goal_l_xyz,
      self._cached_goal_l_quat,
      left_joints,
      self._curobo_robot_file_l,
    )
    if traj_l is None:
      self._plan_progress('failed', 'left+lift planning failed')
      return
    if not self._confirm_and_execute_trajectory(traj_l, 'left+lift'):
      self._plan_progress('stopped', 'left+lift not executed — right arm skipped')
      return

    self._plan_progress(
      'dual sequence',
      'left+lift execute done — preparing right arm plan (step 2/2)',
    )
    if not self._wait_joint_states_after_move('after left+lift'):
      self._plan_progress('failed', 'no /joint_states after left+lift — right arm skipped')
      return

    try:
      state_names, state_positions = self._current_curobo_joint_state()
    except RuntimeError as e:
      self.get_logger().error(str(e))
      self._plan_progress('failed', f'right arm skipped: {e}')
      return
    lift = next(
      (p for n, p in zip(state_names, state_positions) if n == 'lift_joint'), None
    )
    self._plan_progress('dual sequence 2/2', f'planning right arm (lift_joint={lift})')
    traj_r = self._plan_pose_arm_trajectory(
      'right',
      state_names,
      state_positions,
      self._cached_goal_r_xyz,
      self._cached_goal_r_quat,
      TRAJECTORY_JOINTS_R,
      self._curobo_robot_file_r,
    )
    if traj_r is None:
      self._plan_progress('failed', 'right arm planning failed (see cuRobo server log)')
      return

    require_right_confirm = bool(self.get_parameter('dual_require_right_confirm').value)
    if not self._confirm_and_execute_trajectory(
      traj_r, 'right', require_confirm=require_right_confirm
    ):
      self._plan_progress('stopped', 'right arm not executed')

  def _obstacles_for_curobo_plan(
    self,
    goal_xyz: tuple[float, float, float],
    arm_label: str,
  ) -> List[dict]:
    if not bool(self.get_parameter('send_obstacles_to_curobo').value):
      return []
    with self._obstacle_lock:
      raw = list(self._last_obstacle_cuboids)
    clearance = float(self.get_parameter('curobo_goal_obstacle_clearance_m').value)
    filtered, dropped = filter_curobo_obstacles_for_goal(raw, goal_xyz, clearance)
    if dropped:
      self._plan_progress(
        f'obstacles filtered ({arm_label})',
        f'dropped {len(dropped)} near goal (clearance={clearance:.3f}m): '
        f'{", ".join(dropped[:4])}',
      )
    return filtered

  def _build_plan_payload(
    self,
    state_names: List[str],
    state_positions: List[float],
    xyz: tuple[float, float, float],
    quat,
    trajectory_joint_names: List[str],
    robot_file: str,
    plan_label: str = '',
  ) -> dict:
    send_obs = bool(self.get_parameter('send_obstacles_to_curobo').value)
    x, y, z = xyz
    obstacles = (
      self._obstacles_for_curobo_plan((x, y, z), plan_label or robot_file)
      if send_obs
      else []
    )
    return {
      'joint_names': state_names,
      'current_joint_positions': state_positions,
      'trajectory_joint_names': trajectory_joint_names,
      'robot_file': robot_file,
      'target_pose': {
        'position': [x, y, z],
        'quaternion': [float(quat.w), float(quat.x), float(quat.y), float(quat.z)],
      },
      'obstacles': obstacles,
      'obstacle_dim_scale': self._curobo_server_dim_scale(),
      'max_attempts': int(self.get_parameter('max_attempts').value),
      'skip_start_feasibility_check': self._resolve_skip_start_feasibility(),
      'orientation_mode': self._server_orientation_mode(),
      'orientation_weight': float(self.get_parameter('orientation_weight').value),
    }

  def _call_curobo_plan(self, payload: dict, arm_label: str) -> Optional[dict]:
    target = payload.get('target_pose', {}).get('position', [])
    robot = payload.get('robot_file', '?')
    n_obs = len(payload.get('obstacles') or [])
    arm = self._resolve_final_approach_arm(arm_label)
    self._motion_log(
      'cuRobo',
      arm,
      f'STAGE 1 planning HTTP /plan: robot={robot}, '
      f'pre-grasp target={[round(float(x), 3) for x in target]}, obstacles={n_obs}',
    )
    t0 = time.monotonic()
    try:
      response = requests.post(self._curobo_url, json=payload, timeout=60.0)
      response.raise_for_status()
      plan = response.json()
    except requests.RequestException as e:
      self.get_logger().error(f'cuRobo HTTP failed ({arm_label}): {e}')
      self._plan_progress(f'cuRobo /plan HTTP error ({arm_label})', str(e))
      return None
    elapsed = time.monotonic() - t0
    if not plan.get('success', False):
      self._log_plan_failure(plan.get('message', ''), payload)
      self._plan_progress(
        f'cuRobo /plan rejected ({arm_label})',
        f'{elapsed:.2f}s — {plan.get("message", "")}',
      )
      return None
    msg = str(plan.get('message', ''))
    msg_l = msg.lower()
    if n_obs > 0 and (
      'linear fallback' in msg_l
      or 'not obstacle-checked' in msg_l
      or 'sim lenient' in msg_l
    ):
      self.get_logger().error(
        f'Rejecting cuRobo plan ({arm_label}): unsafe path with obstacles — {msg}'
      )
      self._plan_progress(f'cuRobo /plan rejected ({arm_label})', f'unsafe fallback: {msg}')
      return None
    self._plan_progress(
      f'cuRobo /plan OK ({arm_label})',
      f'{elapsed:.2f}s, {self._plan_response_summary(plan)}, msg={plan.get("message", "")}',
    )
    return plan

  def _log_plan_failure(self, msg: str, payload: dict) -> None:
    msg_lower = msg.lower()
    send_obs = bool(payload.get('obstacles'))
    obs_scale = float(payload.get('obstacle_dim_scale', 1.0))
    if 'stopped short of goal' in msg_lower or 'position_error=' in msg_lower:
      self.get_logger().error(
        f'cuRobo planning failed: {msg} '
        '(move click obstacles away from target or raise curobo_goal_obstacle_clearance_m)'
      )
    elif 'ik=0/' in msg_lower or 'goal unreachable' in msg_lower:
      self.get_logger().error(
        f'cuRobo planning failed: {msg} '
        '(goal unreachable: check target xyz, lift height, EE orientation, obstacles at goal)'
      )
    elif 'trajopt' in msg_lower or 'collision-free path' in msg_lower:
      self.get_logger().error(
        f'cuRobo planning failed: {msg} '
        '(sim sphere self-collision along path; restart curobo_plan_server after updates)'
      )
    elif 'start configuration is in self-collision' in msg_lower:
      self.get_logger().error(
        f'cuRobo planning failed: {msg} '
        '(align sim home with cuRobo YAML or skip_start_feasibility_check:=true)'
      )
    elif send_obs and payload['obstacles']:
      obs_detail = ', '.join(
        f"{o.get('name', '?')} dims={[round(d, 3) for d in o.get('dims', [])]} "
        f"@[{o['pose'][0]:.2f},{o['pose'][1]:.2f},{o['pose'][2]:.2f}]"
        for o in payload['obstacles'][:3]
      )
      self.get_logger().error(
        f'cuRobo planning failed: {msg} (obstacles: {obs_detail}, dim_scale={obs_scale})'
      )
    else:
      self.get_logger().error(f'cuRobo planning failed: {msg}')

  @staticmethod
  def _traj_point_time_sec(pt: JointTrajectoryPoint) -> float:
    return float(pt.time_from_start.sec) + float(pt.time_from_start.nanosec) * 1e-9

  def _sample_joint_trajectory(
    self, jt, t: float
  ) -> Dict[str, float]:
    if not jt.points:
      return {}
    times = [self._traj_point_time_sec(p) for p in jt.points]
    if t <= times[0]:
      return dict(zip(jt.joint_names, [float(v) for v in jt.points[0].positions]))
    if t >= times[-1]:
      last = jt.points[-1]
      return dict(zip(jt.joint_names, [float(v) for v in last.positions]))
    for i in range(1, len(times)):
      if t <= times[i]:
        t0, t1 = times[i - 1], times[i]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        p0 = jt.points[i - 1].positions
        p1 = jt.points[i].positions
        pos = [float(a * (1.0 - alpha) + b * alpha) for a, b in zip(p0, p1)]
        return dict(zip(jt.joint_names, pos))
    last = jt.points[-1]
    return dict(zip(jt.joint_names, [float(v) for v in last.positions]))

  @staticmethod
  def _trajectory_final_positions(traj: RobotTrajectory) -> Dict[str, float]:
    jt = traj.joint_trajectory
    if not jt.points:
      return {}
    last = jt.points[-1]
    return {
      name: float(pos)
      for name, pos in zip(jt.joint_names, last.positions)
    }

  @staticmethod
  def _apply_joint_overrides(
    state_names: List[str],
    state_positions: List[float],
    overrides: Dict[str, float],
  ) -> List[float]:
    if not overrides:
      return list(state_positions)
    out = list(state_positions)
    for i, jn in enumerate(state_names):
      if jn in overrides:
        out[i] = float(overrides[jn])
    return out

  def _merge_dual_planned_sequence(
    self,
    traj_l: RobotTrajectory,
    traj_r: RobotTrajectory,
    arm_r_hold: Dict[str, float],
    static_hold: Optional[Dict[str, float]] = None,
  ) -> Optional[RobotTrajectory]:
    """Left arm first, then right arm (right holds still while left moves)."""
    static_hold = static_hold or {}
    jt_l = traj_l.joint_trajectory
    jt_r = traj_r.joint_trajectory
    if not jt_l.points or not jt_r.points:
      self.get_logger().error('Cannot merge dual sequence: empty trajectory')
      return None
    T_l = self._traj_point_time_sec(jt_l.points[-1])
    T_r = self._traj_point_time_sec(jt_r.points[-1])
    times = sorted(
      {self._traj_point_time_sec(p) for p in jt_l.points}
      | {T_l + self._traj_point_time_sec(p) for p in jt_r.points}
    )
    T_end = T_l + T_r
    if not times or times[-1] < T_end:
      times.append(T_end)

    merged_names = list(TRAJECTORY_JOINTS_DUAL)
    left_final = self._sample_joint_trajectory(jt_l, T_l)

    def _joint_position(
      joint_name: str,
      sampled: Dict[str, float],
    ) -> float:
      if joint_name in sampled:
        return float(sampled[joint_name])
      if joint_name in static_hold:
        return float(static_hold[joint_name])
      raise KeyError(joint_name)

    merged = RobotTrajectory()
    merged.joint_trajectory.joint_names = merged_names
    for t in times:
      pt = JointTrajectoryPoint()
      positions: List[float] = []
      if t <= T_l + 1e-9:
        left_at_t = self._sample_joint_trajectory(jt_l, min(t, T_l))
        for n in merged_names:
          if n in ARM_R_JOINTS:
            positions.append(float(arm_r_hold[n]))
          else:
            try:
              positions.append(_joint_position(n, left_at_t))
            except KeyError:
              self.get_logger().error(
                f'Cannot merge dual sequence: joint {n!r} missing from left traj '
                f'and static_hold'
              )
              return None
      else:
        right_at_t = self._sample_joint_trajectory(jt_r, min(t - T_l, T_r))
        for n in merged_names:
          if n in ARM_R_JOINTS:
            positions.append(float(right_at_t[n]))
          else:
            try:
              positions.append(_joint_position(n, left_final))
            except KeyError:
              self.get_logger().error(
                f'Cannot merge dual sequence: joint {n!r} missing from left final '
                f'and static_hold'
              )
              return None
      pt.positions = positions
      pt.time_from_start.sec = int(t)
      pt.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
      merged.joint_trajectory.points.append(pt)
    return merged

  @staticmethod
  def _clone_robot_trajectory(traj: RobotTrajectory) -> RobotTrajectory:
    return copy.deepcopy(traj)

  def _extract_single_joint_trajectory(
    self, traj: RobotTrajectory, joint_name: str
  ) -> Optional[RobotTrajectory]:
    jt = traj.joint_trajectory
    if joint_name not in jt.joint_names or not jt.points:
      return None
    idx = jt.joint_names.index(joint_name)
    out = RobotTrajectory()
    out.joint_trajectory.joint_names = [joint_name]
    for src in jt.points:
      pt = JointTrajectoryPoint()
      pt.positions = [float(src.positions[idx])]
      pt.time_from_start = src.time_from_start
      out.joint_trajectory.points.append(pt)
    return out

  def _extract_trajectory_segment(
    self,
    traj: RobotTrajectory,
    joint_names: List[str],
    t_start: float,
    t_end: float,
    *,
    constant_joints: Optional[Dict[str, float]] = None,
    rebase_time: bool = True,
  ) -> Optional[RobotTrajectory]:
    jt = traj.joint_trajectory
    if not jt.points:
      return None
    constant_joints = constant_joints or {}
    times = sorted(
      {
        self._traj_point_time_sec(p)
        for p in jt.points
        if t_start - 1e-9 <= self._traj_point_time_sec(p) <= t_end + 1e-9
      }
    )
    if not times:
      times = [t_end]
    out = RobotTrajectory()
    out.joint_trajectory.joint_names = list(joint_names)
    for t in times:
      sampled = self._sample_joint_trajectory(jt, t)
      pt = JointTrajectoryPoint()
      positions: List[float] = []
      for n in joint_names:
        if n in constant_joints:
          positions.append(float(constant_joints[n]))
        elif n in sampled:
          positions.append(float(sampled[n]))
        else:
          self.get_logger().error(
            f'Trajectory segment missing joint {n!r} at t={t:.3f}s'
          )
          return None
      pt.positions = positions
      rel_t = (t - t_start) if rebase_time else t
      pt.time_from_start.sec = int(rel_t)
      pt.time_from_start.nanosec = int(round((rel_t - int(rel_t)) * 1e9))
      out.joint_trajectory.points.append(pt)
    return out

  def _make_lift_only_trajectory(
    self,
    start_lift: float,
    goal_lift: float,
    duration_sec: float = 3.0,
  ) -> RobotTrajectory:
    """Simple lift_joint-only trajectory for execute step 1."""
    dur = max(float(duration_sec), 0.5)
    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = ['lift_joint']
    for t, lift in ((0.0, start_lift), (dur, goal_lift)):
      pt = JointTrajectoryPoint()
      pt.positions = [float(lift)]
      pt.time_from_start.sec = int(t)
      pt.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
      traj.joint_trajectory.points.append(pt)
    return traj

  def _compute_dual_lift_midpoint_execute_trajectories(
    self,
    state_names: List[str],
    state_positions: List[float],
  ) -> Optional[tuple[RobotTrajectory, RobotTrajectory, RobotTrajectory]]:
    if (
      self._cached_goal_l_xyz is None
      or self._cached_goal_l_quat is None
      or self._cached_goal_r_xyz is None
      or self._cached_goal_r_quat is None
    ):
      return None

    current_lift = next(
      (float(p) for n, p in zip(state_names, state_positions) if n == 'lift_joint'),
      None,
    )
    if current_lift is None:
      self.get_logger().error('lift_midpoint: lift_joint missing from start state')
      return None

    self._plan_progress(
      'dual lift_midpoint 1/5',
      'probe left arm + lift → left target (read optimal lift_l)',
    )
    traj_l_probe = self._plan_pose_arm_trajectory(
      'left+lift probe',
      state_names,
      state_positions,
      self._cached_goal_l_xyz,
      self._cached_goal_l_quat,
      TRAJECTORY_JOINTS_L,
      self._curobo_robot_file_l,
    )
    if traj_l_probe is None:
      self._plan_progress('failed', 'lift_midpoint 1/5: left+lift probe failed')
      return None
    lift_l = self._trajectory_final_positions(traj_l_probe).get('lift_joint')
    if lift_l is None:
      self.get_logger().error('lift_midpoint 1/5: lift_joint missing from left probe')
      return None

    self._plan_progress(
      'dual lift_midpoint 2/5',
      'probe right arm + lift → right target (read optimal lift_r)',
    )
    traj_r_probe = self._plan_pose_arm_trajectory(
      'right+lift probe',
      state_names,
      state_positions,
      self._cached_goal_r_xyz,
      self._cached_goal_r_quat,
      TRAJECTORY_JOINTS_R_LIFT,
      self._curobo_robot_file_r,
    )
    if traj_r_probe is None:
      self._plan_progress('failed', 'lift_midpoint 2/5: right+lift probe failed')
      return None
    lift_r = self._trajectory_final_positions(traj_r_probe).get('lift_joint')
    if lift_r is None:
      self.get_logger().error('lift_midpoint 2/5: lift_joint missing from right probe')
      return None

    shared_lift = 0.5 * (float(lift_l) + float(lift_r))
    self._plan_progress(
      'dual lift_midpoint 3/5',
      f'shared lift={(shared_lift):.3f} m (midpoint of lift_l={float(lift_l):.3f}, '
      f'lift_r={float(lift_r):.3f}); current={current_lift:.3f}',
    )

    state_at_shared = self._apply_joint_overrides(
      state_names, state_positions, {'lift_joint': shared_lift}
    )

    self._plan_progress(
      'dual lift_midpoint 4/5',
      f'replan left arm @ shared lift={shared_lift:.3f}',
    )
    traj_l = self._plan_pose_arm_trajectory(
      'left @ shared lift',
      state_names,
      state_at_shared,
      self._cached_goal_l_xyz,
      self._cached_goal_l_quat,
      TRAJECTORY_JOINTS_L_DUAL,
      self._curobo_robot_file_l,
    )
    if traj_l is None:
      self._plan_progress(
        'failed',
        f'lift_midpoint 4/5: left replan failed @ lift={shared_lift:.3f} '
        f'(lift_l={float(lift_l):.3f}, lift_r={float(lift_r):.3f})',
      )
      return None

    self._plan_progress(
      'dual lift_midpoint 5/5',
      f'replan right arm @ shared lift={shared_lift:.3f}',
    )
    traj_r = self._plan_pose_arm_trajectory(
      'right @ shared lift',
      state_names,
      state_at_shared,
      self._cached_goal_r_xyz,
      self._cached_goal_r_quat,
      TRAJECTORY_JOINTS_R,
      self._curobo_robot_file_r,
    )
    if traj_r is None:
      self._plan_progress(
        'failed',
        f'lift_midpoint 5/5: right replan failed @ lift={shared_lift:.3f} '
        f'(lift_l={float(lift_l):.3f}, lift_r={float(lift_r):.3f})',
      )
      return None

    traj_lift = self._make_lift_only_trajectory(current_lift, shared_lift)
    traj_left = self._extract_trajectory_segment(
      traj_l,
      list(TRAJECTORY_JOINTS_L_DUAL),
      0.0,
      self._traj_point_time_sec(traj_l.joint_trajectory.points[-1]),
    )
    traj_right = self._clone_robot_trajectory(traj_r)
    if traj_left is None:
      return None

    t_l = self._traj_point_time_sec(traj_l.joint_trajectory.points[-1])
    t_r = self._traj_point_time_sec(traj_r.joint_trajectory.points[-1])
    self._plan_progress(
      'dual lift_midpoint planned',
      f'after confirm: lift {current_lift:.3f}→{shared_lift:.3f} → '
      f'left ({t_l:.2f}s) → right ({t_r:.2f}s)',
    )
    return traj_lift, traj_left, traj_right

  def _run_dual_lift_midpoint(self, reason: str) -> None:
    if not self._goals_ready_for_plan():
      self._plan_progress('failed', f'goals/back wall not ready ({reason})')
      return
    if not self._joint_state_is_fresh():
      age = time.monotonic() - self._latest_joint_state_mono
      self._plan_progress('failed', f'/joint_states stale ({age:.2f}s)')
      return
    try:
      state_names, state_positions = self._current_curobo_joint_state()
    except RuntimeError as e:
      self.get_logger().error(str(e))
      return

    planned = self._compute_dual_lift_midpoint_execute_trajectories(
      state_names, state_positions
    )
    if planned is None:
      self._plan_progress('failed', 'lift_midpoint planning failed')
      return

    traj_lift, traj_left, traj_right = planned
    self._confirm_and_execute_trajectory_sequence(
      [
        (traj_lift, f'lift → shared'),
        (traj_left, 'left arm'),
        (traj_right, 'right arm'),
      ],
      exec_label='dual lift_midpoint',
    )

  def _compute_dual_plan_merged_trajectory(
    self,
    state_names: List[str],
    state_positions: List[float],
  ) -> Optional[RobotTrajectory]:
    if (
      self._cached_goal_l_xyz is None
      or self._cached_goal_l_quat is None
      or self._cached_goal_r_xyz is None
      or self._cached_goal_r_quat is None
    ):
      return None

    left_joints = self._dual_left_trajectory_joints()
    left_label = 'left arm + lift' if self._dual_plan_lift_enabled() else 'left arm (lift fixed)'
    self._plan_progress('dual plan 1/2', left_label)
    traj_l = self._plan_pose_arm_trajectory(
      'left',
      state_names,
      state_positions,
      self._cached_goal_l_xyz,
      self._cached_goal_l_quat,
      left_joints,
      self._curobo_robot_file_l,
    )
    if traj_l is None:
      return None

    left_final = self._trajectory_final_positions(traj_l)
    shared_override = {
      jn: left_final[jn] for jn in left_joints if jn in left_final
    }
    state_positions_r = self._apply_joint_overrides(
      state_names, state_positions, shared_override
    )
    name_to_pos = dict(zip(state_names, state_positions))
    try:
      arm_r_hold = {jn: float(name_to_pos[jn]) for jn in ARM_R_JOINTS}
    except KeyError as e:
      self.get_logger().error(f'Missing arm_r joint in start state: {e}')
      return None

    lift_goal = shared_override.get('lift_joint')
    lift_start = next(
      (p for n, p in zip(state_names, state_positions) if n == 'lift_joint'), None
    )
    if self._dual_plan_lift_enabled():
      right_ctx = f'left+lift goal, lift_joint={lift_goal}'
    else:
      right_ctx = f'left goal, lift fixed at start ({lift_start})'
    self._plan_progress('dual plan 2/2', f'right arm (cuRobo start with {right_ctx})')
    traj_r = self._plan_pose_arm_trajectory(
      'right',
      state_names,
      state_positions_r,
      self._cached_goal_r_xyz,
      self._cached_goal_r_quat,
      TRAJECTORY_JOINTS_R,
      self._curobo_robot_file_r,
    )
    if traj_r is None:
      return None

    dur_l = self._traj_point_time_sec(traj_l.joint_trajectory.points[-1])
    dur_r = self._traj_point_time_sec(traj_r.joint_trajectory.points[-1])
    static_hold: Dict[str, float] = {}
    if not self._dual_plan_lift_enabled() and lift_start is not None:
      static_hold['lift_joint'] = float(lift_start)
    traj = self._merge_dual_planned_sequence(
      traj_l, traj_r, arm_r_hold, static_hold=static_hold
    )
    if traj is not None:
      self._plan_progress(
        'dual plan merged',
        f'RViz sequence: left+lift {dur_l:.2f}s → right {dur_r:.2f}s '
        f'(total≈{dur_l + dur_r:.2f}s), then one confirm_execute',
      )
    return traj

  def _merge_robot_trajectories(
    self, traj_a: RobotTrajectory, traj_b: RobotTrajectory
  ) -> Optional[RobotTrajectory]:
    jt_a = traj_a.joint_trajectory
    jt_b = traj_b.joint_trajectory
    if not jt_a.points or not jt_b.points:
      self.get_logger().error('Cannot merge: empty arm trajectory')
      return None
    times = sorted(
      set(self._traj_point_time_sec(p) for p in jt_a.points)
      | set(self._traj_point_time_sec(p) for p in jt_b.points)
    )
    end_t = max(times[-1], self._traj_point_time_sec(jt_a.points[-1]),
                self._traj_point_time_sec(jt_b.points[-1]))
    if times[-1] < end_t:
      times.append(end_t)
    merged_names = list(jt_a.joint_names)
    for n in jt_b.joint_names:
      if n not in merged_names:
        merged_names.append(n)
    merged = RobotTrajectory()
    merged.joint_trajectory.joint_names = merged_names
    for t in times:
      sa = self._sample_joint_trajectory(jt_a, t)
      sb = self._sample_joint_trajectory(jt_b, t)
      pt = JointTrajectoryPoint()
      pt.positions = [float(sa[n] if n in sa else sb[n]) for n in merged_names]
      pt.time_from_start.sec = int(t)
      pt.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
      merged.joint_trajectory.points.append(pt)
    return merged

  def _compute_pose_plan_trajectory(self, reason: str) -> Optional[RobotTrajectory]:
    if not self._goals_ready():
      self.get_logger().warn(f'Skip plan ({reason}): goal(s) not ready')
      return None
    if not self._joint_state_is_fresh():
      age = time.monotonic() - self._latest_joint_state_mono
      self.get_logger().warn(
        f'Skip plan ({reason}): /joint_states stale ({age:.2f}s old).'
      )
      return None
    try:
      state_names, state_positions = self._current_curobo_joint_state()
    except RuntimeError as e:
      self.get_logger().error(str(e))
      return None

    lift = next(
      (p for n, p in zip(state_names, state_positions) if n == 'lift_joint'),
      None,
    )
    skip = self._resolve_skip_start_feasibility()
    self._plan_progress(
      'prepare',
      f'reason={reason}, mode={self._planning_mode}, state_joints={len(state_names)}, '
      f'lift={lift}, skip_start_check={skip}',
    )

    traj: Optional[RobotTrajectory] = None
    if self._planning_mode == 'left':
      if self._cached_goal_l_xyz is None or self._cached_goal_l_quat is None:
        return None
      payload = self._build_plan_payload(
        state_names,
        state_positions,
        self._cached_goal_l_xyz,
        self._cached_goal_l_quat,
        TRAJECTORY_JOINTS_L,
        self._curobo_robot_file_l,
      )
      plan = self._call_curobo_plan(payload, 'left')
      if plan is None:
        return None
      traj = self._plan_dict_to_trajectory(plan)
    elif self._planning_mode == 'right':
      if self._cached_goal_r_xyz is None or self._cached_goal_r_quat is None:
        return None
      payload = self._build_plan_payload(
        state_names,
        state_positions,
        self._cached_goal_r_xyz,
        self._cached_goal_r_quat,
        TRAJECTORY_JOINTS_R,
        self._curobo_robot_file_r,
      )
      plan = self._call_curobo_plan(payload, 'right')
      if plan is None:
        return None
      traj = self._plan_dict_to_trajectory(plan)
    else:
      dual_mode = self._dual_sequence_mode()
      if dual_mode == 'plan_merged':
        traj = self._compute_dual_plan_merged_trajectory(state_names, state_positions)
      elif dual_mode == 'parallel':
        if (
          self._cached_goal_l_xyz is None
          or self._cached_goal_l_quat is None
          or self._cached_goal_r_xyz is None
          or self._cached_goal_r_quat is None
        ):
          return None
        payload_l = self._build_plan_payload(
          state_names,
          state_positions,
          self._cached_goal_l_xyz,
          self._cached_goal_l_quat,
          self._dual_left_trajectory_joints(),
          self._curobo_robot_file_l,
        )
        left_parallel_label = (
          'left arm + lift' if self._dual_plan_lift_enabled() else 'left arm (lift fixed at start)'
        )
        self._plan_progress('dual parallel 1/2', left_parallel_label)
        plan_l = self._call_curobo_plan(payload_l, 'left')
        if plan_l is None:
          return None
        traj_l = self._plan_dict_to_trajectory(plan_l)
        if traj_l is None:
          return None

        self._plan_progress('dual parallel 2/2', 'right arm')
        payload_r = self._build_plan_payload(
          state_names,
          state_positions,
          self._cached_goal_r_xyz,
          self._cached_goal_r_quat,
          TRAJECTORY_JOINTS_R,
          self._curobo_robot_file_r,
        )
        plan_r = self._call_curobo_plan(payload_r, 'right')
        if plan_r is None:
          return None
        traj_r = self._plan_dict_to_trajectory(plan_r)
        if traj_r is None:
          return None

        dur_l = self._traj_point_time_sec(traj_l.joint_trajectory.points[-1])
        dur_r = self._traj_point_time_sec(traj_r.joint_trajectory.points[-1])
        traj = self._merge_robot_trajectories(traj_l, traj_r)
        if traj is not None:
          self._plan_progress(
            'dual parallel merged',
            f'joints={traj.joint_trajectory.joint_names}, '
            f'duration_l={dur_l:.2f}s, duration_r={dur_r:.2f}s',
          )
      else:
        self.get_logger().warn(
          f'dual_sequence={dual_mode!r} uses execute-between worker; '
          'falling back to plan_merged in compute path'
        )
        traj = self._compute_dual_plan_merged_trajectory(state_names, state_positions)

    if traj is None:
      return None
    return traj

  def _home_trajectory_joints(self) -> List[str]:
    if self._planning_mode == 'right':
      return list(TRAJECTORY_JOINTS_R)
    if self._planning_mode == 'dual':
      return list(TRAJECTORY_JOINTS_DUAL)
    return list(TRAJECTORY_JOINTS_L)

  def _home_robot_file(self) -> str:
    if self._planning_mode == 'right':
      return self._curobo_robot_file_r
    return self._curobo_robot_file_l

  def _home_joint_goal(self) -> tuple[List[str], List[float]]:
    joint_names = self._home_trajectory_joints()
    positions = [float(ROBOT_INITIAL_JOINTS[jn]) for jn in joint_names]
    return joint_names, positions

  def _build_plan_joint_payload(
    self,
    state_names: List[str],
    state_positions: List[float],
    goal_joint_names: List[str],
    goal_joint_positions: List[float],
    trajectory_joint_names: List[str],
    robot_file: str,
  ) -> dict:
    send_obs = bool(self.get_parameter('send_obstacles_to_curobo').value)
    return {
      'joint_names': state_names,
      'current_joint_positions': state_positions,
      'goal_joint_names': goal_joint_names,
      'goal_joint_positions': goal_joint_positions,
      'trajectory_joint_names': trajectory_joint_names,
      'robot_file': robot_file,
      'obstacles': self._last_obstacle_cuboids if send_obs else [],
      'obstacle_dim_scale': self._curobo_server_dim_scale(),
      'max_attempts': int(self.get_parameter('max_attempts').value),
      'skip_start_feasibility_check': self._resolve_skip_start_feasibility(),
    }

  def _call_curobo_plan_joint(self, payload: dict) -> Optional[dict]:
    robot = payload.get('robot_file', '?')
    goal_joints = payload.get('goal_joint_names') or []
    self._plan_progress(
      'cuRobo /plan_joint request',
      f'robot={robot}, goal_joints={goal_joints}',
    )
    t0 = time.monotonic()
    try:
      response = requests.post(self._curobo_plan_joint_url, json=payload, timeout=60.0)
      response.raise_for_status()
      plan = response.json()
    except requests.RequestException as e:
      self.get_logger().error(f'cuRobo joint plan HTTP failed: {e}')
      self._plan_progress('cuRobo /plan_joint HTTP error', str(e))
      return None
    elapsed = time.monotonic() - t0
    if not plan.get('success', False):
      self.get_logger().error(
        f'cuRobo joint planning failed: {plan.get("message", "")}'
      )
      self._plan_progress(
        'cuRobo /plan_joint rejected',
        f'{elapsed:.2f}s — {plan.get("message", "")}',
      )
      return None
    self._plan_progress(
      'cuRobo /plan_joint OK',
      f'{elapsed:.2f}s, {self._plan_response_summary(plan)}',
    )
    return plan

  def _joint_home_config_candidates(self) -> List[str]:
    candidates: List[str] = []
    param_path = str(self.get_parameter('joint_home_config').value).strip()
    if param_path:
      candidates.append(param_path)
    candidates.append(
      os.path.join(_SCRIPT_DIR.parent, 'config', 'robot_joint_home.yaml')
    )
    try:
      candidates.append(
        os.path.join(
          get_package_share_directory('ffw_moveit_config'),
          'config',
          'robot_joint_home.yaml',
        )
      )
    except Exception:
      pass
    seen: set[str] = set()
    unique: List[str] = []
    for path in candidates:
      norm = os.path.normpath(path)
      if norm not in seen:
        seen.add(norm)
        unique.append(norm)
    return unique

  def _parse_joint_home_groups(self, groups: List[dict], source: str) -> List[dict]:
    valid: List[dict] = []
    for i, group in enumerate(groups):
      joint_names = list(group.get('joint_names') or [])
      positions = [float(x) for x in (group.get('positions') or [])]
      action_topic = str(group.get('action_topic') or '').strip()
      if not joint_names or not positions or not action_topic:
        self.get_logger().warn(f'joint_home group {i}: missing fields, skipping')
        continue
      if len(joint_names) != len(positions):
        self.get_logger().error(
          f'joint_home group {group.get("name", i)}: '
          f'{len(joint_names)} joints vs {len(positions)} positions'
        )
        continue
      valid.append({
        'name': str(group.get('name') or action_topic),
        'joint_names': joint_names,
        'positions': positions,
        'action_topic': action_topic,
        'duration': float(group.get('duration', 5.0)),
      })
    if valid:
      self.get_logger().info(f'Loaded joint_home config: {source} ({len(valid)} groups)')
    return valid

  def _load_joint_home_groups(self) -> List[dict]:
    for config_path in self._joint_home_config_candidates():
      try:
        with open(config_path, 'r', encoding='utf-8') as f:
          data = yaml.safe_load(f) or {}
      except OSError:
        continue
      except yaml.YAMLError as e:
        self.get_logger().error(f'joint_home config parse error ({config_path}): {e}')
        continue
      valid = self._parse_joint_home_groups(data.get('groups') or [], config_path)
      if valid:
        return valid

    tried = ', '.join(self._joint_home_config_candidates())
    self.get_logger().warn(
      f'joint_home YAML not found or empty ({tried}); using built-in defaults.'
    )
    return [dict(group) for group in DEFAULT_JOINT_HOME_GROUPS]

  def _on_joint_home(self, _msg: Empty) -> None:
    if self._awaiting_confirm:
      self._confirm_session_id += 1
      self._end_confirm_session()
    if not self._joint_home_lock.acquire(blocking=False):
      self.get_logger().warn('joint_home already in progress; skipping.')
      return

    def _run() -> None:
      try:
        ok = self._execute_joint_home_groups()
        if ok:
          self.get_logger().info('joint_home: all groups completed.')
        else:
          self.get_logger().error('joint_home: one or more groups failed.')
      finally:
        self._joint_home_lock.release()

    threading.Thread(target=_run, daemon=True).start()

  def _create_quintic_joint_trajectory(
    self,
    joint_names: List[str],
    start_pos: List[float],
    end_pos: List[float],
    duration: float,
    num_points: int = 100,
  ) -> JointTrajectory:
    traj = JointTrajectory()
    traj.joint_names = list(joint_names)
    duration = max(float(duration), 0.1)
    for i in range(num_points):
      t = duration * i / max(num_points - 1, 1)
      t_norm = t / duration
      t_norm2 = t_norm * t_norm
      t_norm3 = t_norm2 * t_norm
      t_norm4 = t_norm3 * t_norm
      t_norm5 = t_norm4 * t_norm
      pos_coeff = 10.0 * t_norm3 - 15.0 * t_norm4 + 6.0 * t_norm5
      vel_coeff = (30.0 * t_norm2 - 60.0 * t_norm3 + 30.0 * t_norm4) / duration
      acc_coeff = (60.0 * t_norm - 180.0 * t_norm2 + 120.0 * t_norm3) / (
        duration * duration
      )

      point = JointTrajectoryPoint()
      positions: List[float] = []
      velocities: List[float] = []
      accelerations: List[float] = []
      for j in range(len(joint_names)):
        delta = float(end_pos[j]) - float(start_pos[j])
        positions.append(float(start_pos[j]) + delta * pos_coeff)
        velocities.append(delta * vel_coeff)
        accelerations.append(delta * acc_coeff)
      point.positions = positions
      point.velocities = velocities
      point.accelerations = accelerations
      point.time_from_start.sec = int(t)
      point.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
      traj.points.append(point)
    return traj

  def _execute_joint_home_groups(self) -> bool:
    joint_home_groups = self._load_joint_home_groups()
    if not joint_home_groups:
      self.get_logger().error('joint_home: no groups configured.')
      return False
    if not self._joint_state_is_fresh():
      age = time.monotonic() - self._latest_joint_state_mono
      self.get_logger().warn(f'Skip joint_home: /joint_states stale ({age:.2f}s old).')
      return False

    pending: List[tuple[str, object]] = []
    for group in joint_home_groups:
      joint_names = group['joint_names']
      start_pos = self._current_trajectory_positions(joint_names)
      if start_pos is None:
        self.get_logger().error(
          f'joint_home: missing /joint_states for {group["name"]} ({joint_names})'
        )
        return False

      client = ActionClient(
        self,
        FollowJointTrajectory,
        group['action_topic'],
        callback_group=self._cb_group,
      )
      if not client.wait_for_server(timeout_sec=2.0):
        self.get_logger().warn(
          f'joint_home: skipping {group["name"]} — '
          f'action server not available: {group["action_topic"]}'
        )
        continue

      vscale = max(self._resolve_velocity_scale(), 0.05)
      home_duration = min(float(group['duration']) / vscale, 45.0)
      traj = self._create_quintic_joint_trajectory(
        joint_names,
        start_pos,
        group['positions'],
        home_duration,
      )
      traj.header.stamp = self.get_clock().now().to_msg()
      goal = FollowJointTrajectory.Goal()
      goal.trajectory = traj

      self.get_logger().info(
        f'joint_home: sending {group["name"]} -> {group["action_topic"]} '
        f'({len(joint_names)} joints, {home_duration:.1f}s)'
      )
      send_future = client.send_goal_async(goal)
      if not self._wait_future_done(send_future, timeout_sec=10.0):
        return False
      goal_handle = send_future.result()
      if goal_handle is None or not goal_handle.accepted:
        self.get_logger().error(f'joint_home: goal rejected for {group["name"]}')
        return False
      pending.append((group['name'], goal_handle.get_result_async()))

    if not pending:
      self.get_logger().error('joint_home: no action servers available.')
      return False

    all_ok = True
    for name, result_future in pending:
      if not self._wait_future_done(result_future, timeout_sec=180.0):
        all_ok = False
        continue
      wrapped = result_future.result()
      if wrapped is None:
        self.get_logger().error(f'joint_home: no result for {name}')
        all_ok = False
        continue
      error_code = int(wrapped.result.error_code)
      if error_code != 0:
        self.get_logger().error(f'joint_home: {name} failed (error_code={error_code})')
        all_ok = False
      else:
        self.get_logger().info(f'joint_home: {name} done')
    return all_ok

  def _on_plan_home(self, _msg: Empty) -> None:
    self._start_plan_worker(
      self._compute_home_trajectory,
      exec_label='cuRobo home',
      skip_label='plan_home',
    )

  def _compute_home_trajectory(self) -> Optional[RobotTrajectory]:
    if not self._joint_state_is_fresh():
      age = time.monotonic() - self._latest_joint_state_mono
      self.get_logger().warn(
        f'Skip plan_home: /joint_states stale ({age:.2f}s old).'
      )
      return None
    try:
      state_names, state_positions = self._current_curobo_joint_state()
    except RuntimeError as e:
      self.get_logger().error(str(e))
      return None

    goal_joint_names, goal_joint_positions = self._home_joint_goal()
    trajectory_joint_names = self._home_trajectory_joints()
    robot_file = self._home_robot_file()
    self.get_logger().info(
      f'cuRobo plan_home: mode={self._planning_mode}, '
      f'joints={trajectory_joint_names}, robot_file={robot_file}'
    )

    payload = self._build_plan_joint_payload(
      state_names,
      state_positions,
      goal_joint_names,
      goal_joint_positions,
      trajectory_joint_names,
      robot_file,
    )
    plan = self._call_curobo_plan_joint(payload)
    if plan is None:
      return None
    return self._plan_dict_to_trajectory(plan)

  @staticmethod
  def _quat_from_wxyz_tuple(wxyz: tuple[float, float, float, float]) -> Quaternion:
    w, x, y, z = wxyz
    q = Quaternion()
    q.w, q.x, q.y, q.z = float(w), float(x), float(y), float(z)
    return q

  def _arm_for_final_approach_step(self, step_label: str) -> Optional[str]:
    return self._resolve_final_approach_arm(step_label)

  def _wait_before_final_approach(self, arm: str, step_label: str) -> None:
    settle = max(float(self.get_parameter('final_approach_settle_sec').value), 0.0)
    if settle <= 0.0:
      return
    self._motion_log(
      'MoveIt',
      arm,
      f'STAGE 2 pause {settle:.1f}s after cuRobo ({step_label or "coarse"}) — '
      'watch robot hold pre-grasp before contact move',
    )
    time.sleep(settle)

  def _lookup_ee_xyz(self, ee_link: str) -> Optional[tuple[float, float, float]]:
    timeout = max(float(self.get_parameter('final_approach_tf_timeout_sec').value), 0.1)
    try:
      tfm = self._tf_buffer.lookup_transform(
        self._planning_frame,
        ee_link,
        rclpy.time.Time(),
        timeout=Duration(seconds=timeout),
      )
      t = tfm.transform.translation
      return float(t.x), float(t.y), float(t.z)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
      self.get_logger().warn(
        f'EE TF lookup failed ({self._planning_frame} <- {ee_link}): {e}',
        throttle_duration_sec=5.0,
      )
      return None

  def _contact_pose_for_arm(
    self, arm: str
  ) -> tuple[Optional[tuple[float, float, float]], Optional[Quaternion]]:
    if arm == 'l':
      return self._cached_contact_l_xyz, self._cached_contact_l_quat
    return self._cached_contact_r_xyz, self._cached_contact_r_quat

  def _contact_pose_goal_constraints(
    self,
    contact_xyz: tuple[float, float, float],
    quat: Quaternion,
    ee_link: str,
  ) -> Constraints:
    tol = max(float(self.get_parameter('final_approach_goal_tolerance_m').value), 0.001)
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [tol]

    target = Pose()
    target.position.x = float(contact_xyz[0])
    target.position.y = float(contact_xyz[1])
    target.position.z = float(contact_xyz[2])
    target.orientation = quat

    bv = BoundingVolume()
    bv.primitives.append(sphere)
    bv.primitive_poses.append(target)

    pc = PositionConstraint()
    pc.header.frame_id = self._planning_frame
    pc.header.stamp = self.get_clock().now().to_msg()
    pc.link_name = ee_link
    pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
    pc.constraint_region = bv
    pc.weight = 1.0

    oc = OrientationConstraint()
    oc.header.frame_id = self._planning_frame
    oc.header.stamp = self.get_clock().now().to_msg()
    oc.link_name = ee_link
    oc.orientation = quat
    oc.absolute_x_axis_tolerance = 0.12
    oc.absolute_y_axis_tolerance = 0.12
    oc.absolute_z_axis_tolerance = 0.12
    oc.weight = 1.0

    goal_c = Constraints()
    goal_c.position_constraints.append(pc)
    goal_c.orientation_constraints.append(oc)
    return goal_c

  def _call_move_group_plan(
    self,
    request: MotionPlanRequest,
    arm: str,
    *,
    preview: bool = False,
  ) -> Optional[RobotTrajectory]:
    if not self._move_group.server_is_ready():
      return None
    goal = MoveGroup.Goal()
    goal.request = request
    goal.planning_options = PlanningOptions()
    goal.planning_options.plan_only = True
    goal.planning_options.look_around = False
    goal.planning_options.replan = False

    send_future = self._move_group.send_goal_async(goal)
    if not self._wait_future_done(send_future, timeout_sec=30.0):
      return None
    gh = send_future.result()
    if gh is None or not gh.accepted:
      if not preview:
        self._motion_log('MoveIt', arm, 'STAGE 2 MoveGroup goal rejected', level='error')
      return None

    result_future = gh.get_result_async()
    if not self._wait_future_done(result_future, timeout_sec=120.0):
      return None
    wrapped = result_future.result()
    if wrapped is None:
      return None
    result = wrapped.result
    if int(result.error_code.val) != int(MoveItErrorCodes.SUCCESS):
      if not preview:
        self._motion_log(
          'MoveIt',
          arm,
          f'STAGE 2 MoveGroup plan failed: val={result.error_code.val}',
          level='error',
        )
      return None
    traj = result.planned_trajectory
    if not traj.joint_trajectory.points:
      return None
    return traj

  def _build_final_approach_waypoints(
    self,
    from_xyz: tuple[float, float, float],
    contact_xyz: tuple[float, float, float],
    quat: Quaternion,
    steps: int,
  ) -> List[Pose]:
    """Cartesian fallback: straight line from start EE position to contact goal."""
    ax, ay, az = from_xyz
    cx, cy, cz = contact_xyz
    seg_dx = cx - ax
    seg_dy = cy - ay
    seg_dz = cz - az
    seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy + seg_dz * seg_dz)
    if seg_len <= 1e-6:
      wp = Pose()
      wp.position.x = cx
      wp.position.y = cy
      wp.position.z = cz
      wp.orientation = quat
      return [wp]
    waypoints: List[Pose] = []
    for i in range(1, steps + 1):
      t = float(i) / float(steps)
      wp = Pose()
      wp.position.x = ax + t * seg_dx
      wp.position.y = ay + t * seg_dy
      wp.position.z = az + t * seg_dz
      wp.orientation = quat
      waypoints.append(wp)
    return waypoints

  def _plan_final_cartesian_to_contact(
    self,
    arm: str,
    contact_xyz: tuple[float, float, float],
    quat: Quaternion,
    from_xyz: tuple[float, float, float],
    *,
    start_joint_state: Optional[JointState] = None,
    preview: bool = False,
  ) -> Optional[RobotTrajectory]:
    group_name = 'arm_r' if arm == 'r' else 'arm_l'
    ee_link = DEFAULT_EE_LINK_R if arm == 'r' else DEFAULT_EE_LINK_L
    steps = max(int(self.get_parameter('final_approach_steps').value), 1)
    waypoints = self._build_final_approach_waypoints(
      from_xyz, contact_xyz, quat, steps
    )
    if not waypoints:
      return None
    max_step = max(float(self.get_parameter('final_approach_max_step').value), 0.001)
    if not self._cartesian_path_cli.service_is_ready():
      return None

    req = GetCartesianPath.Request()
    req.header.frame_id = self._planning_frame
    req.header.stamp = self.get_clock().now().to_msg()
    req.start_state = RobotState()
    if start_joint_state is not None:
      req.start_state.joint_state = start_joint_state
    elif self._latest_joint_state is not None:
      req.start_state.joint_state = self._latest_joint_state
    req.group_name = group_name
    req.link_name = ee_link
    req.waypoints = waypoints
    req.max_step = max_step
    req.jump_threshold = 0.0
    req.prismatic_jump_threshold = 0.0
    req.revolute_jump_threshold = 0.0
    req.avoid_collisions = True

    fut = self._cartesian_path_cli.call_async(req)
    if not self._wait_future_done(fut, timeout_sec=60.0):
      return None
    resp = fut.result()
    if resp is None:
      return None
    if int(resp.error_code.val) != int(MoveItErrorCodes.SUCCESS):
      if not preview:
        self._motion_log(
          'MoveIt',
          arm,
          f'STAGE 2 Cartesian failed: val={resp.error_code.val} fraction={resp.fraction}',
          level='error',
        )
      return None
    if float(resp.fraction) <= 0.0 or not resp.solution.joint_trajectory.points:
      return None
    if float(resp.fraction) < 0.99 and not preview:
      self._motion_log(
        'MoveIt',
        arm,
        f'STAGE 2 Cartesian incomplete: fraction={resp.fraction}',
        level='warn',
      )
    return resp.solution

  def _robot_state_from_trajectory_end(
    self, traj: RobotTrajectory
  ) -> Optional[RobotState]:
    jt = traj.joint_trajectory
    if not jt.points:
      return None
    end = jt.points[-1]
    state = RobotState()
    if self._latest_joint_state is not None:
      js = JointState()
      js.header = self._latest_joint_state.header
      js.name = list(self._latest_joint_state.name)
      js.position = [float(p) for p in self._latest_joint_state.position]
      traj_idx = {n: i for i, n in enumerate(jt.joint_names)}
      for i, name in enumerate(js.name):
        if name in traj_idx:
          js.position[i] = float(end.positions[traj_idx[name]])
      state.joint_state = js
      return state
    js = JointState()
    js.name = list(jt.joint_names)
    js.position = [float(p) for p in end.positions]
    state.joint_state = js
    return state

  def _plan_final_tool_approach_trajectory(
    self,
    arm: str,
    *,
    start_joint_state: Optional[JointState] = None,
    preview_from_approach: bool = False,
    step_label: str = '',
  ) -> Optional[RobotTrajectory]:
    if not bool(self.get_parameter('enable_final_tool_approach').value):
      return None

    contact_xyz, quat_raw = self._contact_pose_for_arm(arm)
    approach_xyz = self._goal_xyz_for_arm(arm)
    if contact_xyz is None or quat_raw is None:
      return None
    quat = quat_raw if isinstance(quat_raw, Quaternion) else self._quat_from_wxyz_tuple(quat_raw)
    group_name = 'arm_r' if arm == 'r' else 'arm_l'
    ee_link = DEFAULT_EE_LINK_R if arm == 'r' else DEFAULT_EE_LINK_L
    goal_tol = max(float(self.get_parameter('final_approach_goal_tolerance_m').value), 0.001)
    max_dist = max(float(self.get_parameter('final_approach_max_distance_m').value), 0.01)
    vscale = max(
      min(float(self.get_parameter('final_approach_velocity_scale').value), 1.0),
      0.01,
    )

    if preview_from_approach:
      if approach_xyz is None:
        return None
      from_xyz = approach_xyz
      dist_to_contact = math.dist(from_xyz, contact_xyz)
    else:
      ee_xyz = self._lookup_ee_xyz(ee_link)
      if ee_xyz is None:
        return None
      from_xyz = ee_xyz
      dist_to_contact = math.dist(ee_xyz, contact_xyz)
      dist_to_approach = (
        math.dist(ee_xyz, approach_xyz) if approach_xyz is not None else float('nan')
      )
      self._motion_log(
        'MoveIt',
        arm,
        f'STAGE 2 check ({step_label or "final"}): EE=({ee_xyz[0]:.3f},{ee_xyz[1]:.3f},'
        f'{ee_xyz[2]:.3f}) → contact=({contact_xyz[0]:.3f},{contact_xyz[1]:.3f},'
        f'{contact_xyz[2]:.3f}), remain={dist_to_contact:.3f}m '
        f'(max {max_dist:.2f}m), dist EE→cuRobo goal={dist_to_approach:.3f}m',
      )
      if dist_to_contact > max_dist:
        self._motion_log(
          'MoveIt',
          arm,
          f'STAGE 2 skipped: EE too far from contact ({dist_to_contact:.3f}m > '
          f'{max_dist:.3f}m)',
          level='warn',
        )
        return None

    if dist_to_contact <= goal_tol:
      return None

    self._motion_log(
      'MoveIt',
      arm,
      f'STAGE 2 goal = clicked contact pose '
      f'({contact_xyz[0]:.3f},{contact_xyz[1]:.3f},{contact_xyz[2]:.3f}) '
      f'from current EE (remain {dist_to_contact:.3f}m)',
    )

    traj: Optional[RobotTrajectory] = None
    if bool(self.get_parameter('final_approach_use_move_group').value):
      req = MotionPlanRequest()
      req.group_name = group_name
      req.num_planning_attempts = max(
        int(self.get_parameter('final_approach_planning_attempts').value), 1
      )
      req.allowed_planning_time = max(
        float(self.get_parameter('final_approach_planning_time').value), 1.0
      )
      req.max_velocity_scaling_factor = vscale
      req.max_acceleration_scaling_factor = vscale
      req.goal_constraints = [
        self._contact_pose_goal_constraints(contact_xyz, quat, ee_link)
      ]
      if start_joint_state is not None:
        req.start_state = RobotState()
        req.start_state.joint_state = start_joint_state
      elif self._latest_joint_state is not None:
        req.start_state = RobotState()
        req.start_state.joint_state = self._latest_joint_state
      traj = self._call_move_group_plan(
        req, arm, preview=preview_from_approach
      )
      if traj is not None:
        self._motion_log('MoveIt', arm, 'STAGE 2 MoveGroup plan OK → contact pose')

    if traj is None:
      self._motion_log(
        'MoveIt',
        arm,
        'STAGE 2 Cartesian fallback → contact pose',
      )
      traj = self._plan_final_cartesian_to_contact(
        arm,
        contact_xyz,
        quat,
        from_xyz,
        start_joint_state=start_joint_state,
        preview=preview_from_approach,
      )

    return traj

  def _append_final_approach_display_previews(
    self,
    display: DisplayTrajectory,
    prepared: List[tuple[RobotTrajectory, str]],
    vscale: float,
  ) -> int:
    if not bool(self.get_parameter('enable_final_tool_approach').value):
      return 0
    min_sec = 1.0
    added = 0
    for traj, step_label in prepared:
      arm = self._resolve_final_approach_arm(step_label)
      if arm is None:
        continue
      start_state = self._robot_state_from_trajectory_end(traj)
      if start_state is None:
        continue
      preview = self._plan_final_tool_approach_trajectory(
        arm,
        start_joint_state=start_state.joint_state,
        preview_from_approach=True,
        step_label=step_label,
      )
      if preview is None:
        self._motion_log(
          'RViz',
          arm,
          f'STAGE 2 preview unavailable for {step_label} '
          '(GetCartesianPath failed — contact segment not shown)',
          level='warn',
        )
        continue
      self._scale_trajectory_speed(preview, vscale)
      self._enforce_min_trajectory_duration(preview, min_sec)
      display.trajectory.append(preview)
      added += 1
      jt = preview.joint_trajectory
      dur = self._traj_point_time_sec(jt.points[-1]) if jt.points else 0.0
      self._motion_log(
        'RViz',
        arm,
        f'STAGE 2 contact approach preview appended ({step_label}, '
        f'{len(jt.points)} pts, duration≈{dur:.2f}s)',
      )
    return added

  def _execute_final_tool_approach(self, arm: str, step_label: str = '') -> bool:
    if not bool(self.get_parameter('enable_final_tool_approach').value):
      self._motion_log('MoveIt', arm, 'STAGE 2 skipped (enable_final_tool_approach:=false)')
      return True
    contact_xyz, quat_raw = self._contact_pose_for_arm(arm)
    if contact_xyz is None or quat_raw is None:
      self._motion_log(
        'MoveIt',
        arm,
        'STAGE 2 skipped: contact pose missing — publish /target_pose_contact* '
        'from perception_click_planning',
        level='warn',
      )
      return True

    traj = self._plan_final_tool_approach_trajectory(
      arm, preview_from_approach=False, step_label=step_label
    )
    if traj is None:
      ee_link = DEFAULT_EE_LINK_R if arm == 'r' else DEFAULT_EE_LINK_L
      ee_xyz = self._lookup_ee_xyz(ee_link)
      goal_tol = max(float(self.get_parameter('final_approach_goal_tolerance_m').value), 0.001)
      max_dist = max(float(self.get_parameter('final_approach_max_distance_m').value), 0.01)
      if (
        ee_xyz is not None
        and goal_tol < math.dist(ee_xyz, contact_xyz) <= max_dist
      ):
        self._motion_log(
          'MoveIt',
          arm,
          'STAGE 2 failed: could not plan to contact pose',
          level='error',
        )
        return False
      return True

    n_pts = len(traj.joint_trajectory.points)
    dur = 0.0
    if n_pts:
      dur = self._traj_point_time_sec(traj.joint_trajectory.points[-1])
    self._motion_log(
      'MoveIt',
      arm,
      f'STAGE 2 path ready: {n_pts} pts, duration≈{dur:.2f}s',
    )
    vscale = max(
      min(float(self.get_parameter('final_approach_velocity_scale').value), 1.0),
      0.01,
    )
    self._scale_trajectory_speed(traj, vscale)
    self._enforce_min_trajectory_duration(traj, 1.0)
    self._publish_display_trajectory(traj, motion_arm=arm, stage=2)
    self._prepend_trajectory_start_state(traj)
    t0 = time.monotonic()
    ok = self._execute_trajectory(traj)
    if ok:
      self._motion_log(
        'MoveIt',
        arm,
        f'STAGE 2 done: contact reached in {time.monotonic() - t0:.2f}s',
      )
    else:
      self._motion_log('MoveIt', arm, 'STAGE 2 execute failed', level='error')
    return ok

  def _begin_confirm_session(self) -> int:
    self._confirm_session_id += 1
    self._awaiting_confirm = True
    self._execute_confirm_event.clear()
    return self._confirm_session_id

  def _end_confirm_session(self) -> None:
    self._awaiting_confirm = False

  def _confirm_and_execute_trajectory(
    self,
    traj: RobotTrajectory,
    label: str,
    *,
    require_confirm: Optional[bool] = None,
  ) -> bool:
    session_id = self._begin_confirm_session()
    try:
      vscale = self._resolve_velocity_scale()
      self._scale_trajectory_speed(traj, vscale)
      self._enforce_min_trajectory_duration(
        traj, float(self.get_parameter('min_trajectory_duration_sec').value)
      )
      arm = self._resolve_final_approach_arm(label)
      display_trajs: List[RobotTrajectory] = [traj]
      start_state = self._robot_state_from_trajectory_end(traj)
      if arm is not None and start_state is not None:
        preview = self._plan_final_tool_approach_trajectory(
          arm,
          start_joint_state=start_state.joint_state,
          preview_from_approach=True,
          step_label=label,
        )
        if preview is not None:
          self._scale_trajectory_speed(preview, vscale)
          self._enforce_min_trajectory_duration(preview, 1.0)
          display_trajs.append(preview)
      self._publish_display_trajectories(display_trajs, motion_arm=arm, label=label)
      need_confirm = (
        bool(self.get_parameter('confirm_before_execute').value)
        if require_confirm is None
        else require_confirm
      )
      if need_confirm:
        confirm_topic = str(self.get_parameter('execute_confirm_topic').value).strip()
        arm = self._resolve_final_approach_arm(label)
        self._motion_log(
          'RViz',
          arm,
          f'STAGE 1 path shown — confirm before cuRobo coarse execute (label={label})',
        )
        self._plan_progress(
          'awaiting confirm',
          f'label={label}, velocity={vscale * 100:.0f}% — check RViz then confirm',
        )
        self.get_logger().info(
          f'>>> [{label}] confirm execution:\n'
          f'    ros2 topic pub --once {confirm_topic} std_msgs/msg/Empty "{{}}"'
        )
      else:
        arm = self._resolve_final_approach_arm(label)
        self._motion_log(
          'cuRobo',
          arm,
          f'STAGE 1 auto-execute coarse path (label={label}, velocity={vscale * 100:.0f}%)',
        )
        self._plan_progress(
          'auto execute',
          f'label={label}, velocity={vscale * 100:.0f}% (no confirm)',
        )

      if not self._wait_for_execute_confirmation(session_id, require_confirm=need_confirm):
        if self._confirm_session_id != session_id:
          self.get_logger().info('Execution skipped (superseded by newer plan).')
        else:
          self.get_logger().info('Execution skipped.')
        return False

      arm = self._resolve_final_approach_arm(label)
      self._motion_log(
        'cuRobo',
        arm,
        f'STAGE 1 executing coarse trajectory (label={label})',
      )
      t0 = time.monotonic()
      if self._execute_trajectory(traj):
        self._last_plan_mono = time.monotonic()
        self._motion_log(
          'cuRobo',
          arm,
          f'STAGE 1 done in {time.monotonic() - t0:.2f}s (label={label})',
        )
        if arm is not None:
          self._wait_joint_states_after_move(f'after {label}')
          self._wait_before_final_approach(arm, label)
          if not self._execute_final_tool_approach(arm, label):
            return False
        elif self._planning_mode == 'dual':
          for final_arm, final_label in (('l', 'left arm'), ('r', 'right arm')):
            self._wait_joint_states_after_move(f'after {final_label}')
            self._wait_before_final_approach(final_arm, final_label)
            if not self._execute_final_tool_approach(final_arm, final_label):
              return False
        return True
      self._plan_progress(
        'execute failed',
        f'label={label}, elapsed={time.monotonic() - t0:.2f}s',
      )
      self.get_logger().error(
        'ExecuteTrajectory failed (check move_group, arm_l/arm_r/lift controllers, '
        'and move_group_namespace; for ffw moveit.launch.py use '
        'move_group_namespace:=root)'
      )
      return False
    finally:
      if self._confirm_session_id == session_id:
        self._end_confirm_session()

  def _confirm_and_execute_trajectory_sequence(
    self,
    steps: List[tuple[RobotTrajectory, str]],
    exec_label: str,
    *,
    require_confirm: Optional[bool] = None,
  ) -> bool:
    if not steps:
      return False
    session_id = self._begin_confirm_session()
    try:
      vscale = self._resolve_velocity_scale()
      min_sec = float(self.get_parameter('min_trajectory_duration_sec').value)
      prepared: List[tuple[RobotTrajectory, str]] = []
      for traj, step_label in steps:
        cloned = self._clone_robot_trajectory(traj)
        self._scale_trajectory_speed(cloned, vscale)
        self._enforce_min_trajectory_duration(cloned, min_sec)
        prepared.append((cloned, step_label))

      display = DisplayTrajectory()
      display.model_id = str(self.get_parameter('robot_model_id').value)
      if self._latest_joint_state is not None:
        start = RobotState()
        start.joint_state = self._latest_joint_state
        display.trajectory_start = start
      for traj, _ in prepared:
        display.trajectory.append(traj)
      preview_count = self._append_final_approach_display_previews(
        display, prepared, vscale
      )
      count = max(1, int(self.get_parameter('display_path_publish_count').value))
      for _ in range(count):
        self._display_path_pub.publish(display)
        time.sleep(0.05)
      step_summary = ' → '.join(label for _, label in prepared)
      self.get_logger().info(
        f'[RViz] STAGE 1 path sequence ({len(prepared)} segments): {step_summary}'
      )
      if preview_count > 0:
        self.get_logger().info(
          f'[RViz] STAGE 2 contact approach preview ({preview_count} segment(s)) '
          f'appended — pre-grasp → contact on {self._display_path_pub.topic_name}'
        )

      need_confirm = (
        bool(self.get_parameter('confirm_before_execute').value)
        if require_confirm is None
        else require_confirm
      )
      if need_confirm:
        confirm_topic = str(self.get_parameter('execute_confirm_topic').value).strip()
        self._plan_progress(
          'awaiting confirm',
          f'label={exec_label}, steps={len(prepared)} ({step_summary}) — check RViz',
        )
        self.get_logger().info(
          f'>>> [{exec_label}] confirm ALL steps:\n'
          f'    ros2 topic pub --once {confirm_topic} std_msgs/msg/Empty "{{}}"'
        )
      else:
        self._plan_progress(
          'auto execute sequence',
          f'label={exec_label}, steps={len(prepared)}',
        )

      if not self._wait_for_execute_confirmation(session_id, require_confirm=need_confirm):
        if self._confirm_session_id != session_id:
          self.get_logger().info('Execution skipped (superseded by newer plan).')
        else:
          self.get_logger().info('Execution skipped.')
        return False

      self.get_logger().info(
        f'[EXEC] sequence start ({len(prepared)} cuRobo steps): {step_summary}'
      )

      stage2_queue: List[tuple[str, str]] = []
      for idx, (traj, step_label) in enumerate(prepared):
        if idx > 0 and not self._wait_joint_states_after_move(
          f'before STAGE 1 step {idx + 1}/{len(prepared)} ({step_label})'
        ):
          self._plan_progress(
            'failed',
            f'no /joint_states before {step_label}',
          )
          return False
        arm = self._resolve_final_approach_arm(step_label)
        self._motion_log(
          'cuRobo',
          arm,
          f'STAGE 1 executing step {idx + 1}/{len(prepared)}: {step_label}',
        )
        t0 = time.monotonic()
        if not self._execute_trajectory(traj):
          self._plan_progress(
            'execute failed',
            f'STAGE 1 step {idx + 1}/{len(prepared)}: {step_label}',
          )
          return False
        self._motion_log(
          'cuRobo',
          arm,
          f'STAGE 1 step {idx + 1}/{len(prepared)} done: {step_label}, '
          f'elapsed={time.monotonic() - t0:.2f}s',
        )
        if arm is not None:
          stage2_queue.append((arm, step_label))
        if idx + 1 < len(prepared):
          self._wait_joint_states_after_move(
            f'after STAGE 1 step {idx + 1}/{len(prepared)} ({step_label})'
          )

      if stage2_queue:
        stage2_summary = ' → '.join(label for _, label in stage2_queue)
        self.get_logger().info(
          f'[EXEC] STAGE 2 phase ({len(stage2_queue)} arms, matches RViz order): '
          f'{stage2_summary}'
        )
        if not self._wait_joint_states_after_move('before STAGE 2 phase'):
          self._plan_progress('failed', 'no /joint_states before STAGE 2 phase')
          return False
        self._wait_before_final_approach(stage2_queue[0][0], 'STAGE 2 phase')
        for sidx, (arm, step_label) in enumerate(stage2_queue):
          if sidx > 0:
            if not self._wait_joint_states_after_move(
              f'before STAGE 2 step {sidx + 1}/{len(stage2_queue)} ({step_label})'
            ):
              self._plan_progress(
                'failed',
                f'no /joint_states before MoveIt {step_label}',
              )
              return False
          self._motion_log(
            'MoveIt',
            arm,
            f'STAGE 2 executing {sidx + 1}/{len(stage2_queue)}: {step_label}',
          )
          if not self._execute_final_tool_approach(arm, step_label):
            self._plan_progress(
              'execute failed',
              f'MoveIt STAGE 2 after {step_label}',
            )
            return False
          if sidx + 1 < len(stage2_queue):
            self._wait_joint_states_after_move(
              f'after STAGE 2 step {sidx + 1}/{len(stage2_queue)} ({step_label})'
            )

      self._last_plan_mono = time.monotonic()
      self.get_logger().info(
        f'[EXEC] sequence complete: STAGE 1 ({step_summary})'
        + (
          f' → STAGE 2 ({stage2_summary})'
          if stage2_queue
          else ''
        )
      )
      self._plan_progress('execute done', f'label={exec_label}, all {len(prepared)} steps')
      return True
    finally:
      if self._confirm_session_id == session_id:
        self._end_confirm_session()

  def _current_trajectory_positions(self, joint_names: List[str]) -> Optional[List[float]]:
    if self._latest_joint_state is None:
      return None
    name_to_pos = dict(
      zip(self._latest_joint_state.name, self._latest_joint_state.position)
    )
    try:
      return [float(name_to_pos[jn]) for jn in joint_names]
    except KeyError:
      return None

  def _prepend_trajectory_start_state(self, traj: RobotTrajectory) -> None:
    """First point at t=0 from /joint_states — avoids CONTROL_FAILED (-4)."""
    jnames = list(traj.joint_trajectory.joint_names)
    start_pos = self._current_trajectory_positions(jnames)
    if start_pos is None:
      return
    pt0 = JointTrajectoryPoint()
    pt0.positions = start_pos
    pt0.time_from_start.sec = 0
    pt0.time_from_start.nanosec = 0
    if not traj.joint_trajectory.points:
      traj.joint_trajectory.points.append(pt0)
      return
    first = traj.joint_trajectory.points[0]
    if len(first.positions) == len(start_pos):
      max_delta = max(abs(float(a) - float(b)) for a, b in zip(first.positions, start_pos))
      if max_delta < 0.02:
        first.time_from_start.sec = 0
        first.time_from_start.nanosec = 0
        first.positions = start_pos
        return
    traj.joint_trajectory.points.insert(0, pt0)

  def _plan_dict_to_trajectory(self, plan: dict) -> Optional[RobotTrajectory]:
    points = plan.get('points') or []
    if not points:
      self.get_logger().error('Empty trajectory from cuRobo')
      return None

    traj = RobotTrajectory()
    joint_names = list(plan.get('joint_names', self._trajectory_joints))
    traj.joint_trajectory.joint_names = joint_names

    for p in points:
      pt = JointTrajectoryPoint()
      pt.positions = [float(x) for x in p['positions']]
      if len(pt.positions) != len(joint_names):
        self.get_logger().error(
          f'Trajectory point has {len(pt.positions)} joints, expected {len(joint_names)}'
        )
        return None
      if p.get('velocities') is not None:
        pt.velocities = [float(x) for x in p['velocities']]
      if p.get('accelerations') is not None:
        pt.accelerations = [float(x) for x in p['accelerations']]
      t = float(p['time_from_start'])
      pt.time_from_start.sec = int(t)
      pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
      traj.joint_trajectory.points.append(pt)

    self._prepend_trajectory_start_state(traj)
    traj.joint_trajectory.header.stamp = self.get_clock().now().to_msg()
    return traj

  def _stretch_trajectory_time(self, traj: RobotTrajectory, time_factor: float) -> None:
    """Multiply all point times and scale vel/acc (time_factor > 1 = slower)."""
    time_factor = max(float(time_factor), 1.0)
    if time_factor <= 1.0001:
      return
    vel_scale = 1.0 / time_factor
    acc_scale = vel_scale * vel_scale
    jt = traj.joint_trajectory
    for pt in jt.points:
      t = self._traj_point_time_sec(pt) * time_factor
      pt.time_from_start.sec = int(t)
      pt.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
      if pt.velocities:
        pt.velocities = [float(v) * vel_scale for v in pt.velocities]
      if pt.accelerations:
        pt.accelerations = [float(a) * acc_scale for a in pt.accelerations]

  def _scale_trajectory_speed(self, traj: RobotTrajectory, speed_scale: float) -> None:
    """Stretch trajectory time and scale vel/acc; speed_scale in (0, 1]."""
    speed_scale = max(min(float(speed_scale), 1.0), 0.01)
    if speed_scale >= 0.999:
      return
    self._stretch_trajectory_time(traj, 1.0 / speed_scale)
    jt = traj.joint_trajectory
    if jt.points:
      last_t = self._traj_point_time_sec(jt.points[-1])
      self.get_logger().info(
        f'Trajectory slowed: velocity_scale={speed_scale:.2f}, '
        f'duration≈{last_t:.2f}s ({len(jt.points)} points)'
      )

  def _enforce_min_trajectory_duration(self, traj: RobotTrajectory, min_sec: float) -> None:
    min_sec = max(float(min_sec), 0.0)
    if min_sec <= 0.0:
      return
    jt = traj.joint_trajectory
    if not jt.points:
      return
    last_t = self._traj_point_time_sec(jt.points[-1])
    if last_t >= min_sec:
      return
    factor = min_sec / max(last_t, 1e-6)
    self._stretch_trajectory_time(traj, factor)
    self.get_logger().info(
      f'Trajectory stretched to min duration: {min_sec:.1f}s '
      f'(was {last_t:.2f}s, factor={factor:.2f})'
    )

  def _publish_display_trajectories(
    self,
    trajectories: List[RobotTrajectory],
    *,
    motion_arm: Optional[str] = None,
    label: str = '',
  ) -> None:
    if not trajectories:
      return
    display = DisplayTrajectory()
    display.model_id = str(self.get_parameter('robot_model_id').value)
    if self._latest_joint_state is not None:
      start = RobotState()
      start.joint_state = self._latest_joint_state
      display.trajectory_start = start
    for traj in trajectories:
      display.trajectory.append(traj)
    count = max(1, int(self.get_parameter('display_path_publish_count').value))
    for _ in range(count):
      self._display_path_pub.publish(display)
      time.sleep(0.05)
    stage_note = ''
    if len(trajectories) > 1:
      stage_note = f' (STAGE 1 + STAGE 2 contact, {len(trajectories)} segments)'
    elif label:
      stage_note = f' ({label})'
    total_pts = sum(len(t.joint_trajectory.points) for t in trajectories)
    self._motion_log(
      'RViz',
      motion_arm,
      f'planned path → {self._display_path_pub.topic_name}{stage_note}, '
      f'{total_pts} pts total. MotionPlanning → Planned Path',
    )

  def _publish_display_trajectory(
    self, traj: RobotTrajectory, *, motion_arm: Optional[str] = None, stage: int = 1
  ) -> None:
    stage_label = 'STAGE 1 coarse' if stage == 1 else 'STAGE 2 contact'
    self._publish_display_trajectories(
      [traj], motion_arm=motion_arm, label=stage_label
    )

  def _on_execute_confirm(self, _msg: Empty) -> None:
    if not self._awaiting_confirm:
      self.get_logger().warn(
        'Execute confirm ignored: no planned path is waiting for confirmation.',
        throttle_duration_sec=5.0,
      )
      return
    self.get_logger().info(
      '[EXEC] confirm received — STAGE 1 all cuRobo steps, then STAGE 2 MoveIt per arm '
      '(same order as RViz Planned Path)'
    )
    self._execute_confirm_event.set()

  def _wait_for_execute_confirmation(
    self, session_id: int, *, require_confirm: Optional[bool] = None
  ) -> bool:
    need_confirm = (
      bool(self.get_parameter('confirm_before_execute').value)
      if require_confirm is None
      else require_confirm
    )
    if not need_confirm:
      return True
    confirm_topic = str(self.get_parameter('execute_confirm_topic').value).strip()

    def _stdin_confirm() -> None:
      if not sys.stdin.isatty():
        self.get_logger().warn(
          'stdin is not a TTY — Enter in this terminal will NOT confirm execution. '
          f'Use: ros2 topic pub --once {confirm_topic} std_msgs/msg/Empty "{{}}"'
        )
        return
      try:
        input()
        if self._confirm_session_id == session_id:
          self.get_logger().info('Execute confirmed (Enter).')
          self._execute_confirm_event.set()
      except EOFError:
        pass

    threading.Thread(target=_stdin_confirm, daemon=True).start()

    while rclpy.ok():
      if self._confirm_session_id != session_id:
        return False
      if self._execute_confirm_event.wait(timeout=0.2):
        return self._confirm_session_id == session_id
    return False

  def _execute_trajectory(self, traj: RobotTrajectory) -> bool:
    # Re-sync start right before sending to avoid controller start mismatch.
    self._prepend_trajectory_start_state(traj)
    traj.joint_trajectory.header.stamp = self.get_clock().now().to_msg()
    joint_names = list(traj.joint_trajectory.joint_names)

    goal = ExecuteTrajectory.Goal()
    goal.trajectory = traj

    send_future = self._execute.send_goal_async(goal)
    if not self._wait_future_done(send_future):
      return False
    gh = send_future.result()
    if not gh or not gh.accepted:
      self.get_logger().error('ExecuteTrajectory goal rejected')
      return False

    result_future = gh.get_result_async()
    if not self._wait_future_done(result_future):
      return False
    result = result_future.result().result
    if result.error_code.val != MoveItErrorCodes.SUCCESS:
      v = int(result.error_code.val)
      hint = _MOVEIT_ERROR_HINTS.get(v, 'see moveit_msgs/MoveItErrorCodes')
      self.get_logger().error(f'ExecuteTrajectory failed: {v} ({hint})')
      if v == -4 and traj.joint_trajectory.points:
        cur = self._current_trajectory_positions(joint_names)
        goal_pos = list(traj.joint_trajectory.points[-1].positions)
        if cur is not None:
          deltas = [abs(float(c) - float(g)) for c, g in zip(cur, goal_pos)]
          self.get_logger().error(
            f'  joints={joint_names}; max|current-goal|={max(deltas):.3f} rad'
          )
      return False
    return True


def main() -> None:
  rclpy.init()
  node = CuRoboPathPlanningNode()
  if not node.ready_for_planning():
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(1)

  executor = MultiThreadedExecutor(num_threads=4)
  executor.add_node(node)
  node.arm_planning()
  try:
    executor.spin()
  except KeyboardInterrupt:
    pass
  finally:
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
