#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# obstacle_node (/obstacle_markers) -> MoveIt planning scene + cuRobo HTTP /plan -> ExecuteTrajectory
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
#   and /curobo_target_pose (PoseStamped). Fixed Frame = planning_frame (default base_link).
# Manual goal: publish geometry_msgs/PoseStamped on goal_pose_topic (default /target_pose).

from __future__ import annotations

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

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import ColorRGBA, Header
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import ApplyPlanningScene
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

try:
  import requests
except ImportError as e:
  print('Install requests: pip install requests', file=sys.stderr)
  raise SystemExit(1) from e

from curobo_obstacle_utils import (
  DEFAULT_PLANNING_FRAME,
  marker_array_signature,
  markers_to_curobo_cuboids,
  markers_to_planning_scene,
  quat_toward_robot_y_up,
)

DEFAULT_EE_LINK = 'end_effector_l_link'
ARM_L_JOINTS = [
  'arm_l_joint1',
  'arm_l_joint2',
  'arm_l_joint3',
  'arm_l_joint4',
  'arm_l_joint5',
  'arm_l_joint6',
  'arm_l_joint7',
]

# ExecuteTrajectory: lift + arm_l (cuRobo IK often changes lift for reach height).
TRAJECTORY_JOINTS = ['lift_joint', *ARM_L_JOINTS]

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

TARGET_MARKER_NS = 'curobo_target'
TARGET_MARKER_ID_ARROW = 0
TARGET_MARKER_ID_SPHERE = 1


class CuRoboPathPlanningNode(Node):
  def __init__(self) -> None:
    super().__init__('curobo_pathplanning_node')
    self._cb_group = ReentrantCallbackGroup()

    self.declare_parameter('obstacle_topic', '/obstacle_markers')
    self.declare_parameter('apply_planning_scene_service', '')
    self.declare_parameter('move_group_namespace', '/move_group')
    _default_curobo_url = os.environ.get('CUROBO_URL', 'http://127.0.0.1:8000/plan')
    self.declare_parameter('curobo_url', _default_curobo_url)
    self.declare_parameter('curobo_health_url', '')
    self.declare_parameter('joint_state_topic', '/joint_states')
    self.declare_parameter('goal_pose_topic', '/target_pose')
    self.declare_parameter('planning_frame', DEFAULT_PLANNING_FRAME)
    self.declare_parameter('auto_plan', True)
    self.declare_parameter('min_plan_interval_sec', 0.5)
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
    self.declare_parameter('home', False)
    self.declare_parameter('sync_moveit_scene', True)
    self.declare_parameter('send_obstacles_to_curobo', True)
    self.declare_parameter('curobo_obstacle_dim_scale', 0.92)
    self.declare_parameter('use_passive_defaults_for_start', True)
    self.declare_parameter('skip_start_feasibility_check', False)
    self.declare_parameter('publish_target_visualization', True)
    self.declare_parameter('target_marker_topic', '/curobo_target_markers')
    self.declare_parameter('target_pose_viz_topic', '/curobo_target_pose')
    self.declare_parameter('target_marker_publish_period_sec', 1.0)
    self.declare_parameter('target_arrow_length', 0.12)
    self.declare_parameter('target_sphere_diameter', 0.06)

    self._curobo_url = str(self.get_parameter('curobo_url').value).strip()
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
    self._arm_joints = list(ARM_L_JOINTS)
    self._trajectory_joints = list(TRAJECTORY_JOINTS)
    self._curobo_state_joints = list(CUROBO_STATE_JOINTS)
    self._latest_joint_state: Optional[JointState] = None
    self._last_obstacle_cuboids: List[dict] = []
    self._last_plan_mono = -1e10
    self._last_marker_sig: tuple | None = None
    self._obstacle_lock = threading.Lock()

    srv = self._resolve_apply_scene_service()
    self._apply_scene_cli = self.create_client(
      ApplyPlanningScene, srv, callback_group=self._cb_group
    )
    self.get_logger().info(f'ApplyPlanningScene service: {srv}')

    raw = str(self.get_parameter('move_group_namespace').value).strip()
    low = raw.lower()
    if low in ('', 'root', '/'):
      exec_name = '/execute_trajectory'
    else:
      ns = raw if raw.startswith('/') else '/' + raw
      ns = ns.rstrip('/') or '/move_group'
      exec_name = f'{ns}/execute_trajectory'

    self._execute = ActionClient(
      self, ExecuteTrajectory, exec_name, callback_group=self._cb_group
    )
    self.get_logger().info(f'ExecuteTrajectory action: {exec_name}')

    self._qos = QoSProfile(
      depth=10,
      durability=DurabilityPolicy.VOLATILE,
      reliability=ReliabilityPolicy.RELIABLE,
    )

    js_topic = str(self.get_parameter('joint_state_topic').value)
    self.create_subscription(
      JointState,
      js_topic,
      self._on_joint_states,
      10,
      callback_group=self._cb_group,
    )
    self.get_logger().info(f'Subscribed to {js_topic}')

    goal_topic = str(self.get_parameter('goal_pose_topic').value).strip()
    if goal_topic:
      self.create_subscription(
        PoseStamped,
        goal_topic,
        self._on_goal_pose,
        10,
        callback_group=self._cb_group,
      )
      self.get_logger().info(f'Manual goals on {goal_topic}')

    self._init_goal_cache()
    self._setup_target_visualization()

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

  def _goal_quat_for_position(self, xyz: tuple[float, float, float]):
    bx = float(self.get_parameter('robot_base_x').value)
    by = float(self.get_parameter('robot_base_y').value)
    return quat_toward_robot_y_up(xyz, robot_base_xy=(bx, by))

  def _init_goal_cache(self) -> None:
    home = self.get_parameter('home').value
    home_on = home if isinstance(home, bool) else str(home).strip().lower() in ('true', '1', 'yes')
    if home_on:
      self._cached_goal_xyz = (0.45, 0.25, 1.35)
      self._cached_goal_quat = self._goal_quat_for_position(self._cached_goal_xyz)
      self.get_logger().info('home=true: using fixed reach pose toward home configuration.')
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

    x, y, z = _u('x'), _u('y'), _u('z')
    self._cached_goal_xyz = (x, y, z)
    self._cached_goal_quat = self._goal_quat_for_position(self._cached_goal_xyz)
    self.get_logger().info(
      f'Cached goal {self._planning_frame} xyz=({x:.3f},{y:.3f},{z:.3f}) '
      f'(EE +X→robot, +Y↑ground)'
    )

  def _setup_target_visualization(self) -> None:
    if not bool(self.get_parameter('publish_target_visualization').value):
      return
    marker_topic = str(self.get_parameter('target_marker_topic').value).strip()
    pose_topic = str(self.get_parameter('target_pose_viz_topic').value).strip()
    self._target_marker_pub = self.create_publisher(MarkerArray, marker_topic, self._qos)
    self._target_pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
    period = float(self.get_parameter('target_marker_publish_period_sec').value)
    self._publish_target_visualization()
    if period > 0.0:
      self.create_timer(period, self._publish_target_visualization)
    self.get_logger().info(
      f'Target RViz: MarkerArray "{marker_topic}", PoseStamped "{pose_topic}" '
      f'(frame={self._planning_frame})'
    )

  def _cached_goal_pose_msg(self) -> Pose:
    x, y, z = self._cached_goal_xyz
    q = self._cached_goal_quat
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.x = float(q.x)
    pose.orientation.y = float(q.y)
    pose.orientation.z = float(q.z)
    pose.orientation.w = float(q.w)
    return pose

  def _make_target_markers(self, stamp) -> MarkerArray:
    pose = self._cached_goal_pose_msg()
    frame = self._planning_frame
    arrow_len = float(self.get_parameter('target_arrow_length').value)
    sphere_d = float(self.get_parameter('target_sphere_diameter').value)

    arrow = Marker()
    arrow.header = Header(stamp=stamp, frame_id=frame)
    arrow.ns = TARGET_MARKER_NS
    arrow.id = TARGET_MARKER_ID_ARROW
    arrow.type = Marker.ARROW
    arrow.action = Marker.ADD
    arrow.pose = pose
    arrow.scale.x = max(arrow_len, 0.02)
    arrow.scale.y = 0.025
    arrow.scale.z = 0.025
    arrow.color = ColorRGBA(r=0.1, g=0.85, b=0.25, a=0.95)

    sphere = Marker()
    sphere.header = Header(stamp=stamp, frame_id=frame)
    sphere.ns = TARGET_MARKER_NS
    sphere.id = TARGET_MARKER_ID_SPHERE
    sphere.type = Marker.SPHERE
    sphere.action = Marker.ADD
    sphere.pose = pose
    sphere.scale.x = sphere_d
    sphere.scale.y = sphere_d
    sphere.scale.z = sphere_d
    sphere.color = ColorRGBA(r=0.1, g=0.55, b=1.0, a=0.75)

    arr = MarkerArray()
    arr.markers.append(arrow)
    arr.markers.append(sphere)
    return arr

  def _publish_target_visualization(self) -> None:
    if not bool(self.get_parameter('publish_target_visualization').value):
      return
    if not hasattr(self, '_target_marker_pub'):
      return
    stamp = self.get_clock().now().to_msg()
    self._target_marker_pub.publish(self._make_target_markers(stamp))
    pose_msg = PoseStamped()
    pose_msg.header.stamp = stamp
    pose_msg.header.frame_id = self._planning_frame
    pose_msg.pose = self._cached_goal_pose_msg()
    self._target_pose_pub.publish(pose_msg)

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
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok():
      if fut.done():
        return True
      if time.monotonic() >= deadline:
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
    return True

  def create_obstacle_subscription(self) -> None:
    topic = str(self.get_parameter('obstacle_topic').value)
    self.create_subscription(
      MarkerArray,
      topic,
      self._on_obstacles,
      self._qos,
      callback_group=self._cb_group,
    )
    self.get_logger().info(f'Subscribed to {topic} (MarkerArray)')

  def _on_joint_states(self, msg: JointState) -> None:
    self._latest_joint_state = msg

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
      if jn in passive_defaults:
        names.append(jn)
        positions.append(passive_defaults[jn])
      elif jn in name_to_pos:
        names.append(jn)
        positions.append(float(name_to_pos[jn]))
      else:
        missing.append(jn)
    if not names:
      raise RuntimeError('No cuRobo state joints found in /joint_states')
    for jn in self._arm_joints:
      if jn not in name_to_pos:
        raise RuntimeError(f'Joint {jn} missing from /joint_states')
    if 'lift_joint' not in name_to_pos:
      raise RuntimeError('Joint lift_joint missing from /joint_states')
    if missing:
      self.get_logger().warn(
        f'cuRobo start state: {len(missing)} joints missing from /joint_states '
        f'(cuRobo server YAML defaults): {missing}',
        throttle_duration_sec=30.0,
      )
    return names, positions

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

  def _on_goal_pose(self, msg: PoseStamped) -> None:
    xyz, _ = self._goal_from_pose_stamped(msg)
    self._cached_goal_xyz = xyz
    self._cached_goal_quat = self._goal_quat_for_position(xyz)
    self.get_logger().info(f'Manual goal updated: xyz={xyz} (orientation: +X→robot, +Y↑)')
    self._publish_target_visualization()
    self._plan_and_execute(reason='goal_pose_topic')

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
    with self._obstacle_lock:
      sig = marker_array_signature(msg)
      if self._last_marker_sig is not None and sig == self._last_marker_sig:
        return

      if bool(self.get_parameter('sync_moveit_scene').value):
        stamp = self.get_clock().now().to_msg()
        scene = markers_to_planning_scene(msg, stamp, self.get_logger())
        if scene is not None and not self._call_apply_planning_scene(scene):
          return

      self._last_obstacle_cuboids = markers_to_curobo_cuboids(msg)
      self._last_marker_sig = sig
      self.get_logger().info(
        f'Obstacles for cuRobo: {len(self._last_obstacle_cuboids)} cuboid(s)'
      )

      settle = float(self.get_parameter('scene_settle_sec').value)
      if settle > 0.0:
        time.sleep(settle)

      if not bool(self.get_parameter('auto_plan').value):
        self.get_logger().info('auto_plan=false; scene updated only.')
        return

      now = time.monotonic()
      min_dt = float(self.get_parameter('min_plan_interval_sec').value)
      if now - self._last_plan_mono < min_dt:
        return

      self._plan_and_execute(reason='obstacle_update')

  def _plan_and_execute(self, reason: str) -> None:
    try:
      state_names, state_positions = self._current_curobo_joint_state()
    except RuntimeError as e:
      self.get_logger().error(str(e))
      return

    x, y, z = self._cached_goal_xyz
    q = self._cached_goal_quat
    send_obs = bool(self.get_parameter('send_obstacles_to_curobo').value)
    obs_scale = float(self.get_parameter('curobo_obstacle_dim_scale').value)
    payload = {
      'joint_names': state_names,
      'current_joint_positions': state_positions,
      'trajectory_joint_names': self._trajectory_joints,
      'target_pose': {
        'position': [x, y, z],
        'quaternion': [float(q.w), float(q.x), float(q.y), float(q.z)],
      },
      'obstacles': self._last_obstacle_cuboids if send_obs else [],
      'obstacle_dim_scale': obs_scale,
      'max_attempts': int(self.get_parameter('max_attempts').value),
      'skip_start_feasibility_check': bool(
        self.get_parameter('skip_start_feasibility_check').value
      ),
    }

    lift = next(
      (p for n, p in zip(state_names, state_positions) if n == 'lift_joint'),
      None,
    )
    self.get_logger().info(
      f'cuRobo plan ({reason}): target=({x:.3f},{y:.3f},{z:.3f}), '
      f'state_joints={len(state_names)}, obstacles={len(payload["obstacles"])}, '
      f'lift_joint={lift}'
    )
    try:
      response = requests.post(self._curobo_url, json=payload, timeout=60.0)
      response.raise_for_status()
      plan = response.json()
    except requests.RequestException as e:
      self.get_logger().error(f'cuRobo HTTP failed: {e}')
      return

    if not plan.get('success', False):
      msg = plan.get('message', '')
      if send_obs and payload['obstacles']:
        obs_detail = ', '.join(
          f"{o.get('name', '?')}@[{o['pose'][0]:.2f},{o['pose'][1]:.2f},{o['pose'][2]:.2f}]"
          for o in payload['obstacles'][:3]
        )
        self.get_logger().error(
          f'cuRobo planning failed: {msg} (obstacles: {obs_detail}, dim_scale={obs_scale})'
        )
      else:
        self.get_logger().error(f'cuRobo planning failed: {msg}')
      return

    if self._send_execute_trajectory(plan):
      self._last_plan_mono = time.monotonic()
      self.get_logger().info('cuRobo trajectory executed.')

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

  def _send_execute_trajectory(self, plan: dict) -> bool:
    points = plan.get('points') or []
    if not points:
      self.get_logger().error('Empty trajectory from cuRobo')
      return False

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
        return False
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
        goal = list(traj.joint_trajectory.points[-1].positions)
        if cur is not None:
          deltas = [abs(float(c) - float(g)) for c, g in zip(cur, goal)]
          self.get_logger().error(
            f'  joints={joint_names}; max|current-goal|={max(deltas):.3f} rad/m'
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
  node.create_obstacle_subscription()
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
