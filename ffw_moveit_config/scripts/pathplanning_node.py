#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Subscribes to visualization_msgs/MarkerArray (obstacle_node.py), applies spheres/boxes
# to the MoveIt planning scene via ApplyPlanningScene, then plans and executes.
#
# Service/action futures from subscription callbacks must not call spin_until_future_complete
# on the same executor (RuntimeError: Executor is already spinning). We poll futures with
# short sleeps while MultiThreadedExecutor.spin() runs on other threads. Subscription is
# created only after move_group is ready so spin_once during wait does not run _on_obstacles.
#
# Params:
#   obstacle_topic (str, default /obstacle_markers)
#   move_group_namespace, apply_planning_scene_service (same as moveit_planning_scene_obstacles.py)
#   goal_mode: pose | joint
#   auto_plan (bool, default true) — plan+execute on each obstacle message
#   min_plan_interval_sec (float, default 0.5) — throttle repeated plans
#   scene_settle_sec (float, default 0.05) — sleep after ApplyPlanningScene before MoveGroup
#   goal_seed (int, default -1) — >=0 reproducible goal; -1 random goal once per node start
#   goal_pose_{x,y,z}_{min,max}, goal_pose_yaw_{min,max} — pose sampling in base_link (m, rad)
#   goal_joint_jitter_rad — uniform +/- jitter on nominal joint demo (joint mode)
#   home (bool, default false) — if true, plan to arm_l seven joints all 0 (ignores goal_mode)
#
# Requires MultiThreadedExecutor (>=2 threads): futures complete while a callback sleeps.
#
# Example:
#   ros2 run ffw_moveit_config pathplanning_node.py --ros-args -p use_sim_time:=true -p move_group_namespace:=root
#   ros2 run ffw_moveit_config pathplanning_node.py --ros-args -p goal_seed:=7 -p goal_pose_z_max:=1.5
#   ros2 run ffw_moveit_config pathplanning_node.py --ros-args -p home:=true -p move_group_namespace:=root

import math
import random
import secrets
import shutil
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose, Quaternion, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
  BoundingVolume,
  CollisionObject,
  Constraints,
  JointConstraint,
  MotionPlanRequest,
  MoveItErrorCodes,
  OrientationConstraint,
  PlanningOptions,
  PlanningScene,
  PositionConstraint,
)
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray

DEFAULT_GROUP = 'arm_l'
DEFAULT_EE_LINK = 'end_effector_l_link'
DEFAULT_PLANNING_FRAME = 'base_link'

ARM_L_JOINTS = [
  'arm_l_joint1',
  'arm_l_joint2',
  'arm_l_joint3',
  'arm_l_joint4',
  'arm_l_joint5',
  'arm_l_joint6',
  'arm_l_joint7',
]

_MOVEIT_ERROR_HINTS = {
  -1: 'PLANNING_FAILED',
  -2: 'INVALID_MOTION_PLAN',
  -3: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
  -4: 'CONTROL_FAILED',
  -10: 'START_STATE_IN_COLLISION',
  -12: 'GOAL_IN_COLLISION',
  -15: 'INVALID_GROUP_NAME',
  -21: 'FRAME_TRANSFORM_FAILURE',
  -23: 'ROBOT_STATE_STALE',
  -26: 'START_STATE_INVALID',
  -27: 'GOAL_STATE_INVALID',
  -31: 'NO_IK_SOLUTION',
}


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
  cy = math.cos(yaw * 0.5)
  sy = math.sin(yaw * 0.5)
  cp = math.cos(pitch * 0.5)
  sp = math.sin(pitch * 0.5)
  cr = math.cos(roll * 0.5)
  sr = math.sin(roll * 0.5)
  q = Quaternion()
  q.x = sr * cp * cy - cr * sp * sy
  q.y = cr * sp * cy + sr * cp * sy
  q.z = cr * cp * sy - sr * sp * cy
  q.w = cr * cp * cy + sr * sp * sy
  return q


def _moveit_error_explain(val: int) -> str:
  name = _MOVEIT_ERROR_HINTS.get(val, 'UNKNOWN')
  extra = ''
  if val == -26:
    extra = ' (start state invalid)'
  elif val == -31:
    extra = ' (no IK)'
  elif val == -21:
    extra = ' (TF / frame)'
  elif val == -12:
    extra = ' (goal in collision)'
  elif val == -10:
    extra = ' (start in collision)'
  elif val == -4:
    extra = ' (trajectory execution / controller; overlapping goals or invalidation)'
  elif val == -3:
    extra = ' (planning scene changed during execution)'
  return f'{name}{extra}'


def _collision_object_id(m: Marker) -> str:
  if m.ns:
    return f'{m.ns}_{m.id}'
  return f'obstacle_marker_{m.id}'


def _markers_to_planning_scene(msg: MarkerArray, stamp, logger) -> PlanningScene | None:
  scene = PlanningScene()
  scene.is_diff = True
  count = 0
  for m in msg.markers:
    if m.action in (Marker.DELETE, Marker.DELETEALL):
      continue
    if m.type == Marker.SPHERE:
      if m.scale.x <= 0.0:
        logger.warning(f'Skip sphere {_collision_object_id(m)}: non-positive scale.x')
        continue
      radius = float(m.scale.x) / 2.0
      prim = SolidPrimitive()
      prim.type = SolidPrimitive.SPHERE
      prim.dimensions = [radius]
      pose = m.pose
      frame = m.header.frame_id
    elif m.type == Marker.CUBE:
      prim = SolidPrimitive()
      prim.type = SolidPrimitive.BOX
      prim.dimensions = [
        float(max(m.scale.x, 1e-6)),
        float(max(m.scale.y, 1e-6)),
        float(max(m.scale.z, 1e-6)),
      ]
      pose = m.pose
      frame = m.header.frame_id
    else:
      continue

    co = CollisionObject()
    co.header.frame_id = frame if frame else DEFAULT_PLANNING_FRAME
    co.header.stamp = stamp
    co.id = _collision_object_id(m)
    co.primitives.append(prim)
    co.primitive_poses.append(pose)
    co.operation = CollisionObject.ADD
    scene.world.collision_objects.append(co)
    count += 1

  if count == 0:
    logger.warning('No SPHERE/CUBE markers to apply; empty MarkerArray?')
    return None
  return scene


def _marker_array_signature(msg: MarkerArray) -> tuple:
  """Stable tuple so identical obstacle republishes can be skipped."""
  rows = []
  for m in msg.markers:
    p = m.pose.position
    q = m.pose.orientation
    rows.append(
      (
        m.ns,
        int(m.id),
        int(m.type),
        round(float(m.scale.x), 6),
        round(float(m.scale.y), 6),
        round(float(m.scale.z), 6),
        round(float(p.x), 6),
        round(float(p.y), 6),
        round(float(p.z), 6),
        round(float(q.x), 6),
        round(float(q.y), 6),
        round(float(q.z), 6),
        round(float(q.w), 6),
      )
    )
  return tuple(sorted(rows, key=lambda r: (r[0], r[1])))


class PathPlanningNode(Node):
  def __init__(self) -> None:
    super().__init__('pathplanning_node')
    self._cb_group = ReentrantCallbackGroup()

    self.declare_parameter('obstacle_topic', '/obstacle_markers')
    self.declare_parameter('apply_planning_scene_service', '')
    self.declare_parameter('move_group_namespace', '/move_group')
    self.declare_parameter('goal_mode', 'pose')
    self.declare_parameter('auto_plan', True)
    self.declare_parameter('min_plan_interval_sec', 0.5)
    self.declare_parameter('scene_settle_sec', 0.05)
    self.declare_parameter('goal_seed', -1)
    self.declare_parameter('goal_pose_x_min', 0.38)
    self.declare_parameter('goal_pose_x_max', 0.52)
    self.declare_parameter('goal_pose_y_min', 0.15)
    self.declare_parameter('goal_pose_y_max', 0.35)
    self.declare_parameter('goal_pose_z_min', 1.15)
    self.declare_parameter('goal_pose_z_max', 1.42)
    self.declare_parameter('goal_pose_yaw_min', -math.pi)
    self.declare_parameter('goal_pose_yaw_max', math.pi)
    self.declare_parameter('goal_joint_jitter_rad', 0.25)
    self.declare_parameter('home', False)

    custom = str(self.get_parameter('apply_planning_scene_service').value).strip()
    if custom:
      srv = custom if custom.startswith('/') else '/' + custom
    else:
      raw = str(self.get_parameter('move_group_namespace').value).strip()
      low = raw.lower()
      if low in ('', 'root', '/'):
        srv = '/apply_planning_scene'
      else:
        ns = raw if raw.startswith('/') else '/' + raw
        ns = ns.rstrip('/')
        srv = f'{ns}/apply_planning_scene'

    self._apply_scene_cli = self.create_client(ApplyPlanningScene, srv)
    self.get_logger().info(f'ApplyPlanningScene service: {srv}')

    raw = str(self.get_parameter('move_group_namespace').value).strip()
    low = raw.lower()
    if low in ('', 'root', '/'):
      self._mg_ns = ''
      self._move_group = ActionClient(
        self, MoveGroup, '/move_action', callback_group=self._cb_group
      )
      self._execute = ActionClient(
        self, ExecuteTrajectory, '/execute_trajectory', callback_group=self._cb_group
      )
    else:
      ns = raw if raw.startswith('/') else '/' + raw
      ns = ns.rstrip('/') or '/move_group'
      self._mg_ns = ns
      self._move_group = ActionClient(
        self, MoveGroup, f'{ns}/move_action', callback_group=self._cb_group
      )
      self._execute = ActionClient(
        self, ExecuteTrajectory, f'{ns}/execute_trajectory', callback_group=self._cb_group
      )

    self._last_plan_mono = -1e10
    self._last_marker_sig: tuple | None = None
    self._obstacle_lock = threading.Lock()
    # Match obstacle_node publisher (VOLATILE) so RViz and this node both work.
    self._qos = QoSProfile(
      depth=10,
      durability=DurabilityPolicy.VOLATILE,
      reliability=ReliabilityPolicy.RELIABLE,
    )

    self._init_goal_cache()

  def _init_goal_cache(self) -> None:
    home = self.get_parameter('home').value
    if isinstance(home, bool):
      home_on = home
    else:
      home_on = str(home).strip().lower() in ('true', '1', 'yes')

    if home_on:
      self._cached_joint_targets = [0.0] * 8
      self._cached_goal_xyz = (0.45, 0.25, 1.35)
      self._cached_goal_quat = _quat_from_rpy(0.0, math.pi, 0.0)
      self.get_logger().info(
        'home=true: joint goal is arm_l seven zeros (goal_mode is ignored for motion).'
      )
      return

    raw_gs = self.get_parameter('goal_seed').value
    try:
      goal_seed_param = int(raw_gs)
    except (TypeError, ValueError):
      goal_seed_param = -1

    if goal_seed_param >= 0:
      rng = random.Random(goal_seed_param)
      seed_note = f'goal_seed={goal_seed_param} (same goal every restart with this seed)'
    else:
      ephemeral = secrets.randbelow(2**63)
      rng = random.Random(ephemeral)
      seed_note = f'goal_seed=-1, ephemeral={ephemeral} (new goal each node start)'

    x0 = float(self.get_parameter('goal_pose_x_min').value)
    x1 = float(self.get_parameter('goal_pose_x_max').value)
    y0 = float(self.get_parameter('goal_pose_y_min').value)
    y1 = float(self.get_parameter('goal_pose_y_max').value)
    z0 = float(self.get_parameter('goal_pose_z_min').value)
    z1 = float(self.get_parameter('goal_pose_z_max').value)
    yaw0 = float(self.get_parameter('goal_pose_yaw_min').value)
    yaw1 = float(self.get_parameter('goal_pose_yaw_max').value)

    lo, hi = min(x0, x1), max(x0, x1)
    x = rng.uniform(lo, hi)
    lo, hi = min(y0, y1), max(y0, y1)
    y = rng.uniform(lo, hi)
    lo, hi = min(z0, z1), max(z0, z1)
    z = rng.uniform(lo, hi)
    lo, hi = min(yaw0, yaw1), max(yaw0, yaw1)
    yaw = rng.uniform(lo, hi)

    self._cached_goal_xyz = (x, y, z)
    self._cached_goal_quat = _quat_from_rpy(0.0, math.pi, yaw)

    jitter = float(self.get_parameter('goal_joint_jitter_rad').value)
    nominal = [-0.57, 0.57, 1.57, 1.57, 0.0, 0.57, 0.0]
    self._cached_joint_targets = [b + rng.uniform(-jitter, jitter) for b in nominal]

    self.get_logger().info(
      f'{seed_note} | cached pose base_link xyz=({x:.4f},{y:.4f},{z:.4f}) yaw={yaw:.4f}'
    )
    self.get_logger().info(
      f'Cached joint goal (rad): {[round(t, 4) for t in self._cached_joint_targets]}'
    )

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

  def _wait_service(self, timeout_sec: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
      if self._apply_scene_cli.service_is_ready():
        return True
      rclpy.spin_once(self, timeout_sec=0.05)
    self.get_logger().error('Timeout: apply_planning_scene not available')
    return False

  def _wait_action_server(self, client: ActionClient, name: str, timeout_sec: float = 60.0) -> bool:
    self.get_logger().info(f'Waiting for {name}...')
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
      if client.server_is_ready():
        return True
      rclpy.spin_once(self, timeout_sec=0.05)
    self.get_logger().error(f'Timeout: {name}')
    if shutil.which('ros2'):
      try:
        out = subprocess.run(
          ['ros2', 'action', 'list'],
          capture_output=True,
          text=True,
          timeout=5,
          check=False,
        )
        if (out.stdout or '').strip():
          self.get_logger().error(f'ros2 action list:\n{out.stdout}')
      except Exception:
        pass
    return False

  def _wait_future_done(self, fut, timeout_sec: float = 120.0) -> bool:
    """Block without calling executor.spin* (re-entrant safe). Executor keeps spinning elsewhere."""
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
    if not self._wait_service():
      return False
    if not self._wait_action_server(self._move_group, 'MoveGroup'):
      return False
    if not self._wait_action_server(self._execute, 'ExecuteTrajectory'):
      return False
    return True

  def _call_apply_planning_scene(self, scene: PlanningScene) -> bool:
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

  def _send_move_group(self, request: MotionPlanRequest) -> bool:
    goal = MoveGroup.Goal()
    goal.request = request
    goal.planning_options = PlanningOptions()
    goal.planning_options.plan_only = False
    goal.planning_options.look_around = False
    goal.planning_options.replan = False

    send_future = self._move_group.send_goal_async(goal)
    if not self._wait_future_done(send_future):
      return False
    gh = send_future.result()
    if not gh or not gh.accepted:
      self.get_logger().error('MoveGroup goal rejected')
      return False
    result_future = gh.get_result_async()
    if not self._wait_future_done(result_future):
      return False
    result = result_future.result().result
    if result.error_code.val != MoveItErrorCodes.SUCCESS:
      v = result.error_code.val
      self.get_logger().error(f'MoveGroup failed: {v} {_moveit_error_explain(v)}')
      return False
    self.get_logger().info('MoveGroup plan+execute succeeded.')
    return True

  def _build_pose_goal(self) -> MotionPlanRequest:
    x, y, z = self._cached_goal_xyz
    q = self._cached_goal_quat
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [0.01]
    target = Pose()
    target.position.x = x
    target.position.y = y
    target.position.z = z
    target.orientation = q
    bv = BoundingVolume()
    bv.primitives.append(sphere)
    bv.primitive_poses.append(target)
    pc = PositionConstraint()
    pc.header.frame_id = DEFAULT_PLANNING_FRAME
    pc.header.stamp = self.get_clock().now().to_msg()
    pc.link_name = DEFAULT_EE_LINK
    pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
    pc.constraint_region = bv
    pc.weight = 1.0
    oc = OrientationConstraint()
    oc.header.frame_id = DEFAULT_PLANNING_FRAME
    oc.header.stamp = self.get_clock().now().to_msg()
    oc.link_name = DEFAULT_EE_LINK
    oc.orientation = q
    oc.absolute_x_axis_tolerance = 0.1
    oc.absolute_y_axis_tolerance = 0.1
    oc.absolute_z_axis_tolerance = 0.1
    oc.weight = 1.0
    goal_c = Constraints()
    goal_c.position_constraints.append(pc)
    goal_c.orientation_constraints.append(oc)
    req = MotionPlanRequest()
    req.group_name = DEFAULT_GROUP
    req.goal_constraints = [goal_c]
    req.num_planning_attempts = 15
    req.allowed_planning_time = 20.0
    req.max_velocity_scaling_factor = 0.3
    req.max_acceleration_scaling_factor = 0.3
    return req

  def _build_joint_goal_from_targets(self, targets: list[float]) -> MotionPlanRequest:
    if len(targets) != len(ARM_L_JOINTS):
      raise ValueError('targets must have same length as ARM_L_JOINTS')
    constraints = Constraints()
    for name, pos in zip(ARM_L_JOINTS, targets):
      jc = JointConstraint()
      jc.joint_name = name
      jc.position = float(pos)
      jc.tolerance_above = 0.05
      jc.tolerance_below = 0.05
      jc.weight = 1.0
      constraints.joint_constraints.append(jc)
    req = MotionPlanRequest()
    req.group_name = DEFAULT_GROUP
    req.goal_constraints = [constraints]
    req.num_planning_attempts = 15
    req.allowed_planning_time = 15.0
    req.max_velocity_scaling_factor = 0.3
    req.max_acceleration_scaling_factor = 0.3
    return req

  def _build_joint_goal(self) -> MotionPlanRequest:
    return self._build_joint_goal_from_targets(self._cached_joint_targets)

  def _on_obstacles(self, msg: MarkerArray) -> None:
    with self._obstacle_lock:
      sig = _marker_array_signature(msg)
      if self._last_marker_sig is not None and sig == self._last_marker_sig:
        return

      stamp = self.get_clock().now().to_msg()
      scene = _markers_to_planning_scene(msg, stamp, self.get_logger())
      if scene is None:
        return

      if not self._call_apply_planning_scene(scene):
        return

      self._last_marker_sig = sig

      settle = float(self.get_parameter('scene_settle_sec').value)
      if settle > 0.0:
        time.sleep(settle)

      if not bool(self.get_parameter('auto_plan').value):
        self.get_logger().info('auto_plan is false; planning scene updated, no motion.')
        return

      now = time.monotonic()
      min_dt = float(self.get_parameter('min_plan_interval_sec').value)
      if now - self._last_plan_mono < min_dt:
        self.get_logger().info('Planning scene updated; motion skipped (min_plan_interval_sec).')
        return

      home = self.get_parameter('home').value
      home_on = home if isinstance(home, bool) else str(home).strip().lower() in ('true', '1', 'yes')
      if home_on:
        req = self._build_joint_goal_from_targets([0.0] * len(ARM_L_JOINTS))
      else:
        mode = str(self.get_parameter('goal_mode').value).strip().lower()
        if mode == 'joint':
          req = self._build_joint_goal()
        else:
          req = self._build_pose_goal()

      ok = self._send_move_group(req)
      if ok:
        self._last_plan_mono = time.monotonic()


def main() -> None:
  rclpy.init()
  node = PathPlanningNode()
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
