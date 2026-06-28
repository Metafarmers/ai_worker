#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# MoveIt 2 control examples for ffw (ROS 2 Humble):
# - Joint-space goal (industrial "MoveJ"-style: plan in joint space)
# - Cartesian pose goal (PTP to a target pose in the planning frame)
# - Cartesian path (industrial "MoveL"-style straight segments via compute_cartesian_path)
# - movel_rel: relative linear move in base_link or tool (EE) frame from current pose
#
# Prerequisites: move_group running, e.g. ros2 launch ffw_moveit_config moveit.launch.py
# With Gazebo/sim and /clock: pass --ros-args -p use_sim_time:=true
#
# MoveGroup action path (check: ros2 action list | grep move_action)
# - Default: /move_group/move_action (typical MoveIt)
# - If you see /move_action (no move_group prefix): --ros-args -p move_group_namespace:=root

import argparse
import math
import shutil
import subprocess
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
import tf2_ros

from geometry_msgs.msg import Pose, Quaternion, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
  BoundingVolume,
  Constraints,
  JointConstraint,
  MotionPlanRequest,
  MoveItErrorCodes,
  OrientationConstraint,
  PlanningOptions,
  PositionConstraint,
  RobotState,
)
from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive


# Planning groups and tip links (from ffw.srdf / URDF)
DEFAULT_GROUP = 'arm_l'
DEFAULT_GROUP_DUAL = 'dual_arms'
DEFAULT_EE_LINK = 'end_effector_l_link'
DEFAULT_EE_LINK_R = 'end_effector_r_link'
DEFAULT_PLANNING_FRAME = 'base_link'

# Seven main arm joints (fixed EE joint is not actuated)
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
DUAL_ARM_JOINTS = ['lift_joint'] + ARM_L_JOINTS + ARM_R_JOINTS


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


def _quat_rotate_vector(q: Quaternion, vx: float, vy: float, vz: float) -> tuple:
  """Rotate vector (vx, vy, vz) by unit quaternion q (vector in base frame)."""
  x, y, z = q.x, q.y, q.z
  w = q.w
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  return (
    vx + w * tx + y * tz - z * ty,
    vy + w * ty + z * tx - x * tz,
    vz + w * tz + x * ty - y * tx,
  )


# moveit_msgs/msg/MoveItErrorCodes.msg (common failures)
_MOVEIT_ERROR_HINTS = {
  -1: 'PLANNING_FAILED',
  -2: 'INVALID_MOTION_PLAN',
  -10: 'START_STATE_IN_COLLISION',
  -12: 'GOAL_IN_COLLISION',
  -15: 'INVALID_GROUP_NAME',
  -21: 'FRAME_TRANSFORM_FAILURE',
  -23: 'ROBOT_STATE_STALE',
  -26: 'START_STATE_INVALID',
  -27: 'GOAL_STATE_INVALID',
  -31: 'NO_IK_SOLUTION',
}


def _moveit_error_explain(val: int) -> str:
  name = _MOVEIT_ERROR_HINTS.get(val, 'UNKNOWN')
  extra = ''
  if val == -26:
    extra = (
      ' (start state invalid: incomplete /joint_states, missing TF, or planning scene '
      'out of sync; fix sim time and ensure joint_state_broadcaster publishes all arm joints)'
    )
  elif val == -31:
    extra = ' (no IK for this pose; change position/orientation or seed from current state)'
  elif val == -21:
    extra = ' (check planning frame vs TF, e.g. base_link)'
  return f'{name}{extra}'


class MoveItControlExample(Node):
  def __init__(self) -> None:
    super().__init__('moveit_control_example')
    self.declare_parameter('move_group_namespace', '/move_group')
    raw = self.get_parameter('move_group_namespace').value
    ns = str(raw).strip()
    low = ns.lower()
    # Some launches remap actions to the root: /move_action, /execute_trajectory, /compute_cartesian_path
    if low in ('', 'root', '/'):
      self._mg_ns = ''
      self._move_group = ActionClient(self, MoveGroup, '/move_action')
      self._execute = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
      self._cartesian = self.create_client(GetCartesianPath, '/compute_cartesian_path')
    else:
      if not ns.startswith('/'):
        ns = '/' + ns
      ns = ns.rstrip('/')
      if not ns:
        ns = '/move_group'
      self._mg_ns = ns
      self._move_group = ActionClient(self, MoveGroup, f'{ns}/move_action')
      self._execute = ActionClient(self, ExecuteTrajectory, f'{ns}/execute_trajectory')
      self._cartesian = self.create_client(GetCartesianPath, f'{ns}/compute_cartesian_path')

    self._tf_buffer = tf2_ros.Buffer()
    self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

  def _lookup_ee_pose(self, planning_frame: str, ee_link: str, timeout_sec: float = 10.0) -> Pose | None:
    deadline = time.monotonic() + timeout_sec
    last_err = None
    while time.monotonic() < deadline:
      rclpy.spin_once(self, timeout_sec=0.05)
      try:
        tfm = self._tf_buffer.lookup_transform(
          planning_frame,
          ee_link,
          rclpy.time.Time(),
          timeout=Duration(seconds=1.0),
        )
        p = Pose()
        p.position.x = tfm.transform.translation.x
        p.position.y = tfm.transform.translation.y
        p.position.z = tfm.transform.translation.z
        p.orientation = tfm.transform.rotation
        return p
      except Exception as e:
        last_err = e
    self.get_logger().error(
      f'TF failed for {planning_frame} <- {ee_link}: {last_err}'
    )
    return None

  def move_cartesian_relative_linear(
    self,
    dx: float,
    dy: float,
    dz: float,
    delta_frame: str = 'base',
    group_name: str = DEFAULT_GROUP,
    ee_link: str = DEFAULT_EE_LINK,
    planning_frame: str = DEFAULT_PLANNING_FRAME,
    max_step: float = 0.01,
  ) -> bool:
    """
    MoveL-style straight segment from current EE pose by (dx,dy,dz).
    delta_frame: 'base' = deltas along planning_frame axes; 'tool' = deltas along EE axes.
    """
    pose = self._lookup_ee_pose(planning_frame, ee_link)
    if pose is None:
      return False
    if max(abs(dx), abs(dy), abs(dz)) > 2.0:
      self.get_logger().warn(
        f'Large relative delta ({dx},{dy},{dz}) m — did you mean centimeters? '
        'Values are in meters (0.05 = 5 cm).'
      )
    goal = Pose()
    if delta_frame == 'tool':
      rx, ry, rz = _quat_rotate_vector(pose.orientation, dx, dy, dz)
      goal.position.x = pose.position.x + rx
      goal.position.y = pose.position.y + ry
      goal.position.z = pose.position.z + rz
    else:
      goal.position.x = pose.position.x + dx
      goal.position.y = pose.position.y + dy
      goal.position.z = pose.position.z + dz
    goal.orientation = pose.orientation
    self.get_logger().info(
      f'MoveL relative ({delta_frame}): delta=({dx},{dy},{dz}) -> '
      f'goal pos=({goal.position.x:.4f},{goal.position.y:.4f},{goal.position.z:.4f})'
    )
    return self.move_cartesian_linear([goal], group_name, ee_link, max_step=max_step)

  def _log_move_group_troubleshooting(self) -> None:
    log = self.get_logger().error
    log('MoveGroup action server is not available. Check the following:')
    log(
      '  1) Start MoveIt in another terminal: '
      'ros2 launch ffw_moveit_config moveit.launch.py'
    )
    log(
      '  2) If you use simulation time / Gazebo, start launch with use_sim:=true '
      'and run this node with --ros-args -p use_sim_time:=true (must match /clock).'
    )
    log(
      '  3) Without simulation, omit use_sim_time: '
      'ros2 run ffw_moveit_config moveit_control_example.py --mode joint'
    )
    log(
      '  4) Same ROS_DOMAIN_ID in every terminal; use same ROS_LOCALHOST_ONLY if set.'
    )
    log(
      f'  5) Expecting MoveGroup actions under namespace: '
      f'{"(root) /move_action" if not self._mg_ns else self._mg_ns}'
    )
    log(
      '  6) If ros2 action list shows /move_action (not /move_group/move_action), run with: '
      '--ros-args -p move_group_namespace:=root'
    )
    if shutil.which('ros2'):
      try:
        out = subprocess.run(
          ['ros2', 'action', 'list'],
          capture_output=True,
          text=True,
          timeout=5,
          check=False,
        )
        text = (out.stdout or '').strip() or (out.stderr or '').strip()
        if text:
          log(f'ros2 action list (this process):\n{text}')
        else:
          log('ros2 action list returned no actions (no move_group or wrong domain).')
      except Exception as e:
        log(f'Could not run ros2 action list: {e}')

  def _wait_action_server(self, client: ActionClient, name: str, timeout_sec: float = 60.0) -> bool:
    """Wait for an action server while spinning (rclpy wait_for_server only sleeps; graph may not update)."""
    self.get_logger().info(f'Waiting for {name}...')
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
      if client.server_is_ready():
        return True
      rclpy.spin_once(self, timeout_sec=0.1)
    if client.server_is_ready():
      return True
    self.get_logger().error(f'Timeout waiting for {name}')
    self._log_move_group_troubleshooting()
    return False

  def _wait_service_server(self, client, name: str, timeout_sec: float = 60.0) -> bool:
    """Wait for a service while spinning (same issue as actions in rclpy)."""
    self.get_logger().info(f'Waiting for {name}...')
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
      if client.service_is_ready():
        return True
      rclpy.spin_once(self, timeout_sec=0.1)
    if client.service_is_ready():
      return True
    self.get_logger().error(f'Timeout waiting for {name}')
    return False

  def _send_move_group(self, request: MotionPlanRequest, plan_only: bool = False) -> bool:
    goal = MoveGroup.Goal()
    goal.request = request
    goal.planning_options = PlanningOptions()
    goal.planning_options.plan_only = plan_only
    goal.planning_options.look_around = False
    goal.planning_options.replan = False

    self.get_logger().info('Sending MoveGroup goal...')
    send_future = self._move_group.send_goal_async(goal)
    rclpy.spin_until_future_complete(self, send_future)
    goal_handle = send_future.result()
    if not goal_handle.accepted:
      self.get_logger().error('MoveGroup goal rejected')
      return False

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(self, result_future)
    result = result_future.result().result
    if result.error_code.val != MoveItErrorCodes.SUCCESS:
      v = result.error_code.val
      self.get_logger().error(
        f'MoveGroup failed: val={v} {_moveit_error_explain(v)}'
      )
      return False
    self.get_logger().info('MoveGroup succeeded.')
    return True

  def _robot_trajectory_has_points(self, trajectory) -> bool:
    jt = trajectory.joint_trajectory
    if jt.points:
      return True
    md = trajectory.multi_dof_joint_trajectory
    return bool(md.points)

  def _send_execute_trajectory(self, trajectory) -> bool:
    if not self._robot_trajectory_has_points(trajectory):
      self.get_logger().error(
        'Refusing to execute empty trajectory (no joint trajectory points).'
      )
      return False
    goal = ExecuteTrajectory.Goal()
    goal.trajectory = trajectory

    send_future = self._execute.send_goal_async(goal)
    rclpy.spin_until_future_complete(self, send_future)
    goal_handle = send_future.result()
    if not goal_handle.accepted:
      self.get_logger().error('ExecuteTrajectory goal rejected')
      return False

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(self, result_future)
    result = result_future.result().result
    if result.error_code.val != MoveItErrorCodes.SUCCESS:
      self.get_logger().error(
        f'ExecuteTrajectory failed: val={result.error_code.val}'
      )
      return False
    self.get_logger().info('Trajectory execution finished.')
    return True

  def move_joint_goal(
    self,
    joint_positions: list,
    joint_names: list | None = None,
    group_name: str = DEFAULT_GROUP,
  ) -> bool:
    """Plan and execute a joint-space motion (MoveJ-style)."""
    if joint_names is None:
      joint_names = ARM_L_JOINTS
    if len(joint_positions) != len(joint_names):
      self.get_logger().error('joint_positions length must match joint_names')
      return False

    constraints = Constraints()
    for name, pos in zip(joint_names, joint_positions):
      jc = JointConstraint()
      jc.joint_name = name
      jc.position = float(pos)
      jc.tolerance_above = 0.05
      jc.tolerance_below = 0.05
      jc.weight = 1.0
      constraints.joint_constraints.append(jc)

    req = MotionPlanRequest()
    req.group_name = group_name
    req.goal_constraints = [constraints]
    req.num_planning_attempts = 10
    req.allowed_planning_time = 10.0
    req.max_velocity_scaling_factor = 0.3
    req.max_acceleration_scaling_factor = 0.3

    return self._send_move_group(req, plan_only=False)

  def _pose_goal_constraints(
    self,
    x: float,
    y: float,
    z: float,
    q: Quaternion,
    ee_link: str,
    frame_id: str = DEFAULT_PLANNING_FRAME,
  ) -> Constraints:
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
    pc.header.frame_id = frame_id
    pc.header.stamp = self.get_clock().now().to_msg()
    pc.link_name = ee_link
    pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
    pc.constraint_region = bv
    pc.weight = 1.0

    oc = OrientationConstraint()
    oc.header.frame_id = frame_id
    oc.header.stamp = self.get_clock().now().to_msg()
    oc.link_name = ee_link
    oc.orientation = q
    oc.absolute_x_axis_tolerance = 0.1
    oc.absolute_y_axis_tolerance = 0.1
    oc.absolute_z_axis_tolerance = 0.1
    oc.weight = 1.0

    goal_c = Constraints()
    goal_c.position_constraints.append(pc)
    goal_c.orientation_constraints.append(oc)
    return goal_c

  def move_pose_goal(
    self,
    x: float,
    y: float,
    z: float,
    q: Quaternion,
    group_name: str = DEFAULT_GROUP,
    ee_link: str = DEFAULT_EE_LINK,
    frame_id: str = DEFAULT_PLANNING_FRAME,
  ) -> bool:
    """Plan and execute to a Cartesian pose in frame_id (PTP / pose target)."""
    goal_c = self._pose_goal_constraints(x, y, z, q, ee_link, frame_id)

    req = MotionPlanRequest()
    req.group_name = group_name
    req.goal_constraints = [goal_c]
    req.num_planning_attempts = 10
    req.allowed_planning_time = 15.0
    req.max_velocity_scaling_factor = 0.3
    req.max_acceleration_scaling_factor = 0.3

    return self._send_move_group(req, plan_only=False)

  def move_dual_joint_goal(
    self,
    joint_positions: list,
    joint_names: list | None = None,
    group_name: str = DEFAULT_GROUP_DUAL,
  ) -> bool:
    """Plan and execute both arms (+ lift) in joint space via dual_arms group."""
    if joint_names is None:
      joint_names = DUAL_ARM_JOINTS
    return self.move_joint_goal(joint_positions, joint_names, group_name)

  def move_dual_pose_goal(
    self,
    left_xyz: tuple,
    left_q: Quaternion,
    right_xyz: tuple,
    right_q: Quaternion,
    group_name: str = DEFAULT_GROUP_DUAL,
    frame_id: str = DEFAULT_PLANNING_FRAME,
  ) -> bool:
    """Plan both EEs simultaneously (constraints on left + right in one goal)."""
    goal_c = Constraints()
    left_c = self._pose_goal_constraints(
      left_xyz[0], left_xyz[1], left_xyz[2], left_q, DEFAULT_EE_LINK, frame_id
    )
    right_c = self._pose_goal_constraints(
      right_xyz[0], right_xyz[1], right_xyz[2], right_q, DEFAULT_EE_LINK_R, frame_id
    )
    goal_c.position_constraints.extend(left_c.position_constraints)
    goal_c.position_constraints.extend(right_c.position_constraints)
    goal_c.orientation_constraints.extend(left_c.orientation_constraints)
    goal_c.orientation_constraints.extend(right_c.orientation_constraints)

    req = MotionPlanRequest()
    req.group_name = group_name
    req.goal_constraints = [goal_c]
    req.num_planning_attempts = 15
    req.allowed_planning_time = 25.0
    req.max_velocity_scaling_factor = 0.25
    req.max_acceleration_scaling_factor = 0.25

    return self._send_move_group(req, plan_only=False)

  def move_cartesian_linear(
    self,
    waypoints: list,
    group_name: str = DEFAULT_GROUP,
    ee_link: str = DEFAULT_EE_LINK,
    max_step: float = 0.01,
    jump_threshold: float = 0.0,
  ) -> bool:
    """Follow straight-line segments in Cartesian space (MoveL-style)."""
    if not self._wait_service_server(
      self._cartesian, f'{self._mg_ns}/compute_cartesian_path', timeout_sec=60.0
    ):
      self.get_logger().error('compute_cartesian_path service not available')
      return False

    req = GetCartesianPath.Request()
    req.header.frame_id = DEFAULT_PLANNING_FRAME
    req.header.stamp = self.get_clock().now().to_msg()
    req.start_state = RobotState()
    req.group_name = group_name
    req.link_name = ee_link
    req.waypoints = waypoints
    req.max_step = max_step
    req.jump_threshold = jump_threshold
    req.prismatic_jump_threshold = 0.0
    req.revolute_jump_threshold = 0.0
    req.avoid_collisions = True

    future = self._cartesian.call_async(req)
    rclpy.spin_until_future_complete(self, future)
    resp = future.result()
    if resp is None:
      self.get_logger().error('Cartesian path service call failed')
      return False
    if resp.error_code.val != MoveItErrorCodes.SUCCESS:
      self.get_logger().error(
        f'compute_cartesian_path failed: val={resp.error_code.val} fraction={resp.fraction}'
      )
      return False
    if resp.fraction <= 0.0 or not self._robot_trajectory_has_points(resp.solution):
      self.get_logger().error(
        'Cartesian path has no valid motion (fraction=0 or empty trajectory). '
        'Goal is likely unreachable or outside the workspace. '
        'Deltas are in meters (e.g. -0.05 is 5 cm down), not centimeters.'
      )
      return False
    if resp.fraction < 0.99:
      self.get_logger().warn(
        f'Cartesian path incomplete: fraction={resp.fraction}'
      )

    self.get_logger().info(f'Cartesian path fraction={resp.fraction}, executing...')
    return self._send_execute_trajectory(resp.solution)

  def run_demo_joint(self) -> bool:
    # Small change from "init" pose (all zeros); adjust if your scene requires it.
    targets = [-0.57, 0.57, 1.57, 1.57, 0.0, 0.57, 0.0]
    # targets = [-0.0, 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0]
    self.get_logger().info('Demo: joint-space motion (MoveJ-style)')
    return self.move_joint_goal(targets)

  def run_demo_pose(self) -> bool:
    # Example pose in base_link — tune x,y,z for your workspace (use RViz to pick values).
    self.get_logger().info('Demo: pose goal (PTP to pose)')
    q = _quat_from_rpy(0.0, math.pi, 0.0)
    return self.move_pose_goal(0.45, 0.25, 1.35, q)

  def run_demo_cartesian(self) -> bool:
    # Two waypoints: small Cartesian motion from implicit start (current state).
    self.get_logger().info('Demo: Cartesian path (MoveL-style)')
    w0 = Pose()
    w0.position.x = 0.42
    w0.position.y = 0.22
    w0.position.z = 1.30
    w0.orientation = _quat_from_rpy(0.0, math.pi, 0.0)
    w1 = Pose()
    w1.position.x = 0.42
    w1.position.y = 0.28
    w1.position.z = 1.30
    w1.orientation = _quat_from_rpy(0.0, math.pi, 0.0)
    return self.move_cartesian_linear([w0, w1])

  def run_demo_dual_joint(self) -> bool:
    lift = 0.0
    left = [-0.3, 0.4, 1.2, 1.0, 0.0, 0.4, 0.0]
    right = [0.3, -0.4, -1.2, -1.0, 0.0, -0.4, 0.0]
    self.get_logger().info('Demo: dual-arm joint-space motion (dual_arms group)')
    return self.move_dual_joint_goal([lift] + left + right)

  def run_demo_dual_pose(self) -> bool:
    self.get_logger().info('Demo: dual-arm pose goal (left + right EE constraints)')
    q_l = _quat_from_rpy(0.0, math.pi, 0.0)
    q_r = _quat_from_rpy(0.0, math.pi, 0.0)
    return self.move_dual_pose_goal(
      (0.45, 0.25, 1.35),
      q_l,
      (0.45, -0.25, 1.35),
      q_r,
    )


def main() -> int:
  rclpy.init()
  parser = argparse.ArgumentParser(
    description='MoveIt control examples for ffw_moveit_config'
  )
  parser.add_argument(
    '--mode',
    choices=[
      'joint',
      'pose',
      'cartesian',
      'movel_rel',
      'dual_joint',
      'dual_pose',
      'all',
    ],
    default='joint',
    help='Which demo to run',
  )
  parser.add_argument(
    '--dx',
    type=float,
    default=0.0,
    help='Relative move delta x in meters (movel_rel), e.g. 0.02 = 2 cm',
  )
  parser.add_argument(
    '--dy',
    type=float,
    default=0.05,
    help='Relative move delta y in meters (movel_rel), e.g. 0.05 = 5 cm',
  )
  parser.add_argument(
    '--dz',
    type=float,
    default=0.0,
    help='Relative move delta z in meters (movel_rel); e.g. -0.05 = 5 cm down (not -5)',
  )
  parser.add_argument(
    '--delta-frame',
    choices=['base', 'tool'],
    default='base',
    help='Frame for dx,dy,dz: base=planning frame axes, tool=end-effector axes',
  )
  # Strip --ros-args / -p ... so argparse does not see them (rclpy.init already applied them).
  args = parser.parse_args(remove_ros_args(sys.argv)[1:])
  node = MoveItControlExample()
  if node.has_parameter('use_sim_time') and node.get_parameter('use_sim_time').value:
    node.get_logger().info(
      'use_sim_time is true: ensure /clock is published and '
      'moveit.launch.py was started with use_sim:=true.'
    )

  if not node._wait_action_server(node._move_group, f'MoveGroup {node._mg_ns}/move_action'):
    node.destroy_node()
    rclpy.shutdown()
    return 1
  if not node._wait_action_server(node._execute, f'ExecuteTrajectory {node._mg_ns}/execute_trajectory'):
    node.destroy_node()
    rclpy.shutdown()
    return 1

  ok = True
  try:
    if args.mode == 'joint':
      ok = node.run_demo_joint()
    elif args.mode == 'pose':
      ok = node.run_demo_pose()
    elif args.mode == 'cartesian':
      ok = node.run_demo_cartesian()
    elif args.mode == 'movel_rel':
      ok = node.move_cartesian_relative_linear(
        args.dx,
        args.dy,
        args.dz,
        delta_frame=args.delta_frame,
      )
    elif args.mode == 'dual_joint':
      ok = node.run_demo_dual_joint()
    elif args.mode == 'dual_pose':
      ok = node.run_demo_dual_pose()
    else:
      ok = node.run_demo_joint()
      time.sleep(1.0)
      ok = ok and node.run_demo_pose()
      time.sleep(1.0)
      ok = ok and node.run_demo_cartesian()
      time.sleep(1.0)
      ok = ok and node.move_cartesian_relative_linear(0.0, 0.03, 0.0, delta_frame='base')
  finally:
    node.destroy_node()
    rclpy.shutdown()

  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
