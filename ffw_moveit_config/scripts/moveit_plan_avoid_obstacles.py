#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Plan and execute a motion that respects the current planning scene (world
# collision objects). Run moveit_planning_scene_obstacles.py first to place boxes,
# then this script: OMPL / MoveIt will avoid those objects automatically.
#
# Same MoveGroup action pipeline as scripts/moveit_control_example.py (pose or joint goal).
#
# Prerequisites:
#   ros2 launch ffw_moveit_config moveit.launch.py
# Optional (recommended for obstacle demo):
#   ros2 run ffw_moveit_config moveit_planning_scene_obstacles.py
#
# Examples:
#   ros2 run ffw_moveit_config moveit_plan_avoid_obstacles.py --mode pose
#   ros2 run ffw_moveit_config moveit_plan_avoid_obstacles.py --mode joint
#   ros2 run ffw_moveit_config moveit_plan_avoid_obstacles.py --ros-args -p move_group_namespace:=root

import argparse
import math
import shutil
import subprocess
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

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
)
from shape_msgs.msg import SolidPrimitive

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
    extra = (
      ' (start state invalid: /joint_states, TF, or planning scene sync)'
    )
  elif val == -31:
    extra = ' (no IK for this pose)'
  elif val == -21:
    extra = ' (check planning frame vs TF)'
  elif val == -12:
    extra = ' (goal overlaps collision geometry — move target or shrink obstacles)'
  elif val == -10:
    extra = ' (current state collides scene — move arm or remove obstacles)'
  return f'{name}{extra}'


class MoveItAvoidObstacles(Node):
  def __init__(self) -> None:
    super().__init__('moveit_plan_avoid_obstacles')
    self.declare_parameter('move_group_namespace', '/move_group')
    raw = self.get_parameter('move_group_namespace').value
    ns = str(raw).strip()
    low = ns.lower()
    if low in ('', 'root', '/'):
      self._mg_ns = ''
      self._move_group = ActionClient(self, MoveGroup, '/move_action')
      self._execute = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
    else:
      if not ns.startswith('/'):
        ns = '/' + ns
      ns = ns.rstrip('/')
      if not ns:
        ns = '/move_group'
      self._mg_ns = ns
      self._move_group = ActionClient(self, MoveGroup, f'{ns}/move_action')
      self._execute = ActionClient(self, ExecuteTrajectory, f'{ns}/execute_trajectory')

  def _log_move_group_troubleshooting(self) -> None:
    log = self.get_logger().error
    log('MoveGroup action server is not available.')
    log('  Start: ros2 launch ffw_moveit_config moveit.launch.py')
    log(
      '  If ros2 action list shows /move_action at root: '
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
        text = (out.stdout or '').strip()
        if text:
          log(f'ros2 action list:\n{text}')
      except Exception as e:
        log(f'Could not run ros2 action list: {e}')

  def _wait_action_server(self, client: ActionClient, name: str, timeout_sec: float = 60.0) -> bool:
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

  def _send_move_group(self, request: MotionPlanRequest) -> bool:
    goal = MoveGroup.Goal()
    goal.request = request
    goal.planning_options = PlanningOptions()
    goal.planning_options.plan_only = False
    goal.planning_options.look_around = False
    goal.planning_options.replan = False

    self.get_logger().info('Planning with current planning scene (world obstacles included)...')
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
    self.get_logger().info('Planned and executed (collision-aware path).')
    return True

  def move_joint_goal(
    self,
    joint_positions: list[float],
    joint_names: list[str] | None = None,
    group_name: str = DEFAULT_GROUP,
  ) -> bool:
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
    req.num_planning_attempts = 15
    req.allowed_planning_time = 15.0
    req.max_velocity_scaling_factor = 0.3
    req.max_acceleration_scaling_factor = 0.3

    return self._send_move_group(req)

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

    req = MotionPlanRequest()
    req.group_name = group_name
    req.goal_constraints = [goal_c]
    req.num_planning_attempts = 15
    req.allowed_planning_time = 20.0
    req.max_velocity_scaling_factor = 0.3
    req.max_acceleration_scaling_factor = 0.3

    return self._send_move_group(req)

  def run_demo_pose(self) -> bool:
    # Mirror moveit_control_example.py; tune if the goal sits inside your demo box.
    q = _quat_from_rpy(0.0, math.pi, 0.0)
    return self.move_pose_goal(0.45, 0.25, 1.35, q)

  def run_demo_joint(self) -> bool:
    targets = [-0.57, 0.57, 1.57, 1.57, 0.0, 0.57, 0.0]
    return self.move_joint_goal(targets)


def main() -> int:
  rclpy.init()
  parser = argparse.ArgumentParser(
    description='Collision-aware motion planning (uses planning scene obstacles)'
  )
  parser.add_argument(
    '--mode',
    choices=['pose', 'joint'],
    default='pose',
    help='Goal type: Cartesian pose in base_link, or named joint targets',
  )
  args = parser.parse_args(remove_ros_args(sys.argv)[1:])

  node = MoveItAvoidObstacles()
  if node.has_parameter('use_sim_time') and node.get_parameter('use_sim_time').value:
    node.get_logger().info('use_sim_time is true — match simulation clock if using sim.')

  node.get_logger().info(
    'Ensure world obstacles are applied (e.g. moveit_planning_scene_obstacles.py) '
    'before planning; this node does not add objects.'
  )

  if not node._wait_action_server(node._move_group, f'MoveGroup {node._mg_ns}/move_action'):
    node.destroy_node()
    rclpy.shutdown()
    return 1
  if not node._wait_action_server(node._execute, f'ExecuteTrajectory {node._mg_ns}/execute_trajectory'):
    node.destroy_node()
    rclpy.shutdown()
    return 1

  try:
    if args.mode == 'pose':
      ok = node.run_demo_pose()
    else:
      ok = node.run_demo_joint()
    return 0 if ok else 1
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  sys.exit(main())
