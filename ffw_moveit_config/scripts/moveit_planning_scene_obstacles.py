#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Add or remove world collision objects (spheres) in the MoveIt planning scene.
#
# Uses move_group's ApplyPlanningScene service — the same mechanism as the MoveIt
# planning scene tutorials (diff updates with CollisionObject ADD / REMOVE).
# Do NOT publish only to /monitored_planning_scene (RViz display); move_group
# listens on /planning_scene / apply_planning_scene for planning.
#
# Prerequisites: ros2 launch ffw_moveit_config moveit.launch.py
# With Gazebo/sim: add --ros-args -p use_sim_time:=true (match launch use_sim:=true)
#
# Examples:
#   ros2 run ffw_moveit_config moveit_planning_scene_obstacles.py
#   ros2 run ffw_moveit_config moveit_planning_scene_obstacles.py --seed 42
#   ros2 run ffw_moveit_config moveit_planning_scene_obstacles.py --remove-all-spheres
#   ros2 run ffw_moveit_config moveit_planning_scene_obstacles.py --legacy-box
#   ros2 run ffw_moveit_config moveit_planning_scene_obstacles.py --ros-args -p move_group_namespace:=root
#   ros2 run ... -p apply_planning_scene_service:=/apply_planning_scene   # full service override

import argparse
import math
import random
import secrets
import shutil
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive

DEFAULT_FRAME_ID = 'base_link'

# Sphere obstacle reset policy (aligned with training / env reset code).
OBSTACLE_SPHERE_RESET_PARAMS = {
  'obstacle_names': ('obstacle_sphere_1', 'obstacle_sphere_2', 'obstacle_sphere_3'),
  'min_obstacles': 1,
  'max_obstacles': 2,
  'sphere_radius': 0.05,
  'active_pose_range': {
    'x': (0.2, 0.4),
    'y': (-0.0, 0.4),
    'z': (0.9, 1.1),
    'yaw': (-math.pi, math.pi),
  },
  'inactive_park_z': -2.5,
}

# Legacy single-box demo (optional --legacy-box).
LEGACY_OBJECT_ID = 'demo_box'
DEFAULT_BOX_SIZE_XYZ = (0.15, 0.15, 0.35)
DEFAULT_BOX_POSITION_XYZ = (0.42, 0.24, 1.15)


def _quat_from_yaw(yaw: float) -> Quaternion:
  half = yaw * 0.5
  q = Quaternion()
  q.x = 0.0
  q.y = 0.0
  q.z = math.sin(half)
  q.w = math.cos(half)
  return q


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
  return lo + rng.random() * (hi - lo)


class PlanningSceneObstaclesNode(Node):
  def __init__(self) -> None:
    super().__init__('moveit_planning_scene_obstacles')

    self.declare_parameter('apply_planning_scene_service', '')
    custom = str(self.get_parameter('apply_planning_scene_service').value).strip()
    if custom:
      srv = custom if custom.startswith('/') else '/' + custom
    else:
      self.declare_parameter('move_group_namespace', '/move_group')
      raw = self.get_parameter('move_group_namespace').value
      ns = str(raw).strip()
      low = ns.lower()
      if low in ('', 'root', '/'):
        srv = '/apply_planning_scene'
      else:
        if not ns.startswith('/'):
          ns = '/' + ns
        ns = ns.rstrip('/')
        srv = f'{ns}/apply_planning_scene'

    self._apply_scene = self.create_client(ApplyPlanningScene, srv)
    self.get_logger().info(f'Using ApplyPlanningScene service: {srv}')

  def _log_service_troubleshooting(self) -> None:
    log = self.get_logger().error
    log(
      'apply_planning_scene did not appear in time. MoveIt move_group must be running '
      'in the same ROS 2 domain as this process.'
    )
    log('  1) Other terminal: ros2 launch ffw_moveit_config moveit.launch.py')
    log('     Sim: same launch with use_sim:=true, and pass use_sim_time:=true here.')
    log('  2) Same machine/session: echo $ROS_DOMAIN_ID on both terminals (must match).')
    log(
      '  3) If ros2 service list shows apply_planning_scene at root, try: '
      '--ros-args -p move_group_namespace:=root'
    )
    log(
      '  4) Override the exact service name: '
      '--ros-args -p apply_planning_scene_service:=/move_group/apply_planning_scene'
    )
    log('  5) Check: ros2 service list | grep apply_planning_scene')
    if shutil.which('ros2'):
      try:
        out = subprocess.run(
          ['ros2', 'service', 'list'],
          capture_output=True,
          text=True,
          timeout=8,
          check=False,
        )
        lines = [ln for ln in (out.stdout or '').splitlines() if 'apply_planning' in ln]
        if lines:
          log('ros2 service list (filtered):\n  ' + '\n  '.join(lines[:20]))
        elif (out.stdout or '').strip():
          log('No apply_planning_scene in ros2 service list — move_group likely not running.')
      except Exception as e:
        log(f'Could not run ros2 service list: {e}')

  def _wait_service(self, timeout_sec: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
      if self._apply_scene.service_is_ready():
        return True
      rclpy.spin_once(self, timeout_sec=0.1)
    self._log_service_troubleshooting()
    return False

  def apply_diff(self, scene: PlanningScene) -> bool:
    req = ApplyPlanningScene.Request()
    req.scene = scene
    fut = self._apply_scene.call_async(req)
    rclpy.spin_until_future_complete(self, fut)
    resp = fut.result()
    if resp is None:
      self.get_logger().error('ApplyPlanningScene call failed (no response)')
      return False
    if not resp.success:
      self.get_logger().error('ApplyPlanningScene returned success=false')
      return False
    return True

  def reset_obstacle_spheres(
    self,
    frame_id: str,
    params: dict | None = None,
    seed: int | None = None,
  ) -> bool:
    p = params if params is not None else OBSTACLE_SPHERE_RESET_PARAMS
    names = tuple(p['obstacle_names'])
    nmin = int(p['min_obstacles'])
    nmax = int(p['max_obstacles'])
    radius = float(p['sphere_radius'])
    park_z = float(p['inactive_park_z'])
    ar = p['active_pose_range']

    # Explicit entropy when --seed omitted (avoids any fixed/default seed surprises).
    if seed is None:
      seed = secrets.randbelow(2**63)
    rng = random.Random(seed)
    k_active = rng.randint(nmin, nmax)
    k_active = min(k_active, len(names))
    active_set = set(rng.sample(list(names), k=k_active))

    scene = PlanningScene()
    scene.is_diff = True
    stamp = self.get_clock().now().to_msg()

    active_samples: list[tuple[str, float, float, float, float]] = []

    for i, oid in enumerate(names):
      prim = SolidPrimitive()
      prim.type = SolidPrimitive.SPHERE
      prim.dimensions = [radius]

      pose = Pose()
      if oid in active_set:
        pose.position.x = _uniform(rng, ar['x'][0], ar['x'][1])
        pose.position.y = _uniform(rng, ar['y'][0], ar['y'][1])
        pose.position.z = _uniform(rng, ar['z'][0], ar['z'][1])
        yaw = _uniform(rng, ar['yaw'][0], ar['yaw'][1])
        pose.orientation = _quat_from_yaw(yaw)
        active_samples.append(
          (oid, pose.position.x, pose.position.y, pose.position.z, yaw)
        )
      else:
        # Park below workspace; slight X offset so spheres do not fully overlap.
        pose.position.x = 0.12 * float(i)
        pose.position.y = 0.0
        pose.position.z = park_z
        pose.orientation.w = 1.0

      co = CollisionObject()
      co.header.frame_id = frame_id
      co.header.stamp = stamp
      co.id = oid
      co.primitives.append(prim)
      co.primitive_poses.append(pose)
      co.operation = CollisionObject.ADD
      scene.world.collision_objects.append(co)

    ok = self.apply_diff(scene)
    if ok:
      active_list = ', '.join(sorted(active_set))
      self.get_logger().info(
        f'Sphere reset: seed={seed} | {k_active} active / {len(names)} slots — [{active_list}], '
        f'r={radius:.3f} m, inactive park z={park_z:.2f}'
      )
      for oid, px, py, pz, yaw in active_samples:
        self.get_logger().info(
          f'  active {oid}: xyz=({px:.4f},{py:.4f},{pz:.4f}) m, yaw={yaw:.4f} rad'
        )
    return ok

  def remove_objects(self, object_ids: list[str]) -> bool:
    scene = PlanningScene()
    scene.is_diff = True
    for oid in object_ids:
      co = CollisionObject()
      co.id = oid
      co.operation = CollisionObject.REMOVE
      scene.world.collision_objects.append(co)
    ok = self.apply_diff(scene)
    if ok:
      self.get_logger().info(f'Removed collision objects: {object_ids}')
    return ok

  def remove_object(self, object_id: str) -> bool:
    return self.remove_objects([object_id])

  def add_demo_box(
    self,
    object_id: str,
    frame_id: str,
    size_xyz: tuple[float, float, float],
    pos_xyz: tuple[float, float, float],
  ) -> bool:
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]

    pose = Pose()
    pose.orientation.w = 1.0
    pose.position.x = float(pos_xyz[0])
    pose.position.y = float(pos_xyz[1])
    pose.position.z = float(pos_xyz[2])

    co = CollisionObject()
    co.header.frame_id = frame_id
    co.header.stamp = self.get_clock().now().to_msg()
    co.id = object_id
    co.primitives.append(box)
    co.primitive_poses.append(pose)
    co.operation = CollisionObject.ADD

    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects.append(co)

    ok = self.apply_diff(scene)
    if ok:
      self.get_logger().info(
        f'Added collision object "{object_id}" ({size_xyz[0]:.3f}x{size_xyz[1]:.3f}x{size_xyz[2]:.3f} m) '
        f'at {frame_id} pos ({pos_xyz[0]:.3f},{pos_xyz[1]:.3f},{pos_xyz[2]:.3f})'
      )
    return ok


def main() -> int:
  rclpy.init()
  parser = argparse.ArgumentParser(
    description='Planning scene spheres (reset per OBSTACLE_SPHERE_RESET_PARAMS) or legacy box.'
  )
  parser.add_argument(
    '--legacy-box',
    action='store_true',
    help='Add single demo box instead of sphere reset',
  )
  parser.add_argument(
    '--remove',
    action='store_true',
    help='With --object-id: remove that id; otherwise ignored if --remove-all-spheres set',
  )
  parser.add_argument(
    '--remove-all-spheres',
    action='store_true',
    help='Remove obstacle_sphere_1/2/3 from planning scene',
  )
  parser.add_argument(
    '--object-id',
    default=LEGACY_OBJECT_ID,
    help='--remove: single id to remove; --legacy-box: box id to add',
  )
  parser.add_argument(
    '--frame-id',
    default=DEFAULT_FRAME_ID,
    help='TF frame for collision poses (default base_link)',
  )
  parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help='RNG seed for sphere placement and which spheres are active (1–2 of 3)',
  )
  parser.add_argument(
    '--sx',
    type=float,
    default=DEFAULT_BOX_SIZE_XYZ[0],
    help='[legacy-box] Box size X (m)',
  )
  parser.add_argument(
    '--sy',
    type=float,
    default=DEFAULT_BOX_SIZE_XYZ[1],
    help='[legacy-box] Box size Y (m)',
  )
  parser.add_argument(
    '--sz',
    type=float,
    default=DEFAULT_BOX_SIZE_XYZ[2],
    help='[legacy-box] Box size Z (m)',
  )
  parser.add_argument(
    '--px',
    type=float,
    default=DEFAULT_BOX_POSITION_XYZ[0],
    help='[legacy-box] Box center X (m)',
  )
  parser.add_argument(
    '--py',
    type=float,
    default=DEFAULT_BOX_POSITION_XYZ[1],
    help='[legacy-box] Box center Y (m)',
  )
  parser.add_argument(
    '--pz',
    type=float,
    default=DEFAULT_BOX_POSITION_XYZ[2],
    help='[legacy-box] Box center Z (m)',
  )
  args = parser.parse_args(remove_ros_args(sys.argv)[1:])

  node = PlanningSceneObstaclesNode()
  if node.has_parameter('use_sim_time') and node.get_parameter('use_sim_time').value:
    node.get_logger().info('use_sim_time is true (match moveit.launch.py use_sim:=true if sim).')

  try:
    if not node._wait_service():
      return 1

    if args.remove_all_spheres:
      ok = node.remove_objects(list(OBSTACLE_SPHERE_RESET_PARAMS['obstacle_names']))
    elif args.remove:
      ok = node.remove_object(args.object_id)
    elif args.legacy_box:
      ok = node.add_demo_box(
        args.object_id,
        args.frame_id,
        (args.sx, args.sy, args.sz),
        (args.px, args.py, args.pz),
      )
    else:
      ok = node.reset_obstacle_spheres(args.frame_id, seed=args.seed)

    return 0 if ok else 1
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  sys.exit(main())
