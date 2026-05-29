#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Publishes obstacle spheres as visualization_msgs/MarkerArray (pose + diameter in scale).
# pathplanning_node.py subscribes to the same topic, applies them to MoveIt, and plans.
#
# Docker (robotis_lab): edit THIS file under
#   ~/ros2_ws/src/ai_worker/ffw_moveit_config/scripts/obstacle_node.py
# then either:
#   colcon build --packages-select ffw_moveit_config && source install/setup.bash
#   ros2 run ffw_moveit_config obstacle_node.py ...
# or run source directly (no rebuild):
#   ./scripts/run_obstacle_node.sh --ros-args -p use_sim_time:=true
#
# IMPORTANT: `ros2 run` uses install/ffw_moveit_config/lib/... until you rebuild.
#
# Params:
#   obstacle_topic, publish_period_sec, frame_id, seed
#   obstacle_x_min/max, obstacle_y_min/max, obstacle_z_min/max — random layout range (m)
#   use_fixed_pose, fixed_x/y/z — single fixed obstacle position
#   use_custom_poses — if true, place active obstacles at custom_pose_x/y/z (same length)
#   sphere_radius, min_obstacles, max_obstacles
#
# Resample without restart:
#   ros2 service call /obstacle_node/reset_obstacle_layout std_srvs/srv/Trigger

import math
import random
import secrets
from pathlib import Path
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import ColorRGBA, Header
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

OBSTACLE_SPHERE_RESET_PARAMS = {
  'obstacle_names': ('obstacle_sphere_1', 'obstacle_sphere_2', 'obstacle_sphere_3'),
  'min_obstacles': 1,
  'max_obstacles': 1,
  'sphere_radius': 0.05,
  # Keep away from default arm pose (cuRobo start-state collision).
  'active_pose_range': {
    'x': (0.10, 0.12),
    'y': (0.12, 0.14),
    'z': (1.28, 1.48),
    'yaw': (-math.pi, math.pi),
  },
  'inactive_park_z': -2.5,
}

MARKER_NS = 'obstacle_sphere'


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


def _name_to_marker_id(name: str) -> int:
  if name.endswith('_3'):
    return 3
  if name.endswith('_2'):
    return 2
  return 1


def _obstacle_params_from_node(node) -> Dict[str, Any]:
  """Merge file defaults with ROS parameters (runtime overrides)."""
  p: Dict[str, Any] = {
    'obstacle_names': OBSTACLE_SPHERE_RESET_PARAMS['obstacle_names'],
    'min_obstacles': int(node.get_parameter('min_obstacles').value),
    'max_obstacles': int(node.get_parameter('max_obstacles').value),
    'sphere_radius': float(node.get_parameter('sphere_radius').value),
    'inactive_park_z': float(OBSTACLE_SPHERE_RESET_PARAMS['inactive_park_z']),
    'active_pose_range': {
      'x': (
        float(node.get_parameter('obstacle_x_min').value),
        float(node.get_parameter('obstacle_x_max').value),
      ),
      'y': (
        float(node.get_parameter('obstacle_y_min').value),
        float(node.get_parameter('obstacle_y_max').value),
      ),
      'z': (
        float(node.get_parameter('obstacle_z_min').value),
        float(node.get_parameter('obstacle_z_max').value),
      ),
      'yaw': (
        float(node.get_parameter('obstacle_yaw_min').value),
        float(node.get_parameter('obstacle_yaw_max').value),
      ),
    },
  }
  return p


def _marker_delete_all(stamp, frame_id: str) -> Marker:
  m = Marker()
  m.header = Header(stamp=stamp, frame_id=frame_id)
  m.action = Marker.DELETEALL
  return m


def _build_markers(
  rng: random.Random,
  frame_id: str,
  stamp,
  logger,
  params: Dict[str, Any] | None = None,
  *,
  use_fixed_pose: bool = False,
  fixed_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
  use_custom_poses: bool = False,
  custom_poses: list[tuple[float, float, float]] | None = None,
) -> MarkerArray:
  p = params if params is not None else OBSTACLE_SPHERE_RESET_PARAMS
  names = tuple(p['obstacle_names'])
  nmin = int(p['min_obstacles'])
  nmax = int(p['max_obstacles'])
  radius = float(p['sphere_radius'])
  park_z = float(p['inactive_park_z'])
  ar = p['active_pose_range']

  custom = list(custom_poses or [])
  if use_custom_poses and custom:
    k_active = min(len(custom), len(names))
    active_set = set(names[:k_active])
  else:
    k_active = min(rng.randint(nmin, nmax), len(names))
    active_set = set(rng.sample(list(names), k=k_active))

  diameter = 2.0 * radius
  arr = MarkerArray()
  arr.markers.append(_marker_delete_all(stamp, frame_id))

  fx, fy, fz = fixed_xyz
  custom_idx = 0
  for i, oid in enumerate(names):
    pose = Pose()
    if oid in active_set:
      if use_custom_poses and custom_idx < len(custom):
        cx, cy, cz = custom[custom_idx]
        pose.position.x = cx
        pose.position.y = cy
        pose.position.z = cz
        pose.orientation.w = 1.0
        custom_idx += 1
      elif use_fixed_pose:
        pose.position.x = fx
        pose.position.y = fy
        pose.position.z = fz
        pose.orientation.w = 1.0
      else:
        pose.position.x = _uniform(rng, ar['x'][0], ar['x'][1])
        pose.position.y = _uniform(rng, ar['y'][0], ar['y'][1])
        pose.position.z = _uniform(rng, ar['z'][0], ar['z'][1])
        yaw = _uniform(rng, ar['yaw'][0], ar['yaw'][1])
        pose.orientation = _quat_from_yaw(yaw)
    else:
      pose.position.x = 0.12 * float(i)
      pose.position.y = 0.0
      pose.position.z = park_z
      pose.orientation.w = 1.0

    m = Marker()
    m.header = Header()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.ns = MARKER_NS
    m.id = _name_to_marker_id(oid)
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose = pose
    m.scale.x = diameter
    m.scale.y = diameter
    m.scale.z = diameter
    m.color = ColorRGBA(r=1.0, g=0.4, b=0.1, a=0.85)
    m.lifetime.sec = 0
    m.lifetime.nanosec = 0
    arr.markers.append(m)

  logger.info(
    f'Obstacle layout fixed for this run | active={[x for x in names if x in active_set]}'
  )
  ar = p['active_pose_range']
  logger.info(
    f'  range x={ar["x"]} y={ar["y"]} z={ar["z"]} r={float(p["sphere_radius"]):.3f}m'
  )
  for m in arr.markers:
    if float(m.pose.position.z) > park_z + 0.5:
      logger.info(
        f'  -> id={m.id} xyz=({m.pose.position.x:.3f}, {m.pose.position.y:.3f}, '
        f'{m.pose.position.z:.3f})'
      )
  return arr


class ObstacleNode(Node):
  def __init__(self) -> None:
    super().__init__('obstacle_node')
    _d = OBSTACLE_SPHERE_RESET_PARAMS
    _ar = _d['active_pose_range']
    self.declare_parameter('obstacle_topic', '/obstacle_markers')
    self.declare_parameter('publish_period_sec', 1.0)
    self.declare_parameter('frame_id', 'base_link')
    self.declare_parameter('seed', -1)
    self.declare_parameter('obstacle_x_min', float(_ar['x'][0]))
    self.declare_parameter('obstacle_x_max', float(_ar['x'][1]))
    self.declare_parameter('obstacle_y_min', float(_ar['y'][0]))
    self.declare_parameter('obstacle_y_max', float(_ar['y'][1]))
    self.declare_parameter('obstacle_z_min', float(_ar['z'][0]))
    self.declare_parameter('obstacle_z_max', float(_ar['z'][1]))
    self.declare_parameter('obstacle_yaw_min', float(_ar['yaw'][0]))
    self.declare_parameter('obstacle_yaw_max', float(_ar['yaw'][1]))
    self.declare_parameter('sphere_radius', float(_d['sphere_radius']))
    self.declare_parameter('min_obstacles', int(_d['min_obstacles']))
    self.declare_parameter('max_obstacles', int(_d['max_obstacles']))
    self.declare_parameter('use_fixed_pose', False)
    self.declare_parameter('fixed_x', 0.65)
    self.declare_parameter('fixed_y', 0.0)
    self.declare_parameter('fixed_z', 1.35)
    self.declare_parameter('use_custom_poses', False)
    self.declare_parameter('custom_pose_x', [0.31, 0.55])
    self.declare_parameter('custom_pose_y', [0.13, -0.05])
    self.declare_parameter('custom_pose_z', [1.35, 1.40])

    self.get_logger().info(f'obstacle_node script: {Path(__file__).resolve()}')

    topic = str(self.get_parameter('obstacle_topic').value)
    period = float(self.get_parameter('publish_period_sec').value)
    self._frame_id = str(self.get_parameter('frame_id').value)
    self._rng = self._make_rng()

    qos = QoSProfile(
      depth=10,
      durability=DurabilityPolicy.VOLATILE,
      reliability=ReliabilityPolicy.RELIABLE,
    )
    self._pub = self.create_publisher(MarkerArray, topic, qos)
    self.get_logger().info(
      f'Publishing MarkerArray on {topic} (RELIABLE/VOLATILE); frame_id={self._frame_id}'
    )
    self.get_logger().info(
      f'RViz: add display for topic "{topic}" (MarkerArray), Fixed Frame e.g. {self._frame_id}.'
    )
    self.get_logger().info(
      'After param changes: ros2 service call /obstacle_node/reset_obstacle_layout std_srvs/srv/Trigger'
    )

    self._resample_and_publish(log_banner=True)
    if period > 0.0:
      self.create_timer(period, self._publish_cached)
      self.get_logger().info(
        f'Republishing the same layout every {period:.3f} s (restart node for new random poses).'
      )
    else:
      self.get_logger().info('publish_period_sec is 0 — single publish at startup.')

    self.create_service(Trigger, 'reset_obstacle_layout', self._on_reset_layout)

  def _make_rng(self) -> random.Random:
    raw_seed = self.get_parameter('seed').value
    try:
      seed_param = int(raw_seed)
    except (TypeError, ValueError):
      seed_param = -1
    if seed_param >= 0:
      self.get_logger().info(f'RNG seed={seed_param} (same layout on each reset with this seed).')
      return random.Random(seed_param)
    ephemeral = secrets.randbelow(2**63)
    self.get_logger().info(f'RNG one-shot seed={ephemeral}')
    return random.Random(ephemeral)

  def _custom_poses_from_params(self) -> list[tuple[float, float, float]]:
    xs = [float(v) for v in self.get_parameter('custom_pose_x').value]
    ys = [float(v) for v in self.get_parameter('custom_pose_y').value]
    zs = [float(v) for v in self.get_parameter('custom_pose_z').value]
    n = min(len(xs), len(ys), len(zs))
    if n != len(xs) or n != len(ys) or n != len(zs):
      self.get_logger().warn(
        f'custom_pose_x/y/z lengths differ ({len(xs)}/{len(ys)}/{len(zs)}); using first {n}.'
      )
    return list(zip(xs[:n], ys[:n], zs[:n]))

  def _resample_and_publish(self, log_banner: bool = False) -> None:
    stamp = self.get_clock().now().to_msg()
    layout_params = _obstacle_params_from_node(self)
    use_custom = bool(self.get_parameter('use_custom_poses').value)
    custom_poses = self._custom_poses_from_params() if use_custom else []
    use_fixed = bool(self.get_parameter('use_fixed_pose').value) and not use_custom
    fixed = (
      float(self.get_parameter('fixed_x').value),
      float(self.get_parameter('fixed_y').value),
      float(self.get_parameter('fixed_z').value),
    )
    if log_banner:
      ar = layout_params['active_pose_range']
      if use_custom:
        mode = f'CUSTOM poses={custom_poses}'
      elif use_fixed:
        mode = f'FIXED xyz={fixed}'
      else:
        mode = f'random x={ar["x"]} y={ar["y"]} z={ar["z"]}'
      self.get_logger().info(f'=== obstacle layout: {mode} ===')
    self._cached_markers = _build_markers(
      self._rng,
      self._frame_id,
      stamp,
      self.get_logger(),
      layout_params,
      use_fixed_pose=use_fixed,
      fixed_xyz=fixed,
      use_custom_poses=use_custom,
      custom_poses=custom_poses,
    )
    self._publish_cached()

  def _on_reset_layout(self, _request, response):
    """Resample from current ROS parameters (no node restart / rebuild)."""
    self._rng = self._make_rng()
    self._resample_and_publish(log_banner=True)
    response.success = True
    response.message = 'Obstacle layout resampled from current parameters.'
    return response

  def _publish_cached(self) -> None:
    stamp = self.get_clock().now().to_msg()
    for m in self._cached_markers.markers:
      m.header.stamp = stamp
    self._pub.publish(self._cached_markers)


def main() -> None:
  rclpy.init()
  node = ObstacleNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
