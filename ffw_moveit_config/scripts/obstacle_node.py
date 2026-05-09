#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Publishes obstacle spheres as visualization_msgs/MarkerArray (pose + diameter in scale).
# pathplanning_node.py subscribes to the same topic, applies them to MoveIt, and plans.
#
# Layout is sampled once when the node starts; the same poses are republished on a timer
# (for RViz / VOLATILE QoS). Stop and restart the node to draw a new random layout.
#
# Params:
#   obstacle_topic (str, default /obstacle_markers)
#   publish_period_sec (float, default 1.0) — republish the same markers this often;
#                          0.0 = single publish at startup only.
#   frame_id (str, default base_link)
#   seed (int, default -1) — if >= 0, same layout whenever you restart with that seed;
#                          if -1, new random layout on each process start (entropy once).
#
# RViz: Add → By topic → visualization_msgs/msg/MarkerArray → pick this topic (or add
#   "Markers" display and set topic). Fixed Frame must match frame_id (default base_link)
#   or markers will not appear in view.
#
# Example:
#   ros2 run ffw_moveit_config obstacle_node.py --ros-args -p use_sim_time:=true
#   ros2 run ffw_moveit_config obstacle_node.py --ros-args -p seed:=42

import math
import random
import secrets

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

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


def _build_markers(rng: random.Random, frame_id: str, stamp, logger) -> MarkerArray:
  p = OBSTACLE_SPHERE_RESET_PARAMS
  names = tuple(p['obstacle_names'])
  nmin = int(p['min_obstacles'])
  nmax = int(p['max_obstacles'])
  radius = float(p['sphere_radius'])
  park_z = float(p['inactive_park_z'])
  ar = p['active_pose_range']

  k_active = min(rng.randint(nmin, nmax), len(names))
  active_set = set(rng.sample(list(names), k=k_active))

  diameter = 2.0 * radius
  arr = MarkerArray()

  for i, oid in enumerate(names):
    pose = Pose()
    if oid in active_set:
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
  return arr


class ObstacleNode(Node):
  def __init__(self) -> None:
    super().__init__('obstacle_node')
    self.declare_parameter('obstacle_topic', '/obstacle_markers')
    self.declare_parameter('publish_period_sec', 1.0)
    self.declare_parameter('frame_id', 'base_link')
    self.declare_parameter('seed', -1)

    topic = str(self.get_parameter('obstacle_topic').value)
    period = float(self.get_parameter('publish_period_sec').value)
    self._frame_id = str(self.get_parameter('frame_id').value)

    raw_seed = self.get_parameter('seed').value
    try:
      seed_param = int(raw_seed)
    except (TypeError, ValueError):
      seed_param = -1

    if seed_param >= 0:
      rng = random.Random(seed_param)
      self.get_logger().info(f'RNG seed={seed_param} (same layout on each restart with this seed).')
    else:
      ephemeral = secrets.randbelow(2**63)
      rng = random.Random(ephemeral)
      self.get_logger().info(
        f'RNG one-shot seed={ephemeral} (new random layout each time you start the node).'
      )

    stamp = self.get_clock().now().to_msg()
    self._cached_markers = _build_markers(rng, self._frame_id, stamp, self.get_logger())

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

    self._publish_cached()
    if period > 0.0:
      self.create_timer(period, self._publish_cached)
      self.get_logger().info(
        f'Republishing the same layout every {period:.3f} s (restart node for new random poses).'
      )
    else:
      self.get_logger().info('publish_period_sec is 0 — single publish at startup.')

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
