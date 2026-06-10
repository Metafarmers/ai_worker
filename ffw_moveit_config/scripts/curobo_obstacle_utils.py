#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Convert visualization_msgs/MarkerArray (obstacle_node) to cuRobo cuboids and MoveIt scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from geometry_msgs.msg import Quaternion
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray

DEFAULT_PLANNING_FRAME = 'base_link'
# obstacle_node parks inactive spheres below the workspace
INACTIVE_OBSTACLE_Z = -2.0


def _collision_object_id(m: Marker) -> str:
  if m.ns:
    return f'{m.ns}_{m.id}'
  return f'obstacle_marker_{m.id}'


def _marker_is_active(m: Marker) -> bool:
  return float(m.pose.position.z) > INACTIVE_OBSTACLE_Z


def _pose_to_curobo_pose(m: Marker) -> List[float]:
  p = m.pose.position
  q = m.pose.orientation
  return [
    float(p.x),
    float(p.y),
    float(p.z),
    float(q.w),
    float(q.x),
    float(q.y),
    float(q.z),
  ]


def _sphere_marker_radius(m: Marker) -> float:
  """RViz SPHERE marker scale is diameter (REP-103); MoveIt/cuRobo use radius."""
  return float(m.scale.x) * 0.5


def _sphere_marker_cuboid_dims(m: Marker, dim_scale: float = 1.0) -> List[float]:
  """Tight axis-aligned cube around the marker sphere (matches cuRobo cuboid obstacle)."""
  scale = max(float(dim_scale), 0.1)
  edge = 2.0 * _sphere_marker_radius(m) * scale
  return [edge, edge, edge]


def markers_to_curobo_cuboids(
  msg: MarkerArray, dim_scale: float = 1.0,
) -> List[Dict[str, Any]]:
  """Sphere/CUBE markers -> cuRobo plan server cuboid dicts."""
  scale = max(float(dim_scale), 0.1)
  cuboids: List[Dict[str, Any]] = []
  for m in msg.markers:
    if m.action in (Marker.DELETE, Marker.DELETEALL):
      continue
    if not _marker_is_active(m):
      continue

    name = _collision_object_id(m)
    pose = _pose_to_curobo_pose(m)

    if m.type == Marker.SPHERE:
      if m.scale.x <= 0.0:
        continue
      dims = _sphere_marker_cuboid_dims(m, dim_scale)
    elif m.type == Marker.CUBE:
      dims = [
        float(max(m.scale.x, 1e-6)) * scale,
        float(max(m.scale.y, 1e-6)) * scale,
        float(max(m.scale.z, 1e-6)) * scale,
      ]
    else:
      continue

    cuboids.append({'name': name, 'pose': pose, 'dims': dims})

  return cuboids


def markers_to_planning_scene(
  msg: MarkerArray,
  stamp,
  logger,
  *,
  sphere_as_box: bool = True,
) -> Optional[PlanningScene]:
  """Map markers to MoveIt collision objects.

  When sphere_as_box is true (default), SPHERE markers become BOX primitives with the
  same edge length cuRobo uses. RViz obstacle markers still draw spheres, but the
  MoveIt planning scene matches cuRobo cuboids instead of smaller sphere geometry.
  """
  scene = PlanningScene()
  scene.is_diff = True
  count = 0
  for m in msg.markers:
    if m.action in (Marker.DELETE, Marker.DELETEALL):
      continue
    if not _marker_is_active(m):
      continue

    if m.type == Marker.SPHERE:
      if m.scale.x <= 0.0:
        logger.warning(f'Skip sphere {_collision_object_id(m)}: non-positive scale.x')
        continue
      prim = SolidPrimitive()
      if sphere_as_box:
        edge = 2.0 * _sphere_marker_radius(m)
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [edge, edge, edge]
      else:
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [_sphere_marker_radius(m)]
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
    return None
  return scene


def curobo_cuboids_to_marker_array(
  cuboids: List[Dict[str, Any]],
  stamp,
  frame_id: str = DEFAULT_PLANNING_FRAME,
  *,
  namespace: str = 'curobo_obstacle_cuboid',
) -> MarkerArray:
  """Debug viz: exact cuboid pose/dims sent to cuRobo (wireframe cubes in RViz)."""
  arr = MarkerArray()
  for i, obs in enumerate(cuboids):
    pose_vals = obs['pose']
    dims = obs['dims']
    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.ns = namespace
    m.id = i + 1
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = float(pose_vals[0])
    m.pose.position.y = float(pose_vals[1])
    m.pose.position.z = float(pose_vals[2])
    m.pose.orientation.w = float(pose_vals[3])
    m.pose.orientation.x = float(pose_vals[4])
    m.pose.orientation.y = float(pose_vals[5])
    m.pose.orientation.z = float(pose_vals[6])
    m.scale.x = float(dims[0])
    m.scale.y = float(dims[1])
    m.scale.z = float(dims[2])
    m.color.r = 0.1
    m.color.g = 0.9
    m.color.b = 0.2
    m.color.a = 0.35
    arr.markers.append(m)
  return arr


def marker_array_signature(msg: MarkerArray) -> Tuple:
  """Stable tuple so identical obstacle republishes can be skipped."""
  rows = []
  for m in msg.markers:
    if not _marker_is_active(m):
      continue
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


def _normalize3(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
  n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
  if n < 1e-12:
    raise ValueError('zero-length axis vector')
  return (v[0] / n, v[1] / n, v[2] / n)


def _cross3(
  a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
  return (
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  )


def _dot3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def quat_from_xy_axes(
  x_axis: Tuple[float, float, float],
  y_axis: Tuple[float, float, float],
) -> Quaternion:
  """Right-handed EE frame in planning frame: +X along x_axis, +Y along y_axis, +Z = X x Y."""
  y = _normalize3(y_axis)
  x_proj = (
    x_axis[0] - _dot3(x_axis, y) * y[0],
    x_axis[1] - _dot3(x_axis, y) * y[1],
    x_axis[2] - _dot3(x_axis, y) * y[2],
  )
  x = _normalize3(x_proj)
  z = _normalize3(_cross3(x, y))
  x = _normalize3(_cross3(y, z))

  r00, r01, r02 = x[0], y[0], z[0]
  r10, r11, r12 = x[1], y[1], z[1]
  r20, r21, r22 = x[2], y[2], z[2]
  trace = r00 + r11 + r22
  q = Quaternion()
  if trace > 0.0:
    s = math.sqrt(trace + 1.0) * 2.0
    q.w = 0.25 * s
    q.x = (r21 - r12) / s
    q.y = (r02 - r20) / s
    q.z = (r10 - r01) / s
  elif r00 > r11 and r00 > r22:
    s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
    q.w = (r21 - r12) / s
    q.x = 0.25 * s
    q.y = (r01 + r10) / s
    q.z = (r02 + r20) / s
  elif r11 > r22:
    s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
    q.w = (r02 - r20) / s
    q.x = (r01 + r10) / s
    q.y = 0.25 * s
    q.z = (r12 + r21) / s
  else:
    s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
    q.w = (r10 - r01) / s
    q.x = (r02 + r20) / s
    q.y = (r12 + r21) / s
    q.z = 0.25 * s
  return q


def quat_robot_forward_y_up(
  forward: Tuple[float, float, float] = (1.0, 0.0, 0.0),
  up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Quaternion:
  """EE +X = robot forward, EE +Y = ground normal (up) in base_link."""
  return quat_from_xy_axes(forward, up)


def quat_toward_robot_y_up(
  position_xyz: Tuple[float, float, float],
  robot_base_xy: Tuple[float, float] = (0.0, 0.0),
  up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
  default_forward_xy: Tuple[float, float] = (1.0, 0.0),
) -> Quaternion:
  """EE +X horizontal toward robot base; EE +Y along ground normal (base_link +Z)."""
  px, py, _ = position_xyz
  bx, by = robot_base_xy
  dx = bx - px
  dy = by - py
  horiz_len = math.sqrt(dx * dx + dy * dy)
  if horiz_len < 1e-6:
    dx, dy = default_forward_xy
    horiz_len = math.sqrt(dx * dx + dy * dy)
    if horiz_len < 1e-6:
      dx, dy = 1.0, 0.0
      horiz_len = 1.0
  x_axis = (dx / horiz_len, dy / horiz_len, 0.0)
  return quat_from_xy_axes(x_axis, up)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
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
