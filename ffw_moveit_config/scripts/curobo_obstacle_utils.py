#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Convert visualization_msgs/MarkerArray (obstacle_node) to cuRobo cuboids and MoveIt scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

DEFAULT_PLANNING_FRAME = 'base_link'
# obstacle_node parks inactive spheres below the workspace
INACTIVE_OBSTACLE_Z = -2.0
# perception_click_planning_node publishes pre-scaled CUBE markers in this namespace.
PERCEPTION_OBSTACLE_MARKER_NS = 'obstacle_cuboid'
# Virtual back wall from perception_click_planning_node (collision thickness in marker.scale.x).
TARGET_BACK_WALL_ID = 1000000
TARGET_BACK_WALL_MARKER_IDS = frozenset({TARGET_BACK_WALL_ID})
# cuRobo plane thickness = marker_thickness_x * this factor (1.0 = no extra inflate).
TARGET_BACK_WALL_CUROBO_X_INFLATE = 1.0
# Collision-only slab for cuRobo (RViz marker can stay thicker). Keeps near -X face fixed.
TARGET_BACK_WALL_CUROBO_COLLISION_X = 0.04


def target_back_wall_effective_thickness_x(
  marker_thickness_x: float,
  dim_scale: float = 1.0,
) -> float:
  """Final cuRobo cuboid X after client pre-scale + server dim_scale=1.0."""
  ds = max(float(dim_scale), 0.1)
  return float(marker_thickness_x) * (1.0 / ds) * TARGET_BACK_WALL_CUROBO_X_INFLATE


def target_back_wall_center_x(
  near_face_x: float,
  collision_thickness_x: float = TARGET_BACK_WALL_CUROBO_COLLISION_X,
) -> float:
  """Cuboid center so the -X face sits at near_face_x (target + offset)."""
  tx = max(float(collision_thickness_x), 0.02)
  return float(near_face_x) + tx * 0.5


def _is_back_wall_cuboid(cuboid: Dict[str, Any]) -> bool:
  return str(cuboid.get('name', '')).endswith(f'_{TARGET_BACK_WALL_ID}')


def _point_cuboid_clearance_m(
  point_xyz: Tuple[float, float, float],
  pose: Sequence[float],
  dims: Sequence[float],
) -> float:
  """Shortest distance from point to axis-aligned cuboid surface [m] (pose = center)."""
  px, py, pz = point_xyz
  cx, cy, cz = float(pose[0]), float(pose[1]), float(pose[2])
  hx = max(float(dims[0]) * 0.5, 1e-6)
  hy = max(float(dims[1]) * 0.5, 1e-6)
  hz = max(float(dims[2]) * 0.5, 1e-6)
  dx = max(abs(px - cx) - hx, 0.0)
  dy = max(abs(py - cy) - hy, 0.0)
  dz = max(abs(pz - cz) - hz, 0.0)
  return math.sqrt(dx * dx + dy * dy + dz * dz)


def thin_back_wall_cuboid_for_curobo(
  pose: Sequence[float],
  dims: Sequence[float],
  thin_x: float = TARGET_BACK_WALL_CUROBO_COLLISION_X,
) -> Tuple[List[float], List[float]]:
  """Thin wall along X for cuRobo; preserve near -X face (robot-side plane)."""
  cx = float(pose[0])
  half = max(float(dims[0]) * 0.5, 1e-6)
  near_face_x = cx - half
  tx = max(float(thin_x), 0.02)
  new_cx = near_face_x + tx * 0.5
  new_pose = [float(pose[0]), float(pose[1]), float(pose[2]), *pose[3:7]]
  new_pose[0] = new_cx
  return new_pose, [tx, float(dims[1]), float(dims[2])]


def filter_curobo_obstacles_for_goal(
  cuboids: List[Dict[str, Any]],
  goal_xyz: Tuple[float, float, float],
  clearance_m: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
  """Drop click cuboids hugging the goal so cuRobo can reach the clicked point.

  Perception obstacles are often placed on the same plant part as the target.
  cuRobo then stops at the nearest collision-free pose short of the goal.
  Back wall is always kept.
  """
  keep: List[Dict[str, Any]] = []
  dropped: List[str] = []
  clearance = max(float(clearance_m), 0.0)
  for cuboid in cuboids:
    if _is_back_wall_cuboid(cuboid):
      keep.append(cuboid)
      continue
    dist = _point_cuboid_clearance_m(goal_xyz, cuboid['pose'], cuboid['dims'])
    if dist < clearance:
      dropped.append(str(cuboid.get('name', '?')))
      continue
    keep.append(cuboid)
  return keep, dropped


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


def _cuboid_dim_scale(m: Marker, dim_scale: float) -> float:
  """Manual obstacles shrink once on client; back wall is already collision thickness."""
  if int(m.id) in TARGET_BACK_WALL_MARKER_IDS:
    return 1.0
  if m.ns == PERCEPTION_OBSTACLE_MARKER_NS:
    return 1.0
  return max(float(dim_scale), 0.1)


def markers_to_planning_cuboids(
  msg: MarkerArray, dim_scale: float = 1.0,
) -> List[Dict[str, Any]]:
  """Canonical obstacle list for MoveIt scene + cuRobo (same pose/dims everywhere)."""
  cuboids: List[Dict[str, Any]] = []
  for m in msg.markers:
    if m.action in (Marker.DELETE, Marker.DELETEALL):
      continue
    if not _marker_is_active(m):
      continue

    name = _collision_object_id(m)
    pose = _pose_to_curobo_pose(m)
    scale = _cuboid_dim_scale(m, dim_scale)

    if m.type == Marker.SPHERE:
      if m.scale.x <= 0.0:
        continue
      dims = _sphere_marker_cuboid_dims(m, scale)
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


# Back-compat alias
markers_to_curobo_cuboids = markers_to_planning_cuboids


def planning_scene_from_cuboids(
  cuboids: List[Dict[str, Any]],
  stamp,
  frame_id: str = DEFAULT_PLANNING_FRAME,
) -> Optional[PlanningScene]:
  """MoveIt planning scene from the same cuboids sent to cuRobo."""
  scene = PlanningScene()
  scene.is_diff = True
  for obs in cuboids:
    pose_vals = obs['pose']
    dims = obs['dims']
    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [float(dims[0]), float(dims[1]), float(dims[2])]
    pose = Pose()
    pose.position.x = float(pose_vals[0])
    pose.position.y = float(pose_vals[1])
    pose.position.z = float(pose_vals[2])
    pose.orientation.w = float(pose_vals[3])
    pose.orientation.x = float(pose_vals[4])
    pose.orientation.y = float(pose_vals[5])
    pose.orientation.z = float(pose_vals[6])
    co = CollisionObject()
    co.header.frame_id = frame_id
    co.header.stamp = stamp
    co.id = str(obs['name'])
    co.primitives.append(prim)
    co.primitive_poses.append(pose)
    co.operation = CollisionObject.ADD
    scene.world.collision_objects.append(co)
  if not scene.world.collision_objects:
    return None
  return scene


def markers_to_planning_scene(
  msg: MarkerArray,
  stamp,
  logger,
  *,
  sphere_as_box: bool = True,
  dim_scale: float = 1.0,
) -> Optional[PlanningScene]:
  """Build MoveIt scene from the same cuboid conversion cuRobo uses."""
  del logger, sphere_as_box  # kept for call-site compatibility
  cuboids = markers_to_planning_cuboids(msg, dim_scale=dim_scale)
  frame = DEFAULT_PLANNING_FRAME
  for m in msg.markers:
    if m.action in (Marker.DELETE, Marker.DELETEALL):
      continue
    if m.header.frame_id:
      frame = m.header.frame_id
      break
  return planning_scene_from_cuboids(cuboids, stamp, frame_id=frame)


def curobo_obstacle_debug_delete_all(
  stamp,
  frame_id: str = DEFAULT_PLANNING_FRAME,
  *,
  namespace: str = 'curobo_obstacle_cuboid',
) -> MarkerArray:
  """Remove stale /curobo_obstacle_cuboids markers from RViz."""
  arr = MarkerArray()
  delete_all = Marker()
  delete_all.header.stamp = stamp
  delete_all.header.frame_id = frame_id
  delete_all.ns = namespace
  delete_all.action = Marker.DELETEALL
  arr.markers.append(delete_all)
  return arr


def curobo_cuboids_to_marker_array(
  cuboids: List[Dict[str, Any]],
  stamp,
  frame_id: str = DEFAULT_PLANNING_FRAME,
  *,
  namespace: str = 'curobo_obstacle_cuboid',
) -> MarkerArray:
  """RViz: exact cuboid pose/dims used by MoveIt + cuRobo."""
  arr = MarkerArray()
  delete_all = Marker()
  delete_all.header.stamp = stamp
  delete_all.header.frame_id = frame_id
  delete_all.ns = namespace
  delete_all.action = Marker.DELETEALL
  arr.markers.append(delete_all)

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


def append_ee_axes_markers(
  arr: MarkerArray,
  stamp,
  frame_id: str,
  pose: Pose,
  *,
  ns: str,
  marker_id: int,
  axis_length: float = 0.12,
  line_width: float = 0.008,
) -> None:
  """RGB LINE_LIST at pose: +X red (EE forward), +Y green (up), +Z blue."""
  length = max(float(axis_length), 0.02)
  width = max(float(line_width), 0.002)
  axes = Marker()
  axes.header.stamp = stamp
  axes.header.frame_id = frame_id
  axes.ns = ns
  axes.id = marker_id
  axes.type = Marker.LINE_LIST
  axes.action = Marker.ADD
  axes.pose = pose
  axes.scale.x = width
  axes.points = [
    Point(x=0.0, y=0.0, z=0.0),
    Point(x=length, y=0.0, z=0.0),
    Point(x=0.0, y=0.0, z=0.0),
    Point(x=0.0, y=length, z=0.0),
    Point(x=0.0, y=0.0, z=0.0),
    Point(x=0.0, y=0.0, z=length),
  ]
  axes.colors = [
    ColorRGBA(r=1.0, g=0.15, b=0.15, a=0.95),
    ColorRGBA(r=1.0, g=0.15, b=0.15, a=0.95),
    ColorRGBA(r=0.15, g=0.9, b=0.2, a=0.95),
    ColorRGBA(r=0.15, g=0.9, b=0.2, a=0.95),
    ColorRGBA(r=0.2, g=0.45, b=1.0, a=0.95),
    ColorRGBA(r=0.2, g=0.45, b=1.0, a=0.95),
  ]
  arr.markers.append(axes)


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


def quat_multiply(qa: Quaternion, qb: Quaternion) -> Quaternion:
  """Hamilton product qa * qb (apply qb in qa's local frame when post-multiplying)."""
  w1, x1, y1, z1 = float(qa.w), float(qa.x), float(qa.y), float(qa.z)
  w2, x2, y2, z2 = float(qb.w), float(qb.x), float(qb.y), float(qb.z)
  q = Quaternion()
  q.w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  q.x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  q.y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
  q.z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
  return q


def quat_toward_robot_y_up_arm(
  position_xyz: Tuple[float, float, float],
  robot_base_xy: Tuple[float, float] = (0.0, 0.0),
  arm: str = 'l',
  right_roll_rad: float = math.pi,
) -> Quaternion:
  """Left: +X→base, +Y↑. Right: same then roll π about EE +X (gripper flipped)."""
  q = quat_toward_robot_y_up(position_xyz, robot_base_xy=robot_base_xy)
  if str(arm).lower() in ('r', 'right', 'arm_r') and abs(right_roll_rad) > 1e-9:
    q = quat_multiply(q, quat_from_rpy(right_roll_rad, 0.0, 0.0))
  return q


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


def quat_rotate_vector(
  quat: Quaternion,
  vx: float,
  vy: float,
  vz: float,
) -> Tuple[float, float, float]:
  """Rotate vector (vx, vy, vz) by unit quaternion quat (vector in base frame)."""
  x, y, z = float(quat.x), float(quat.y), float(quat.z)
  w = float(quat.w)
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  return (
    vx + w * tx + y * tz - z * ty,
    vy + w * ty + z * tx - x * tz,
    vz + w * tz + x * ty - y * tx,
  )


def tool_axis_unit_in_base(quat: Quaternion, axis: str = 'z') -> Tuple[float, float, float]:
  """Unit vector of tool +X/+Y/+Z expressed in planning frame."""
  key = str(axis).lower().strip()
  if key == 'x':
    local = (1.0, 0.0, 0.0)
  elif key == 'y':
    local = (0.0, 1.0, 0.0)
  else:
    local = (0.0, 0.0, 1.0)
  return quat_rotate_vector(quat, *local)


def offset_along_tool_axis(
  position_xyz: Tuple[float, float, float],
  quat: Quaternion,
  distance_m: float,
  axis: str = 'z',
) -> Tuple[float, float, float]:
  """Translate position by distance_m along +tool axis."""
  ax, ay, az = tool_axis_unit_in_base(quat, axis)
  d = float(distance_m)
  px, py, pz = position_xyz
  return (px + d * ax, py + d * ay, pz + d * az)


def approach_position_from_contact(
  contact_xyz: Tuple[float, float, float],
  quat: Quaternion,
  retreat_tool_m: float,
  axis: str = 'z',
) -> Tuple[float, float, float]:
  """Pre-grasp pose: offset from contact along +tool axis (final move is −axis to contact)."""
  retreat = max(float(retreat_tool_m), 0.0)
  if retreat <= 0.0:
    return contact_xyz
  ax, ay, az = tool_axis_unit_in_base(quat, axis)
  cx, cy, cz = contact_xyz
  return (cx + retreat * ax, cy + retreat * ay, cz + retreat * az)
