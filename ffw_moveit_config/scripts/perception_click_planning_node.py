#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Click on ZED image → planning frame (base_link):
#   mode 1: target  → PoseStamped on /target_pose (EE +X→robot, +Y↑, same as curobo_pathplanning_node)
#   mode 2: obstacle → CUBE MarkerArray on /obstacle_markers (cuRobo cuboid)
#   mode 3: delete nearest click
#
# Run (with curobo_pathplanning_node + ZED):
#   ./scripts/run_perception_click_planning.sh
#
# Keys: 1=target, 2=obstacle, 3=delete, c=clear all, q=quit

from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Header
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped with tf2
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from curobo_obstacle_utils import quat_toward_robot_y_up
from perception_3d_utils import (
  depth_image_to_meters,
  depth_uv_from_color_uv,
  project_camera_xyz_to_uv,
  sample_depth_at_uv,
)

ClickMode = Literal['target', 'obstacle', 'delete']
MODE_COLORS = {
  'target': (0, 255, 0),
  'obstacle': (0, 0, 255),
  'delete': (0, 165, 255),
}

OBSTACLE_MARKER_NS = 'obstacle_cuboid'
TARGET_MARKER_NS = 'click_target'
TARGET_MARKER_ID_SPHERE = 1
TARGET_MARKER_ID_ARROW = 2


def _quat_from_yaw(yaw: float):
  from geometry_msgs.msg import Quaternion
  half = yaw * 0.5
  q = Quaternion()
  q.x = 0.0
  q.y = 0.0
  q.z = math.sin(half)
  q.w = math.cos(half)
  return q


def _identity_quat():
  from geometry_msgs.msg import Quaternion
  q = Quaternion()
  q.x = 0.0
  q.y = 0.0
  q.z = 0.0
  q.w = 1.0
  return q


@dataclass
class _PendingClick:
  u: int
  v: int


@dataclass
class PlanningClickPoint:
  kind: Literal['target', 'obstacle']
  point_id: int
  u: int
  v: int
  x: float
  y: float
  z: float


class PerceptionClickPlanningNode(Node):
  def __init__(self) -> None:
    super().__init__('perception_click_planning_node')

    self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
    self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
    self.declare_parameter('depth_camera_info_topic', '/zed/zed_node/rgb/camera_info')
    self.declare_parameter('planning_frame', 'base_link')
    self.declare_parameter('obstacle_topic', '/obstacle_markers')
    self.declare_parameter('target_pose_topic', '/target_pose')
    self.declare_parameter('publish_target_markers', True)
    self.declare_parameter('target_marker_topic', '/curobo_target_markers')
    self.declare_parameter('min_depth_m', 0.15)
    self.declare_parameter('max_depth_m', 10.0)
    self.declare_parameter('depth_patch_radius', 3)
    self.declare_parameter('depth_sample_mode', 'closest')
    self.declare_parameter('depth_scale', 1.0)
    self.declare_parameter('depth_offset_m', 0.0)
    self.declare_parameter('depth_is_radial', False)
    self.declare_parameter('show_reprojection', True)
    self.declare_parameter('delete_radius_px', 40)
    self.declare_parameter('publish_period_sec', 1.0)
    self.declare_parameter('display_hz', 10.0)
    self.declare_parameter('show_window', True)
    self.declare_parameter('window_name', 'perception_click_planning')
    self.declare_parameter('tf_lookup_timeout_sec', 0.5)
    self.declare_parameter('robot_base_x', 0.0)
    self.declare_parameter('robot_base_y', 0.0)
    self.declare_parameter('cuboid_size_x', 0.02)
    self.declare_parameter('cuboid_size_y', 0.02)
    self.declare_parameter('cuboid_size_z', 0.02)
    self.declare_parameter('obstacle_yaw', 0.0)
    self.declare_parameter('target_arrow_length', 0.12)
    self.declare_parameter('target_sphere_diameter', 0.06)
    self.declare_parameter('tf_parent_frame', '')

    self._planning_frame = str(self.get_parameter('planning_frame').value)
    self._min_depth = float(self.get_parameter('min_depth_m').value)
    self._max_depth = float(self.get_parameter('max_depth_m').value)
    self._patch_radius = int(self.get_parameter('depth_patch_radius').value)
    self._depth_sample_mode = str(self.get_parameter('depth_sample_mode').value)
    self._depth_scale = float(self.get_parameter('depth_scale').value)
    self._depth_offset_m = float(self.get_parameter('depth_offset_m').value)
    self._depth_is_radial = bool(self.get_parameter('depth_is_radial').value)
    self._show_reprojection = bool(self.get_parameter('show_reprojection').value)
    self._delete_radius_px = int(self.get_parameter('delete_radius_px').value)
    self._show_window = bool(self.get_parameter('show_window').value)
    self._window_name = str(self.get_parameter('window_name').value)
    self._tf_parent_override = str(self.get_parameter('tf_parent_frame').value).strip()
    self._tf_timeout = float(self.get_parameter('tf_lookup_timeout_sec').value)
    self._robot_base_x = float(self.get_parameter('robot_base_x').value)
    self._robot_base_y = float(self.get_parameter('robot_base_y').value)
    self._cuboid_size = (
      float(self.get_parameter('cuboid_size_x').value),
      float(self.get_parameter('cuboid_size_y').value),
      float(self.get_parameter('cuboid_size_z').value),
    )
    self._obstacle_yaw = float(self.get_parameter('obstacle_yaw').value)
    self._publish_target_markers = bool(self.get_parameter('publish_target_markers').value)

    self._bridge = CvBridge()
    self._data_lock = threading.Lock()
    self._state_lock = threading.Lock()
    self._latest_bgr: Optional[np.ndarray] = None
    self._latest_header = None
    self._latest_depth_m: Optional[np.ndarray] = None
    self._latest_depth_header = None
    self._latest_camera_info: Optional[CameraInfo] = None
    self._warned_missing_depth = False
    self._warned_tf = False
    self._logged_geometry = False

    self._mode: ClickMode = 'target'
    self._pending_click: Optional[_PendingClick] = None
    self._obstacles: List[PlanningClickPoint] = []
    self._target: Optional[PlanningClickPoint] = None
    self._next_obstacle_id = 0

    qos = QoSProfile(
      depth=10,
      durability=DurabilityPolicy.VOLATILE,
      reliability=ReliabilityPolicy.RELIABLE,
    )
    reliable_qos = QoSProfile(
      reliability=ReliabilityPolicy.RELIABLE,
      history=HistoryPolicy.KEEP_LAST,
      depth=10,
      durability=DurabilityPolicy.VOLATILE,
    )
    sensor_qos = QoSProfile(
      reliability=ReliabilityPolicy.BEST_EFFORT,
      history=HistoryPolicy.KEEP_LAST,
      depth=1,
      durability=DurabilityPolicy.VOLATILE,
    )
    camera_info_qos = QoSProfile(
      reliability=ReliabilityPolicy.RELIABLE,
      history=HistoryPolicy.KEEP_LAST,
      depth=10,
      durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    image_topic = str(self.get_parameter('image_topic').value)
    depth_topic = str(self.get_parameter('depth_topic').value)
    depth_info_topic = str(self.get_parameter('depth_camera_info_topic').value)
    obstacle_topic = str(self.get_parameter('obstacle_topic').value)
    target_topic = str(self.get_parameter('target_pose_topic').value)

    self.create_subscription(Image, image_topic, self._on_image, sensor_qos)
    self.create_subscription(Image, image_topic, self._on_image, reliable_qos)
    self.create_subscription(Image, depth_topic, self._on_depth, reliable_qos)
    self.create_subscription(CameraInfo, depth_info_topic, self._on_camera_info, camera_info_qos)
    self.create_subscription(CameraInfo, depth_info_topic, self._on_camera_info, reliable_qos)

    self._obstacle_pub = self.create_publisher(MarkerArray, obstacle_topic, qos)
    self._target_pose_pub = self.create_publisher(PoseStamped, target_topic, 10)
    if self._publish_target_markers:
      target_marker_topic = str(self.get_parameter('target_marker_topic').value)
      self._target_marker_pub = self.create_publisher(MarkerArray, target_marker_topic, qos)
    else:
      self._target_marker_pub = None

    self._tf_buffer = Buffer()
    self._tf_listener = TransformListener(self._tf_buffer, self)

    display_hz = max(float(self.get_parameter('display_hz').value), 1.0)
    self.create_timer(1.0 / display_hz, self._on_timer)

    period = float(self.get_parameter('publish_period_sec').value)
    if period > 0.0:
      self.create_timer(period, self._republish_planning_outputs)

    if self._show_window:
      cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
      cv2.setMouseCallback(self._window_name, self._on_mouse)

    self.get_logger().info(f'Script: {Path(__file__).resolve()}')
    self.get_logger().info(f'Camera: {image_topic}')
    self.get_logger().info(f'Planning frame: {self._planning_frame}')
    self.get_logger().info(f'Obstacle CUBE → {obstacle_topic}')
    self.get_logger().info(f'Target PoseStamped → {target_topic}')
    self.get_logger().info('Keys: 1=target, 2=obstacle, 3=delete, c=clear, q=quit')

  def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
      return
    with self._state_lock:
      self._pending_click = _PendingClick(u=x, v=y)

  def _set_mode(self, mode: ClickMode) -> None:
    with self._state_lock:
      self._mode = mode
    self.get_logger().info(f'Mode: {mode}')

  def _clear_all(self) -> None:
    with self._state_lock:
      self._obstacles.clear()
      self._target = None
      self._next_obstacle_id = 0
    self._publish_obstacles()
    self.get_logger().info('Cleared all targets and obstacles')

  def _on_image(self, msg: Image) -> None:
    try:
      bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as exc:
      self.get_logger().warn(f'Color cv_bridge failed: {exc}', throttle_duration_sec=5.0)
      return
    with self._data_lock:
      self._latest_bgr = bgr
      self._latest_header = msg.header

  def _on_depth(self, msg: Image) -> None:
    depth_m = depth_image_to_meters(msg, self._bridge)
    if depth_m is None:
      return
    with self._data_lock:
      self._latest_depth_m = depth_m
      self._latest_depth_header = msg.header

  def _on_camera_info(self, msg: CameraInfo) -> None:
    with self._data_lock:
      self._latest_camera_info = msg

  def _camera_frame(self, camera_info: CameraInfo, depth_header, color_header) -> str:
    if self._tf_parent_override:
      return self._tf_parent_override
    # Depth was measured in this frame — use for TF (must match deprojection frame).
    if depth_header is not None and depth_header.frame_id:
      return depth_header.frame_id
    if camera_info.header.frame_id:
      return camera_info.header.frame_id
    if color_header is not None and color_header.frame_id:
      return color_header.frame_id
    return 'camera_optical_frame'

  def _transform_camera_xyz_to_planning(
    self,
    x: float,
    y: float,
    z: float,
    camera_frame: str,
    stamp=None,
  ) -> Optional[Tuple[float, float, float]]:
    pt = PointStamped()
    pt.header.frame_id = camera_frame
    pt.point.x = x
    pt.point.y = y
    pt.point.z = z

    stamps_to_try = []
    if stamp is not None:
      stamps_to_try.append(stamp)
    stamps_to_try.append(rclpy.time.Time().to_msg())

    last_exc: Optional[TransformException] = None
    for st in stamps_to_try:
      pt.header.stamp = st
      try:
        out = self._tf_buffer.transform(
          pt,
          self._planning_frame,
          timeout=rclpy.duration.Duration(seconds=self._tf_timeout),
        )
        return float(out.point.x), float(out.point.y), float(out.point.z)
      except TransformException as exc:
        last_exc = exc

    if not self._warned_tf:
      self._warned_tf = True
      self.get_logger().error(
        f'TF {camera_frame} → {self._planning_frame} failed: {last_exc}. '
        'Check robot_state_publisher / ZED frames.'
      )
    return None

  def _goal_quat_for_position(self, xyz: Tuple[float, float, float]):
    return quat_toward_robot_y_up(
      xyz,
      robot_base_xy=(self._robot_base_x, self._robot_base_y),
    )

  def _make_target_pose_msg(self, pt: PlanningClickPoint, stamp) -> PoseStamped:
    # Position-only goal: orientation is ignored by the cuRobo server.
    # We still publish a valid quaternion to satisfy message type requirements.
    q = _identity_quat()
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = self._planning_frame
    msg.pose.position.x = pt.x
    msg.pose.position.y = pt.y
    msg.pose.position.z = pt.z
    msg.pose.orientation.x = float(q.x)
    msg.pose.orientation.y = float(q.y)
    msg.pose.orientation.z = float(q.z)
    msg.pose.orientation.w = float(q.w)
    return msg

  def _make_target_markers(self, pt: PlanningClickPoint, stamp) -> MarkerArray:
    pose_msg = self._make_target_pose_msg(pt, stamp)
    # Marker arrow keeps pointing toward the target for human readability.
    q = self._goal_quat_for_position((pt.x, pt.y, pt.z))
    pose_msg.pose.orientation.x = float(q.x)
    pose_msg.pose.orientation.y = float(q.y)
    pose_msg.pose.orientation.z = float(q.z)
    pose_msg.pose.orientation.w = float(q.w)
    arrow_len = float(self.get_parameter('target_arrow_length').value)
    sphere_d = float(self.get_parameter('target_sphere_diameter').value)

    arrow = Marker()
    arrow.header = Header(stamp=stamp, frame_id=self._planning_frame)
    arrow.ns = TARGET_MARKER_NS
    arrow.id = TARGET_MARKER_ID_ARROW
    arrow.type = Marker.ARROW
    arrow.action = Marker.ADD
    arrow.pose = pose_msg.pose
    arrow.scale.x = max(arrow_len, 0.02)
    arrow.scale.y = 0.025
    arrow.scale.z = 0.025
    arrow.color = ColorRGBA(r=0.1, g=0.85, b=0.25, a=0.95)

    sphere = Marker()
    sphere.header = arrow.header
    sphere.ns = TARGET_MARKER_NS
    sphere.id = TARGET_MARKER_ID_SPHERE
    sphere.type = Marker.SPHERE
    sphere.action = Marker.ADD
    sphere.pose = pose_msg.pose
    sphere.scale.x = sphere_d
    sphere.scale.y = sphere_d
    sphere.scale.z = sphere_d
    sphere.color = ColorRGBA(r=0.1, g=0.55, b=1.0, a=0.75)

    arr = MarkerArray()
    arr.markers.append(arrow)
    arr.markers.append(sphere)
    return arr

  def _build_obstacle_markers(self, stamp) -> MarkerArray:
    with self._state_lock:
      obstacles = list(self._obstacles)

    arr = MarkerArray()
    delete_all = Marker()
    delete_all.action = Marker.DELETEALL
    arr.markers.append(delete_all)

    sx, sy, sz = self._cuboid_size
    orient = _quat_from_yaw(self._obstacle_yaw)
    for pt in obstacles:
      m = Marker()
      m.header = Header(stamp=stamp, frame_id=self._planning_frame)
      m.ns = OBSTACLE_MARKER_NS
      m.id = pt.point_id + 1
      m.type = Marker.CUBE
      m.action = Marker.ADD
      m.pose.position.x = pt.x
      m.pose.position.y = pt.y
      m.pose.position.z = pt.z
      m.pose.orientation = orient
      m.scale.x = sx
      m.scale.y = sy
      m.scale.z = sz
      m.color = ColorRGBA(r=1.0, g=0.4, b=0.1, a=0.85)
      arr.markers.append(m)
    return arr

  def _publish_obstacles(self) -> None:
    stamp = self.get_clock().now().to_msg()
    self._obstacle_pub.publish(self._build_obstacle_markers(stamp))

  def _publish_target(self) -> None:
    with self._state_lock:
      target = self._target
    if target is None:
      return
    stamp = self.get_clock().now().to_msg()
    pose_msg = self._make_target_pose_msg(target, stamp)
    self._target_pose_pub.publish(pose_msg)
    if self._target_marker_pub is not None:
      self._target_marker_pub.publish(self._make_target_markers(target, stamp))

  def _republish_planning_outputs(self) -> None:
    self._publish_obstacles()
    self._publish_target()

  def _delete_nearest(self, u: int, v: int) -> None:
    with self._state_lock:
      candidates: List[Tuple[float, PlanningClickPoint, str]] = []
      if self._target is not None:
        d = math.hypot(self._target.u - u, self._target.v - v)
        candidates.append((d, self._target, 'target'))
      for obs in self._obstacles:
        d = math.hypot(obs.u - u, obs.v - v)
        candidates.append((d, obs, 'obstacle'))

      if not candidates:
        self.get_logger().info('Delete: nothing to remove')
        return

      dist, pt, kind = min(candidates, key=lambda item: item[0])
      if dist > self._delete_radius_px:
        self.get_logger().info(
          f'Delete: nearest {kind} is {dist:.0f}px away (limit {self._delete_radius_px}px)'
        )
        return

      if kind == 'target':
        self._target = None
        self.get_logger().info('Deleted target')
      else:
        self._obstacles = [o for o in self._obstacles if o.point_id != pt.point_id]
        self.get_logger().info(f'Deleted obstacle_{pt.point_id}')

    self._publish_obstacles()
    self._publish_target()

  def _log_geometry_once(
    self,
    color_h: int,
    color_w: int,
    depth_m: np.ndarray,
    camera_info: CameraInfo,
    camera_frame: str,
  ) -> None:
    if self._logged_geometry:
      return
    self._logged_geometry = True
    depth_h, depth_w = depth_m.shape[:2]
    info_w = int(camera_info.width)
    info_h = int(camera_info.height)
    self.get_logger().info(
      f'Geometry: color={color_w}x{color_h} depth={depth_w}x{depth_h} '
      f'camera_info={info_w}x{info_h} frame={camera_frame!r} → {self._planning_frame}'
    )
    if info_w > 0 and info_h > 0 and (info_w != color_w or info_h != color_h):
      self.get_logger().warn(
        f'CameraInfo size ({info_w}x{info_h}) != image ({color_w}x{color_h}); '
        'intrinsics scaled to match image.'
      )
    if depth_w != color_w or depth_h != color_h:
      self.get_logger().info(
        f'Depth/RGB resolution differs; depth sampled at scaled UV, deproject uses RGB UV.'
      )

  def _process_pending_click(
    self,
    click: _PendingClick,
    mode: ClickMode,
    depth_m: np.ndarray,
    camera_info: CameraInfo,
    color_h: int,
    color_w: int,
    camera_frame: str,
    stamp,
  ) -> None:
    if mode == 'delete':
      self._delete_nearest(click.u, click.v)
      return

    self._log_geometry_once(color_h, color_w, depth_m, camera_info, camera_frame)

    xyz_cam = sample_depth_at_uv(
      click.u, click.v, depth_m, color_h, color_w, camera_info,
      self._min_depth, self._max_depth, self._patch_radius,
      self._depth_sample_mode, self._depth_scale, self._depth_offset_m, self._depth_is_radial,
    )
    if xyz_cam is None:
      self.get_logger().warn(f'No valid depth at ({click.u}, {click.v})')
      return

    depth_h, depth_w = depth_m.shape[:2]
    du, dv = depth_uv_from_color_uv(
      float(click.u), float(click.v), color_w, color_h, depth_w, depth_h,
    )
    du_i, dv_i = int(round(du)), int(round(dv))
    center_d = float(depth_m[dv_i, du_i]) if 0 <= du_i < depth_w and 0 <= dv_i < depth_h else float('nan')
    r = max(self._patch_radius, 1)
    patch = depth_m[
      max(dv_i - r, 0):min(dv_i + r + 1, depth_h),
      max(du_i - r, 0):min(du_i + r + 1, depth_w),
    ]
    valid = patch[np.isfinite(patch) & (patch > self._min_depth) & (patch < self._max_depth)]

    cx, cy, cz = xyz_cam
    xyz_plan = self._transform_camera_xyz_to_planning(cx, cy, cz, camera_frame, stamp)
    if xyz_plan is None:
      return
    x, y, z = xyz_plan

    self.get_logger().info(
      f'Click ({click.u},{click.v}): depth_used={cz:.3f}m '
      f'(center={center_d:.3f} closest={float(np.min(valid)):.3f} '
      f'median={float(np.median(valid)):.3f}) '
      f'cam=({cx:.3f},{cy:.3f},{cz:.3f}) {camera_frame} → '
      f'{self._planning_frame}=({x:.3f},{y:.3f},{z:.3f})'
    )

    if mode == 'target':
      pt = PlanningClickPoint(
        kind='target', point_id=0, u=click.u, v=click.v, x=x, y=y, z=z,
      )
      with self._state_lock:
        self._target = pt
      self.get_logger().info(
        f'Target {self._planning_frame}=({x:.3f}, {y:.3f}, {z:.3f}) '
        f'(EE +X→robot, +Y↑)'
      )
      self._publish_target()
      return

    with self._state_lock:
      point_id = self._next_obstacle_id
      self._next_obstacle_id += 1
    pt = PlanningClickPoint(
      kind='obstacle', point_id=point_id, u=click.u, v=click.v, x=x, y=y, z=z,
    )
    with self._state_lock:
      self._obstacles.append(pt)
    self.get_logger().info(
      f'Obstacle_{point_id} cuboid {self._cuboid_size} m at '
      f'{self._planning_frame}=({x:.3f}, {y:.3f}, {z:.3f})'
    )
    self._publish_obstacles()

  def _plan_point_to_camera_uv(
    self,
    x: float,
    y: float,
    z: float,
    camera_frame: str,
    camera_info: CameraInfo,
    color_w: int,
    color_h: int,
    stamp,
  ) -> Optional[Tuple[int, int]]:
    pt = PointStamped()
    pt.header.frame_id = self._planning_frame
    pt.point.x = x
    pt.point.y = y
    pt.point.z = z
    for st in ([stamp] if stamp is not None else []) + [rclpy.time.Time().to_msg()]:
      pt.header.stamp = st
      try:
        cam_pt = self._tf_buffer.transform(
          pt,
          camera_frame,
          timeout=rclpy.duration.Duration(seconds=self._tf_timeout),
        )
        return project_camera_xyz_to_uv(
          cam_pt.point.x, cam_pt.point.y, cam_pt.point.z,
          camera_info, color_w, color_h,
        )
      except TransformException:
        continue
    return None

  def _draw_overlay(
    self,
    canvas: np.ndarray,
    camera_info: Optional[CameraInfo] = None,
    camera_frame: str = '',
    stamp=None,
  ) -> np.ndarray:
    out = canvas.copy()
    with self._state_lock:
      mode = self._mode
      target = self._target
      obstacles = list(self._obstacles)

    if target is not None:
      cv2.circle(out, (target.u, target.v), 10, MODE_COLORS['target'], 2)
      cv2.putText(
        out, 'T', (target.u + 12, target.v - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MODE_COLORS['target'], 2, cv2.LINE_AA,
      )
    for obs in obstacles:
      cv2.circle(out, (obs.u, obs.v), 8, MODE_COLORS['obstacle'], 2)
      cv2.putText(
        out, f'O{obs.point_id}', (obs.u + 10, obs.v - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, MODE_COLORS['obstacle'], 1, cv2.LINE_AA,
      )

    if self._show_reprojection and camera_info is not None and camera_frame:
      color_h, color_w = out.shape[:2]
      reproj_color = (0, 255, 255)
      for pt in ([target] if target else []) + obstacles:
        uv = self._plan_point_to_camera_uv(
          pt.x, pt.y, pt.z, camera_frame, camera_info, color_w, color_h, stamp,
        )
        if uv is None:
          continue
        ru, rv = uv
        cv2.drawMarker(
          out, (ru, rv), reproj_color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2,
        )
        err = math.hypot(ru - pt.u, rv - pt.v)
        if err > 8.0:
          cv2.line(out, (pt.u, pt.v), (ru, rv), reproj_color, 1, cv2.LINE_AA)

    lines = [
      f'Mode [1=target 2=obstacle 3=delete]: {mode.upper()}',
      f'Target: {"yes" if target else "no"} | Obstacles: {len(obstacles)}',
      f'Frame: {self._planning_frame} | depth={self._depth_sample_mode} scale={self._depth_scale}',
      'Green/red=c click | yellow x=reproject from base_link',
      'Left-click | c=clear | q=quit',
    ]
    for i, line in enumerate(lines):
      cv2.putText(
        out, line, (8, 24 + i * 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, MODE_COLORS.get(mode, (255, 255, 255)), 2, cv2.LINE_AA,
      )
    return out

  def _handle_keys(self) -> bool:
    if not self._show_window:
      return True
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('1'), ord('2'), ord('3')):
      self._set_mode({ord('1'): 'target', ord('2'): 'obstacle', ord('3'): 'delete'}[key])
    elif key in (ord('c'), ord('C')):
      self._clear_all()
    elif key in (ord('q'), ord('Q'), 27):
      return False
    return True

  def _on_timer(self) -> None:
    if self._show_window and not self._handle_keys():
      rclpy.shutdown()
      return

    with self._data_lock:
      if self._latest_bgr is None or self._latest_header is None:
        return
      if self._latest_depth_m is None or self._latest_camera_info is None:
        if not self._warned_missing_depth:
          self._warned_missing_depth = True
          self.get_logger().warn('Waiting for depth + CameraInfo')
        return
      image_bgr = self._latest_bgr.copy()
      header = self._latest_header
      depth_m = self._latest_depth_m.copy()
      depth_header = self._latest_depth_header
      camera_info = self._latest_camera_info

    color_h, color_w = image_bgr.shape[:2]
    camera_frame = self._camera_frame(camera_info, depth_header, header)

    with self._state_lock:
      pending = self._pending_click
      mode = self._mode
      if pending is not None:
        self._pending_click = None

    if pending is not None:
      self._process_pending_click(
        pending, mode, depth_m, camera_info, color_h, color_w, camera_frame, header.stamp,
      )

    if self._show_window:
      cv2.imshow(
        self._window_name,
        self._draw_overlay(image_bgr, camera_info, camera_frame, header.stamp),
      )


def main() -> None:
  rclpy.init(args=sys.argv)
  node = PerceptionClickPlanningNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    if node._show_window:
      cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == '__main__':
  main()
