#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Manual click → 3D TF for planning debug (no YOLO).
#
# Run:
#   ./scripts/run_perception_click_tf.sh
#
# Keys (OpenCV window focused):
#   1 — target mode (green TF: perception/target_N)
#   2 — obstacle mode (red TF: perception/obstacle_N)
#   3 — delete nearest point to click
#   c — clear all points
#   q — quit
#
# Markers: /perception/objects/markers

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from perception_3d_utils import (
  ManualPoint3D,
  depth_image_to_meters,
  manual_points_to_markers,
  manual_points_to_transforms,
  sample_depth_at_uv,
)

ClickMode = Literal['target', 'obstacle', 'delete']

MODE_COLORS = {
  'target': (0, 255, 0),
  'obstacle': (0, 0, 255),
  'delete': (0, 165, 255),
}


@dataclass
class _PendingClick:
  u: int
  v: int


class PerceptionClickTfNode(Node):
  def __init__(self) -> None:
    super().__init__('perception_click_tf_node')

    self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
    self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
    self.declare_parameter('depth_camera_info_topic', '/zed/zed_node/rgb/camera_info')
    self.declare_parameter('output_marker_topic', '/perception/objects/markers')
    self.declare_parameter('min_depth_m', 0.15)
    self.declare_parameter('max_depth_m', 10.0)
    self.declare_parameter('depth_patch_radius', 7)
    self.declare_parameter('delete_radius_px', 40)
    self.declare_parameter('publish_hz', 10.0)
    self.declare_parameter('show_window', True)
    self.declare_parameter('window_name', 'perception_click_tf')
    self.declare_parameter('use_node_clock_stamp', True)
    self.declare_parameter('tf_parent_frame', '')

    self._min_depth = float(self.get_parameter('min_depth_m').value)
    self._max_depth = float(self.get_parameter('max_depth_m').value)
    self._patch_radius = int(self.get_parameter('depth_patch_radius').value)
    self._delete_radius_px = int(self.get_parameter('delete_radius_px').value)
    self._show_window = bool(self.get_parameter('show_window').value)
    self._window_name = str(self.get_parameter('window_name').value)
    self._use_node_clock_stamp = bool(self.get_parameter('use_node_clock_stamp').value)
    self._tf_parent_override = str(self.get_parameter('tf_parent_frame').value).strip()

    self._bridge = CvBridge()
    self._data_lock = threading.Lock()
    self._state_lock = threading.Lock()
    self._latest_bgr: Optional[np.ndarray] = None
    self._latest_header = None
    self._latest_depth_m: Optional[np.ndarray] = None
    self._latest_depth_header = None
    self._latest_camera_info: Optional[CameraInfo] = None
    self._warned_missing_depth = False

    self._mode: ClickMode = 'target'
    self._pending_click: Optional[_PendingClick] = None
    self._points: List[ManualPoint3D] = []
    self._next_target_id = 0
    self._next_obstacle_id = 0

    sensor_qos = QoSProfile(
      reliability=ReliabilityPolicy.BEST_EFFORT,
      history=HistoryPolicy.KEEP_LAST,
      depth=1,
      durability=DurabilityPolicy.VOLATILE,
    )
    reliable_qos = QoSProfile(
      reliability=ReliabilityPolicy.RELIABLE,
      history=HistoryPolicy.KEEP_LAST,
      depth=10,
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
    marker_topic = str(self.get_parameter('output_marker_topic').value)

    self.create_subscription(Image, image_topic, self._on_image, sensor_qos)
    self.create_subscription(Image, image_topic, self._on_image, reliable_qos)
    self.create_subscription(Image, depth_topic, self._on_depth, reliable_qos)
    self.create_subscription(CameraInfo, depth_info_topic, self._on_camera_info, camera_info_qos)
    self.create_subscription(CameraInfo, depth_info_topic, self._on_camera_info, reliable_qos)
    self._marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
    self._tf_broadcaster = TransformBroadcaster(self)

    hz = max(float(self.get_parameter('publish_hz').value), 1.0)
    self.create_timer(1.0 / hz, self._on_timer)

    if self._show_window:
      cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
      cv2.setMouseCallback(self._window_name, self._on_mouse)

    self.get_logger().info(f'Script: {Path(__file__).resolve()}')
    self.get_logger().info(f'Color: {image_topic}')
    self.get_logger().info(f'Depth: {depth_topic}')
    self.get_logger().info(f'Markers: {marker_topic}')
    self.get_logger().info(
      'Keys: 1=target, 2=obstacle, 3=delete click, c=clear all, q=quit'
    )
    if self._show_window:
      self.get_logger().info(f'OpenCV window: {self._window_name} (click here after selecting mode)')
    else:
      self.get_logger().warn('show_window=false — enable window or use params on headless setup')

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
      self._points.clear()
      self._next_target_id = 0
      self._next_obstacle_id = 0
    self.get_logger().info('Cleared all click points')

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
      self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}', throttle_duration_sec=5.0)
      return
    with self._data_lock:
      self._latest_depth_m = depth_m
      self._latest_depth_header = msg.header

  def _on_camera_info(self, msg: CameraInfo) -> None:
    with self._data_lock:
      self._latest_camera_info = msg

  def _camera_frame(
    self,
    camera_info: CameraInfo,
    depth_header,
    color_header,
  ) -> str:
    return (
      self._tf_parent_override
      or (camera_info.header.frame_id if camera_info.header.frame_id else '')
      or (depth_header.frame_id if depth_header is not None and depth_header.frame_id else '')
      or (color_header.frame_id if color_header is not None and color_header.frame_id else '')
      or 'camera_optical_frame'
    )

  def _process_pending_click(
    self,
    click: _PendingClick,
    mode: ClickMode,
    depth_m: np.ndarray,
    camera_info: CameraInfo,
    color_h: int,
    color_w: int,
  ) -> None:
    if mode == 'delete':
      self._delete_nearest(click.u, click.v)
      return

    xyz = sample_depth_at_uv(
      click.u, click.v, depth_m, color_h, color_w, camera_info,
      self._min_depth, self._max_depth, self._patch_radius,
    )
    if xyz is None:
      self.get_logger().warn(
        f'No valid depth at ({click.u}, {click.v}) — click closer / check depth topic'
      )
      return

    x, y, z = xyz
    with self._state_lock:
      if mode == 'target':
        point_id = self._next_target_id
        self._next_target_id += 1
      else:
        point_id = self._next_obstacle_id
        self._next_obstacle_id += 1
      pt = ManualPoint3D(
        kind=mode,
        point_id=point_id,
        u=click.u,
        v=click.v,
        x=x,
        y=y,
        z=z,
      )
      self._points.append(pt)
    self.get_logger().info(
      f'Added {mode}_{point_id} at pixel ({click.u},{click.v}) '
      f'→ ({x:.3f}, {y:.3f}, {z:.3f}) m in camera frame'
    )

  def _delete_nearest(self, u: int, v: int) -> None:
    with self._state_lock:
      if not self._points:
        self.get_logger().info('Delete: no points to remove')
        return
      best_idx = min(
        range(len(self._points)),
        key=lambda i: (self._points[i].u - u) ** 2 + (self._points[i].v - v) ** 2,
      )
      dist = ((self._points[best_idx].u - u) ** 2 + (self._points[best_idx].v - v) ** 2) ** 0.5
      if dist > self._delete_radius_px:
        self.get_logger().info(
          f'Delete: nearest point {dist:.0f}px away (limit {self._delete_radius_px}px)'
        )
        return
      removed = self._points.pop(best_idx)
    self.get_logger().info(f'Deleted {removed.kind}_{removed.point_id}')

  def _draw_overlay(self, canvas: np.ndarray) -> np.ndarray:
    out = canvas.copy()
    with self._state_lock:
      mode = self._mode
      points = list(self._points)

    for pt in points:
      color = MODE_COLORS['target'] if pt.kind == 'target' else MODE_COLORS['obstacle']
      cv2.circle(out, (pt.u, pt.v), 8, color, 2)
      cv2.putText(
        out, f'{pt.kind[0]}{pt.point_id}', (pt.u + 10, pt.v - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
      )

    mode_color = MODE_COLORS[mode]
    lines = [
      f'Mode [1=target 2=obstacle 3=delete]: {mode.upper()}',
      f'Points: target={sum(p.kind == "target" for p in points)} '
      f'obstacle={sum(p.kind == "obstacle" for p in points)}',
      'Left-click in this window | c=clear | q=quit',
    ]
    for i, line in enumerate(lines):
      cv2.putText(
        out, line, (8, 24 + i * 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA,
      )
    return out

  def _handle_keys(self) -> bool:
    """Return False to request shutdown."""
    if not self._show_window:
      return True
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('1'), ord('2'), ord('3')):
      mode_map = {ord('1'): 'target', ord('2'): 'obstacle', ord('3'): 'delete'}
      self._set_mode(mode_map[key])
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
      points = list(self._points)

    if pending is not None:
      self._process_pending_click(
        pending, mode, depth_m, camera_info, color_h, color_w,
      )
      with self._state_lock:
        points = list(self._points)

    stamp = header.stamp
    tf_stamp = self.get_clock().now().to_msg() if self._use_node_clock_stamp else stamp

    if points:
      self._tf_broadcaster.sendTransform(
        manual_points_to_transforms(tf_stamp, camera_frame, points)
      )

    marker_array = MarkerArray()
    delete_all = Marker()
    delete_all.action = Marker.DELETEALL
    marker_array.markers.append(delete_all)
    marker_array.markers.extend(
      manual_points_to_markers(stamp, camera_frame, points)
    )
    self._marker_pub.publish(marker_array)

    if self._show_window:
      display = self._draw_overlay(image_bgr)
      cv2.imshow(self._window_name, display)


def main() -> None:
  rclpy.init(args=sys.argv)
  node = PerceptionClickTfNode()
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
