#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# 3D perception + TF (YOLO, optional). Manual click → use perception_click_tf_node.py.
#
# Run:
#   ./scripts/run_perception_tf.sh          # YOLO off by default
#   ./scripts/run_perception_click_tf.sh    # mouse click targets/obstacles

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from perception_3d_utils import (
  Detection3D,
  depth_image_to_meters,
  detections_3d_from_detection,
  detections_3d_from_segmentation,
  detections_to_markers,
  detections_to_transforms,
)
from perception_visualize_node import (
  LoadedModel,
  _build_predict_kwargs,
  _default_model_path,
  _ensure_ultralytics,
  _load_yolo_model,
)

SCRIPT_DIR = Path(__file__).resolve().parent


class PerceptionTfNode(Node):
  def __init__(self) -> None:
    super().__init__('perception_tf_node')

    self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
    self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
    self.declare_parameter('depth_camera_info_topic', '/zed/zed_node/rgb/camera_info')
    self.declare_parameter('output_marker_topic', '/perception/objects/markers')
    self.declare_parameter('inference_hz', 3.0)
    self.declare_parameter('conf_threshold', 0.15)
    self.declare_parameter('inference_imgsz', 640)
    self.declare_parameter('device', 'cpu')
    self.declare_parameter('min_depth_m', 0.15)
    self.declare_parameter('max_depth_m', 10.0)
    self.declare_parameter('min_mask_pixels', 30)
    self.declare_parameter('depth_sample_mode', 'closest')
    self.declare_parameter('publish_tf', True)
    self.declare_parameter('enable_det', False)
    self.declare_parameter('enable_seg_yolov11', False)
    self.declare_parameter('enable_seg_v2', False)
    # Match visualize node defaults (det + seg_yolov11 on ZED head camera).
    self.declare_parameter('det_model_path', str(_default_model_path('strawberry2024DB')))
    self.declare_parameter('seg_yolov11_model_path', str(_default_model_path('straw_seg_yolov11_L')))
    self.declare_parameter('seg_v2_model_path', str(_default_model_path('straw_seg_v2')))
    self.declare_parameter('use_node_clock_stamp', True)
    # Empty = camera_info.header.frame_id (matches RGB deprojection). Override if needed.
    self.declare_parameter('tf_parent_frame', '')

    self._conf = float(self.get_parameter('conf_threshold').value)
    self._imgsz = int(self.get_parameter('inference_imgsz').value)
    self._device = str(self.get_parameter('device').value)
    self._min_depth = float(self.get_parameter('min_depth_m').value)
    self._max_depth = float(self.get_parameter('max_depth_m').value)
    self._min_mask_pixels = int(self.get_parameter('min_mask_pixels').value)
    self._depth_sample_mode = str(self.get_parameter('depth_sample_mode').value)
    self._publish_tf = bool(self.get_parameter('publish_tf').value)
    self._use_node_clock_stamp = bool(self.get_parameter('use_node_clock_stamp').value)
    self._tf_parent_override = str(self.get_parameter('tf_parent_frame').value).strip()

    self._bridge = CvBridge()
    self._data_lock = threading.Lock()
    self._latest_bgr = None
    self._latest_header = None
    self._latest_depth_m = None
    self._latest_depth_header = None
    self._latest_camera_info: Optional[CameraInfo] = None
    self._warned_missing_depth = False
    self._logged_depth_rx = False
    self._logged_info_rx = False

    self._models: List[LoadedModel] = []
    self._load_enabled_models()

    # ZED on this robot publishes depth as RELIABLE (see: ros2 topic info -v).
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

    hz = max(float(self.get_parameter('inference_hz').value), 0.1)
    self.create_timer(1.0 / hz, self._on_timer)

    self.get_logger().info(f'Script: {Path(__file__).resolve()}')
    self.get_logger().info(f'Color: {image_topic}')
    self.get_logger().info(f'Depth: {depth_topic}')
    self.get_logger().info(f'CameraInfo: {depth_info_topic}')
    self.get_logger().info(f'Markers: {marker_topic}')
    self.get_logger().info(f'Models: {[m.key for m in self._models]}')
    if self._publish_tf:
      self.get_logger().info(
        'TF: parent=camera optical frame → child perception/<label>_<id> '
        '(positions in base_link in RViz are just the fixed-frame view)'
      )

  def _load_enabled_models(self) -> None:
    specs: List[Tuple[str, str]] = [
      ('enable_det', 'det_model_path'),
      ('enable_seg_yolov11', 'seg_yolov11_model_path'),
      ('enable_seg_v2', 'seg_v2_model_path'),
    ]
    if not any(bool(self.get_parameter(p).value) for p, _ in specs):
      return
    _ensure_ultralytics()
    for enable_param, path_param in specs:
      if not bool(self.get_parameter(enable_param).value):
        continue
      path = Path(str(self.get_parameter(path_param).value))
      if not path.is_file():
        self.get_logger().error(f'Model not found: {path}')
        continue
      loaded = _load_yolo_model(path, self._device)
      self._models.append(loaded)
      self.get_logger().info(f'Loaded {loaded.key} from {path}')
    if not self._models:
      self.get_logger().warn(
        'No YOLO models enabled. Use run_perception_click_tf.sh for manual TF, '
        'or enable_det / enable_seg_yolov11 via --ros-args.'
      )
      return

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
      if not self._logged_depth_rx:
        self._logged_depth_rx = True
        self.get_logger().info(
          f'Receiving depth: {msg.width}x{msg.height} {msg.encoding} '
          f'frame={msg.header.frame_id}'
        )

  def _on_camera_info(self, msg: CameraInfo) -> None:
    with self._data_lock:
      self._latest_camera_info = msg
      if not self._logged_info_rx:
        self._logged_info_rx = True
        self.get_logger().info(
          f'Receiving CameraInfo: {msg.width}x{msg.height} frame={msg.header.frame_id}'
        )

  def _on_timer(self) -> None:
    if not self._models:
      return
    with self._data_lock:
      if self._latest_bgr is None or self._latest_header is None:
        return
      if self._latest_depth_m is None or self._latest_camera_info is None:
        if not self._warned_missing_depth:
          self._warned_missing_depth = True
          self.get_logger().warn(
            'Waiting for depth + CameraInfo. '
            'If depth publisher is RELIABLE, subscriber must be RELIABLE too '
            '(check: ros2 topic info /zed/zed_node/depth/depth_registered -v).'
          )
        return
      image_bgr = self._latest_bgr.copy()
      header = self._latest_header
      depth_m = self._latest_depth_m.copy()
      depth_header = self._latest_depth_header
      camera_info = self._latest_camera_info

    stamp = header.stamp
    # 3D points are deprojected with rgb/camera_info intrinsics → parent must be that optical frame.
    camera_frame = (
      self._tf_parent_override
      or (camera_info.header.frame_id if camera_info.header.frame_id else '')
      or (depth_header.frame_id if depth_header is not None and depth_header.frame_id else '')
      or (header.frame_id if header.frame_id else 'camera_optical_frame')
    )
    color_h, color_w = image_bgr.shape[:2]
    all_detections: List[Detection3D] = []

    for loaded in self._models:
      try:
        predict_kwargs = _build_predict_kwargs(self._conf, self._device, self._imgsz)
        results = loaded.model.predict(image_bgr, **predict_kwargs)
        if not results:
          continue
        res = results[0]
        n_raw = len(res.boxes) if res.boxes is not None else 0
        if loaded.task == 'segment':
          dets = detections_3d_from_segmentation(
            res, loaded.names, depth_m, camera_info, color_h, color_w,
            self._min_depth, self._max_depth, self._min_mask_pixels,
            depth_sample_mode=self._depth_sample_mode,
          )
        else:
          dets = detections_3d_from_detection(
            res, loaded.names, depth_m, camera_info, color_h, color_w,
            self._min_depth, self._max_depth,
          )
        all_detections.extend(dets)
        if n_raw > 0 and not dets:
          self.get_logger().warn(
            f'{loaded.key}: {n_raw} 2D detection(s) but 0 with valid depth '
            f'(depth range {self._min_depth:.2f}-{self._max_depth:.2f}m)',
            throttle_duration_sec=5.0,
          )
      except Exception as exc:
        self.get_logger().warn(f'TF inference failed ({loaded.key}): {exc}')
        continue

    self.get_logger().info(
      f'3D objects: {len(all_detections)} TF(s), parent={camera_frame}, conf>={self._conf:.2f}',
      throttle_duration_sec=3.0,
    )

    tf_stamp = self.get_clock().now().to_msg() if self._use_node_clock_stamp else stamp
    if self._publish_tf and all_detections:
      self._tf_broadcaster.sendTransform(
        detections_to_transforms(tf_stamp, camera_frame, all_detections)
      )

    marker_array = MarkerArray()
    delete_all = Marker()
    delete_all.action = Marker.DELETEALL
    marker_array.markers.append(delete_all)
    marker_array.markers.extend(
      detections_to_markers(stamp, camera_frame, all_detections, 0)
    )
    self._marker_pub.publish(marker_array)


def main() -> None:
  rclpy.init(args=sys.argv)
  node = PerceptionTfNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
