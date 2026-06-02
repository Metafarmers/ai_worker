#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Strawberry perception debug node: 2D YOLO overlay on camera image (+ optional 2D RViz markers).
# Default 2D: strawberry2024DB + straw_seg_yolov11_L. 3D/TF → perception_tf_node.py (separate).
# 3D coordinates / TF → use perception_tf_node.py (separate).
#
# Models (config/model/):
#   strawberry2024DB.pt  — detect: background, flower, dead leaves, stalk/root/crown nodes,
#                          unripe/ripe fruit, stem node, ripe completion, calyx
#   straw_seg_yolov11_L.pt — segment: flower, leaf, malformed, riped, stem, unriped
#   straw_seg_v2.pt      — segment: malformed, strawberry
#
# Run (source tree, no colcon rebuild):
#   ./scripts/run_perception_visualize.sh --ros-args -p use_sim_time:=true
#
# View in RViz (robot model alone does not show camera topics):
#   Add → By topic → /perception/visualization/image → Image
#   If still blank: expand Image → Topic → Reliability = Reliable (node default)
#   Also check use_sim_time matches between RViz launch and this node.
# Separate window (optional):
#   apt-get install -y ros-jazzy-image-tools
#   ros2 run image_tools showimage --ros-args -r image:=/perception/visualization/image
#
# Probe class names only:
#   python3 perception_visualize_node.py --probe-only

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Quaternion
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR.parent / 'config' / 'model'

MODEL_CATALOG = {
  'strawberry2024DB': {
    'filename': 'strawberry2024DB.pt',
    'task': 'detect',
    'description': 'Multi-class bbox detector for pollination/harvest scene elements',
  },
  'straw_seg_yolov11_L': {
    'filename': 'straw_seg_yolov11_L.pt',
    'task': 'segment',
    'description': 'Instance segmentation (flower/leaf/fruit/stem)',
  },
  'straw_seg_v2': {
    'filename': 'straw_seg_v2.pt',
    'task': 'segment',
    'description': 'Lightweight strawberry vs malformed segmentation',
  },
}

# BGR colors per class id (cycled)
_CLASS_COLORS = [
  (0, 255, 255),
  (0, 255, 0),
  (255, 0, 0),
  (255, 255, 0),
  (255, 0, 255),
  (0, 128, 255),
  (128, 0, 255),
  (0, 200, 100),
  (100, 100, 255),
  (255, 128, 0),
  (180, 180, 180),
]


@dataclass
class LoadedModel:
  key: str
  path: Path
  task: str
  names: Dict[int, str]
  model: object


def _default_model_path(key: str) -> Path:
  return DEFAULT_MODEL_DIR / MODEL_CATALOG[key]['filename']


def _color_for_class(cls_id: int) -> Tuple[int, int, int]:
  return _CLASS_COLORS[int(cls_id) % len(_CLASS_COLORS)]


def _ensure_ultralytics() -> None:
  try:
    import ultralytics  # noqa: F401
  except ImportError as exc:
    req = SCRIPT_DIR / 'requirements-perception.txt'
    raise ImportError(
      'ultralytics is not installed. Inside the robot Docker container run:\n'
      f'  bash {SCRIPT_DIR / "install_perception_deps.sh"}\n'
      'or:\n'
      f'  python3 -m pip install -r {req}\n'
      'Then restart perception_visualize_node.'
    ) from exc


def _load_yolo_model(path: Path, device: str) -> LoadedModel:
  _ensure_ultralytics()
  from ultralytics import YOLO

  yolo = YOLO(str(path))
  key = path.stem
  return LoadedModel(
    key=key,
    path=path,
    task=str(yolo.task),
    names={int(k): str(v) for k, v in yolo.names.items()},
    model=yolo,
  )


def probe_models(paths: Sequence[Path]) -> None:
  print('=== Perception model catalog ===\n')
  for key, meta in MODEL_CATALOG.items():
    print(f'[{key}]')
    print(f'  file: {meta["filename"]}')
    print(f'  expected task: {meta["task"]}')
    print(f'  note: {meta["description"]}')
    print()
  print('=== Loaded checkpoint metadata ===\n')
  for path in paths:
    if not path.is_file():
      print(f'{path}: MISSING')
      continue
    loaded = _load_yolo_model(path, 'cpu')
    print(f'{path.name}')
    print(f'  task: {loaded.task}')
    for idx, name in sorted(loaded.names.items()):
      print(f'    {idx}: {name}')
    print()


SKIP_DRAW_LABELS = frozenset({'background'})


def _result_box_count(result, names: Optional[Dict[int, str]] = None) -> int:
  if result.boxes is None or len(result.boxes) == 0:
    return 0
  if names is None:
    return len(result.boxes)
  boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
  return sum(
    1 for cls_id in boxes_cls
    if names.get(int(cls_id), str(cls_id)).lower() not in SKIP_DRAW_LABELS
  )


def _build_predict_kwargs(
  conf: float,
  device: str,
  imgsz: int,
) -> dict:
  kwargs = dict(conf=conf, device=device, verbose=False)
  if imgsz > 0:
    kwargs['imgsz'] = imgsz
  return kwargs


def _draw_detections(
  image_bgr: np.ndarray,
  result,
  names: Dict[int, str],
  prefix: str,
) -> np.ndarray:
  out = image_bgr.copy()
  if result.boxes is None or len(result.boxes) == 0:
    return out

  boxes_xyxy = result.boxes.xyxy.cpu().numpy()
  boxes_conf = result.boxes.conf.cpu().numpy()
  boxes_cls = result.boxes.cls.cpu().numpy().astype(int)

  for i in range(len(boxes_xyxy)):
    cls_id = int(boxes_cls[i])
    label = names.get(cls_id, str(cls_id))
    if label.lower() in SKIP_DRAW_LABELS:
      continue
    x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
    conf = float(boxes_conf[i])
    color = _color_for_class(cls_id)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    text = f'{prefix}{label} {conf:.2f}'
    cv2.putText(
      out, text, (x1, max(0, y1 - 6)),
      cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )
  return out


def _mask_bool_at_image_size(mask: np.ndarray, height: int, width: int) -> np.ndarray:
  """Resize YOLO mask (often letterboxed) to match the source image (H, W)."""
  if mask.shape[0] == height and mask.shape[1] == width:
    return mask > 0.5
  resized = cv2.resize(
    mask.astype(np.float32),
    (width, height),
    interpolation=cv2.INTER_LINEAR,
  )
  return resized > 0.5


def _draw_segmentation(
  image_bgr: np.ndarray,
  result,
  names: Dict[int, str],
  prefix: str,
  mask_alpha: float,
) -> np.ndarray:
  out = image_bgr.copy()
  if result.masks is None or result.boxes is None:
    return out

  masks = result.masks.data.cpu().numpy()
  boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
  boxes_conf = result.boxes.conf.cpu().numpy()
  img_h, img_w = out.shape[:2]

  overlay = out.copy()
  for i in range(len(masks)):
    mask = _mask_bool_at_image_size(masks[i], img_h, img_w)
    cls_id = int(boxes_cls[i])
    color = np.array(_color_for_class(cls_id), dtype=np.float32)
    overlay[mask] = (overlay[mask].astype(np.float32) * (1.0 - mask_alpha)
                     + color * mask_alpha).astype(np.uint8)

  out = overlay
  boxes_xyxy = result.boxes.xyxy.cpu().numpy()
  for i in range(len(boxes_xyxy)):
    x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
    cls_id = int(boxes_cls[i])
    label = names.get(cls_id, str(cls_id))
    conf = float(boxes_conf[i])
    color = _color_for_class(cls_id)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
    cv2.putText(
      out, f'{prefix}{label} {conf:.2f}', (x1, max(0, y1 - 6)),
      cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )
  return out


def _results_to_markers(
  stamp,
  frame_id: str,
  result,
  names: Dict[int, str],
  model_key: str,
  marker_id_base: int,
  default_depth_m: float,
) -> List[Marker]:
  markers: List[Marker] = []
  if result.boxes is None or len(result.boxes) == 0:
    return markers

  boxes_xyxy = result.boxes.xyxy.cpu().numpy()
  boxes_conf = result.boxes.conf.cpu().numpy()
  boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
  h, w = result.orig_shape

  for i in range(len(boxes_xyxy)):
    x1, y1, x2, y2 = boxes_xyxy[i]
    cx = 0.5 * (x1 + x2) / float(w)
    cy = 0.5 * (y1 + y2) / float(h)
    cls_id = int(boxes_cls[i])
    label = names.get(cls_id, str(cls_id))
    conf = float(boxes_conf[i])
    color = _color_for_class(cls_id)
    norm_color = ColorRGBA(
      r=color[2] / 255.0,
      g=color[1] / 255.0,
      b=color[0] / 255.0,
      a=0.9,
    )

    text = Marker()
    text.header = Header(stamp=stamp, frame_id=frame_id)
    text.ns = f'perception_{model_key}'
    text.id = marker_id_base + i
    text.type = Marker.TEXT_VIEW_FACING
    text.action = Marker.ADD
    text.pose.position = Point(
      x=float(default_depth_m) * (cx - 0.5) * 2.0,
      y=float(default_depth_m) * (cy - 0.5) * 2.0,
      z=float(default_depth_m),
    )
    text.pose.orientation = Quaternion(w=1.0)
    text.scale.z = 0.04
    text.color = norm_color
    text.text = f'{label} ({conf:.2f})'
    markers.append(text)

    line = Marker()
    line.header = text.header
    line.ns = f'perception_{model_key}_box'
    line.id = marker_id_base + 10000 + i
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.scale.x = 0.003
    line.color = norm_color
    line.pose.orientation = Quaternion(w=1.0)
    z = float(default_depth_m)
    sx = float(default_depth_m) * (x1 / float(w) - 0.5) * 2.0
    sy = float(default_depth_m) * (y1 / float(h) - 0.5) * 2.0
    ex = float(default_depth_m) * (x2 / float(w) - 0.5) * 2.0
    ey = float(default_depth_m) * (y2 / float(h) - 0.5) * 2.0
    line.points = [
      Point(x=sx, y=sy, z=z),
      Point(x=ex, y=sy, z=z),
      Point(x=ex, y=ey, z=z),
      Point(x=sx, y=ey, z=z),
      Point(x=sx, y=sy, z=z),
    ]
    markers.append(line)

  return markers


class PerceptionVisualizeNode(Node):
  def __init__(self) -> None:
    super().__init__('perception_visualize_node')

    self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
    self.declare_parameter('output_image_topic', '/perception/visualization/image')
    self.declare_parameter('output_marker_topic', '/perception/visualization/markers')
    self.declare_parameter('inference_hz', 3.0)
    self.declare_parameter('conf_threshold', 0.15)
    self.declare_parameter('inference_imgsz', 640)
    self.declare_parameter('device', 'cpu')
    self.declare_parameter('mask_alpha', 0.35)
    self.declare_parameter('marker_depth_m', 1.0)
    # 2D overlay: det + seg_yolov11 worked on ZED head camera in testing.
    # seg_v2 alone often finds nothing at head-camera distance — enable via param if needed.
    self.declare_parameter('enable_det', False)
    self.declare_parameter('enable_seg_yolov11', False)
    self.declare_parameter('enable_seg_v2', False)
    self.declare_parameter('det_model_path', str(_default_model_path('strawberry2024DB')))
    self.declare_parameter('seg_yolov11_model_path', str(_default_model_path('straw_seg_yolov11_L')))
    self.declare_parameter('seg_v2_model_path', str(_default_model_path('straw_seg_v2')))
    self.declare_parameter('output_image_reliable_qos', True)
    self.declare_parameter('output_image_best_effort_qos', True)
    self.declare_parameter('use_node_clock_stamp', False)

    image_topic = str(self.get_parameter('image_topic').value)
    out_image_topic = str(self.get_parameter('output_image_topic').value)
    out_marker_topic = str(self.get_parameter('output_marker_topic').value)
    self._conf = float(self.get_parameter('conf_threshold').value)
    self._imgsz = int(self.get_parameter('inference_imgsz').value)
    self._device = str(self.get_parameter('device').value)
    self._mask_alpha = float(self.get_parameter('mask_alpha').value)
    self._marker_depth_m = float(self.get_parameter('marker_depth_m').value)

    self._bridge = CvBridge()
    self._image_lock = threading.Lock()
    self._latest_bgr: Optional[np.ndarray] = None
    self._latest_header: Optional[Header] = None

    self._models: List[LoadedModel] = []
    self._load_enabled_models()

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
    self.create_subscription(Image, image_topic, self._on_image, sensor_qos)

    self._image_pubs: List = []
    qos_notes: List[str] = []
    if bool(self.get_parameter('output_image_reliable_qos').value):
      self._image_pubs.append(self.create_publisher(Image, out_image_topic, reliable_qos))
      qos_notes.append('RELIABLE')
    if bool(self.get_parameter('output_image_best_effort_qos').value):
      self._image_pubs.append(self.create_publisher(Image, out_image_topic, sensor_qos))
      qos_notes.append('BEST_EFFORT')
    if not self._image_pubs:
      raise RuntimeError('Enable at least one of output_image_reliable_qos / output_image_best_effort_qos')
    qos_note = ' + '.join(qos_notes)
    self._use_node_clock_stamp = bool(self.get_parameter('use_node_clock_stamp').value)
    self._logged_first_publish = False
    self._sanity_checked_models: set = set()
    self._frames_received = 0
    self._warned_no_camera = False
    self._marker_pub = self.create_publisher(MarkerArray, out_marker_topic, 10)

    hz = max(float(self.get_parameter('inference_hz').value), 0.1)
    self.create_timer(1.0 / hz, self._on_inference_timer)

    self.get_logger().info(f'Script: {Path(__file__).resolve()}')
    self.get_logger().info(f'Subscribe Image: {image_topic}')
    self.get_logger().info(f'Publish Image: {out_image_topic}')
    self.get_logger().info(f'Publish MarkerArray (2D): {out_marker_topic}')
    self.get_logger().info('3D/TF: run ./scripts/run_perception_tf.sh separately')
    self.get_logger().info(f'Loaded models: {[m.key for m in self._models]}')
    self.get_logger().info(
      f'Inference: conf>={self._conf:.2f}, imgsz={self._imgsz if self._imgsz > 0 else "auto"}'
    )
    for m in self._models:
      classes = ', '.join(f'{i}:{n}' for i, n in sorted(m.names.items()))
      self.get_logger().info(f'  {m.key} ({m.task}): {classes}')
    self.get_logger().info(f'Output image QoS: {qos_note}')
    self.get_logger().info(
      'RViz: Add → By topic → %s → Image' % out_image_topic
    )
    self.get_logger().info(
      'RViz Image blank? expand Topic → Reliability: try Reliable, then Best Effort'
    )

  def _load_enabled_models(self) -> None:
    specs: List[Tuple[str, str, bool]] = [
      ('enable_det', 'det_model_path', False),
      ('enable_seg_yolov11', 'seg_yolov11_model_path', False),
      ('enable_seg_v2', 'seg_v2_model_path', False),
    ]
    if not any(bool(self.get_parameter(p).value) for p, _, _ in specs):
      self.get_logger().warn(
        'No YOLO models enabled — publishing raw camera passthrough only. '
        'Use run_perception_click_tf.sh for manual TF.'
      )
      return
    try:
      _ensure_ultralytics()
    except ImportError as exc:
      self.get_logger().error(str(exc))
      raise

    for enable_param, path_param, _default in specs:
      if not bool(self.get_parameter(enable_param).value):
        continue
      path = Path(str(self.get_parameter(path_param).value))
      if not path.is_file():
        self.get_logger().error(f'Model not found: {path} (from {enable_param})')
        continue
      try:
        loaded = _load_yolo_model(path, self._device)
        self._models.append(loaded)
        self.get_logger().info(f'Loaded {loaded.key} task={loaded.task} from {path}')
      except Exception as exc:
        self.get_logger().error(f'Failed to load {path}: {exc}')

    if not self._models:
      self.get_logger().warn('No models loaded — inference disabled.')
      return

  def _on_image(self, msg: Image) -> None:
    try:
      encoding = msg.encoding.lower()
      if encoding in ('rgb8', 'rgba8'):
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
      else:
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as exc:
      self.get_logger().warn(f'cv_bridge failed ({msg.encoding}): {exc}')
      return

    with self._image_lock:
      self._latest_bgr = bgr
      self._latest_header = msg.header
      self._frames_received += 1

  def _on_inference_timer(self) -> None:
    with self._image_lock:
      if self._latest_bgr is None or self._latest_header is None:
        if not self._warned_no_camera:
          self._warned_no_camera = True
          self.get_logger().warn(
            f'No camera frames on {self.get_parameter("image_topic").value} — '
            'check ZED node / use_sim_time'
          )
        return
      image_bgr = self._latest_bgr.copy()
      header = self._latest_header

    stamp = header.stamp
    frame_id = header.frame_id or 'camera_optical_frame'
    canvas = image_bgr.copy()
    marker_array = MarkerArray()
    delete_all = Marker()
    delete_all.action = Marker.DELETEALL
    marker_array.markers.append(delete_all)
    marker_id = 0
    total_detections = 0
    per_model_counts: List[str] = []

    for loaded in self._models:
      try:
        predict_kwargs = _build_predict_kwargs(self._conf, self._device, self._imgsz)

        if loaded.key not in self._sanity_checked_models:
          sanity_kwargs = _build_predict_kwargs(0.01, self._device, self._imgsz)
          sanity = loaded.model.predict(image_bgr, **sanity_kwargs)
          if sanity:
            n_low = _result_box_count(sanity[0], loaded.names)
            self.get_logger().info(
              f'Sanity {loaded.key}: {n_low} box(es) at conf=0.01 '
              f'(excl. background), image {image_bgr.shape[1]}x{image_bgr.shape[0]}, '
              f'imgsz={self._imgsz if self._imgsz > 0 else "auto"}'
            )
          self._sanity_checked_models.add(loaded.key)

        results = loaded.model.predict(image_bgr, **predict_kwargs)
        if not results:
          continue
        res = results[0]
        n_boxes = _result_box_count(res, loaded.names)
        total_detections += n_boxes
        per_model_counts.append(f'{loaded.key}={n_boxes}')
        if n_boxes == 0:
          self.get_logger().debug(
            f'{loaded.key}: 0 detections at conf>={self._conf:.2f}',
            throttle_duration_sec=5.0,
          )

        prefix = f'{loaded.key}:'
        if loaded.task == 'detect':
          canvas = _draw_detections(canvas, res, loaded.names, prefix)
        elif loaded.task == 'segment':
          canvas = _draw_segmentation(
            canvas, res, loaded.names, prefix, self._mask_alpha,
          )
        else:
          self.get_logger().warn(f'Unsupported task {loaded.task} for {loaded.key}')
          continue
      except Exception as exc:
        self.get_logger().warn(f'Visualization failed ({loaded.key}): {exc}')
        continue

      marker_id += 20000
      marker_array.markers.extend(
        _results_to_markers(
          stamp, frame_id, res, loaded.names, loaded.key,
          marker_id, self._marker_depth_m,
        )
      )

    legend_y = 24
    status_color = (0, 255, 0) if total_detections > 0 else (0, 0, 255)
    cv2.putText(
      canvas,
      f'detections: {total_detections} | conf>={self._conf:.2f} | models: {", ".join(m.key for m in self._models)}',
      (8, legend_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      status_color,
      1,
      cv2.LINE_AA,
    )
    if total_detections == 0:
      cv2.putText(
        canvas,
        'NO DETECTIONS — try -p conf_threshold:=0.05 or check camera view',
        (8, legend_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
      )

    self.get_logger().info(
      f'Frame detections: {total_detections} ({", ".join(per_model_counts) or "none"}) '
      f'conf>={self._conf:.2f}, camera_frames={self._frames_received}',
      throttle_duration_sec=3.0,
    )

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    out_msg = self._bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
    out_msg.header.frame_id = frame_id
    if self._use_node_clock_stamp:
      out_msg.header.stamp = self.get_clock().now().to_msg()
    else:
      out_msg.header.stamp = stamp

    for pub in self._image_pubs:
      pub.publish(out_msg)
    self._marker_pub.publish(marker_array)

    if not self._logged_first_publish:
      self._logged_first_publish = True
      self.get_logger().info(
        f'First image published: {out_msg.width}x{out_msg.height} {out_msg.encoding} '
        f'frame={out_msg.header.frame_id}'
      )


def _probe_only_from_argv(argv: Sequence[str]) -> bool:
  return '--probe-only' in argv


def main() -> None:
  if _probe_only_from_argv(sys.argv):
    paths = [_default_model_path(k) for k in MODEL_CATALOG]
    probe_models(paths)
    return

  rclpy.init(args=sys.argv)
  node = PerceptionVisualizeNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
