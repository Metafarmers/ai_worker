#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Shared 3D deprojection helpers for perception TF node."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Quaternion, TransformStamped, Vector3
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker

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
class Detection3D:
  label: str
  cls_id: int
  conf: float
  u: int
  v: int
  x: float
  y: float
  z: float


def color_for_class(cls_id: int) -> Tuple[int, int, int]:
  return _CLASS_COLORS[int(cls_id) % len(_CLASS_COLORS)]


def sanitize_tf_child_name(label: str, index: int) -> str:
  safe = ''.join(c if c.isalnum() or c in '_-' else '_' for c in label.lower())
  return f'perception/{safe}_{index}'


def mask_bool_at_image_size(mask: np.ndarray, height: int, width: int) -> np.ndarray:
  if mask.shape[0] == height and mask.shape[1] == width:
    return mask > 0.5
  resized = cv2.resize(
    mask.astype(np.float32),
    (width, height),
    interpolation=cv2.INTER_LINEAR,
  )
  return resized > 0.5


def depth_image_to_meters(msg: Image, bridge: CvBridge) -> Optional[np.ndarray]:
  try:
    if msg.encoding in ('32FC1', '32fc1'):
      return bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1').astype(np.float32)
    if msg.encoding in ('16UC1', '16uc1'):
      depth_mm = bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
      return depth_mm.astype(np.float32) * 0.001
  except Exception:
    return None
  return None


def read_point_cloud_xyz(msg: PointCloud2) -> Optional[np.ndarray]:
  """Organized cloud_registered → (H, W, 3) XYZ in camera optical frame [m]."""
  if msg.height < 2 or msg.width < 2:
    return None
  try:
    from sensor_msgs_py import point_cloud2 as pc2
    pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=False)
    xyz = pts.reshape(msg.height, msg.width, 3).astype(np.float32)
    return xyz
  except Exception:
    return None


def sample_xyz_from_point_cloud(
  u: int,
  v: int,
  color_h: int,
  color_w: int,
  cloud_xyz: np.ndarray,
  min_depth: float,
  max_depth: float,
  patch_radius: int = 3,
  sample_mode: str = 'closest',
) -> Optional[Tuple[float, float, float]]:
  """Sample 3D point from ZED organized cloud (SDK ground truth, no deproject)."""
  ch, cw = cloud_xyz.shape[:2]
  du, dv = depth_uv_from_color_uv(float(u), float(v), color_w, color_h, cw, ch)
  du_i = int(round(du))
  dv_i = int(round(dv))
  r = max(int(patch_radius), 1)
  u0 = max(du_i - r, 0)
  u1 = min(du_i + r + 1, cw)
  v0 = max(dv_i - r, 0)
  v1 = min(dv_i + r + 1, ch)
  patch = cloud_xyz[v0:v1, u0:u1, :]
  flat = patch.reshape(-1, 3)
  z = flat[:, 2]
  valid = np.isfinite(flat).all(axis=1) & np.isfinite(z) & (z > min_depth) & (z < max_depth)
  if not np.any(valid):
    return None
  cand = flat[valid]
  zc = cand[:, 2]
  mode = sample_mode.strip().lower()
  if mode == 'center':
    center = cloud_xyz[dv_i, du_i]
    if (
      np.isfinite(center).all()
      and min_depth < float(center[2]) < max_depth
    ):
      return float(center[0]), float(center[1]), float(center[2])
    mode = 'closest'
  if mode == 'median':
    idx = len(zc) // 2
    pick = cand[np.argsort(zc)[idx]]
  else:
    pick = cand[np.argmin(zc)]
  return float(pick[0]), float(pick[1]), float(pick[2])


def point_cloud_to_depth_m(msg: PointCloud2) -> Optional[np.ndarray]:
  """Build a depth map (H,W) from ZED organized cloud_registered (z in meters)."""
  if msg.height < 2 or msg.width < 2:
    return None
  try:
    from sensor_msgs_py import point_cloud2 as pc2
    z = pc2.read_points_numpy(msg, field_names=('z',), skip_nans=False)
    depth = z.reshape(msg.height, msg.width).astype(np.float32)
    depth[~np.isfinite(depth)] = np.nan
    return depth
  except Exception:
    return None


def intrinsics_from_camera_info(
  camera_info: CameraInfo,
  image_w: Optional[int] = None,
  image_h: Optional[int] = None,
) -> Tuple[float, float, float, float]:
  """Rectified RGB: prefer CameraInfo.P; fall back to K."""
  if len(camera_info.p) >= 12 and abs(float(camera_info.p[0])) > 1e-6:
    fx = float(camera_info.p[0])
    fy = float(camera_info.p[5])
    cx = float(camera_info.p[2])
    cy = float(camera_info.p[6])
  else:
    fx = float(camera_info.k[0])
    fy = float(camera_info.k[4])
    cx = float(camera_info.k[2])
    cy = float(camera_info.k[5])
  info_w = int(camera_info.width) if camera_info.width else 0
  info_h = int(camera_info.height) if camera_info.height else 0
  if image_w and image_h and info_w > 0 and info_h > 0:
    if info_w != image_w or info_h != image_h:
      sx = image_w / float(info_w)
      sy = image_h / float(info_h)
      fx *= sx
      fy *= sy
      cx *= sx
      cy *= sy
  return fx, fy, cx, cy


def depth_reading_to_z(
  depth_reading: float,
  u: float,
  v: float,
  fx: float,
  fy: float,
  cx: float,
  cy: float,
  is_radial_range: bool,
) -> float:
  """If depth map stores radial range R, convert to optical-axis Z."""
  if not is_radial_range or depth_reading <= 0.0:
    return depth_reading
  norm = math.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
  return depth_reading / max(norm, 1e-9)


def deproject_uv(
  u: float,
  v: float,
  depth_m: float,
  camera_info: CameraInfo,
  image_w: Optional[int] = None,
  image_h: Optional[int] = None,
  depth_is_radial: bool = False,
) -> Tuple[float, float, float]:
  """Deproject pixel (u,v) in the color/RGB image using camera_info intrinsics."""
  fx, fy, cx, cy = intrinsics_from_camera_info(camera_info, image_w, image_h)
  z = depth_reading_to_z(depth_m, u, v, fx, fy, cx, cy, depth_is_radial)
  x = (u - cx) * z / fx
  y = (v - cy) * z / fy
  return x, y, z


def offset_along_camera_ray(
  x: float,
  y: float,
  z: float,
  inset_m: float,
) -> Tuple[float, float, float]:
  """Move a camera-frame point deeper along the view ray (origin → surface click)."""
  inset = float(inset_m)
  if inset <= 0.0:
    return x, y, z
  norm = math.sqrt(x * x + y * y + z * z)
  if norm < 1e-9:
    return x, y, z
  scale = (norm + inset) / norm
  return x * scale, y * scale, z * scale


def project_camera_xyz_to_uv(
  x: float,
  y: float,
  z: float,
  camera_info: CameraInfo,
  image_w: int,
  image_h: int,
) -> Optional[Tuple[int, int]]:
  if z <= 1e-6:
    return None
  fx, fy, cx, cy = intrinsics_from_camera_info(camera_info, image_w, image_h)
  u = fx * x / z + cx
  v = fy * y / z + cy
  if not (0 <= u < image_w and 0 <= v < image_h):
    return None
  return int(round(u)), int(round(v))


def _depth_scale(color_w: int, color_h: int, depth_w: int, depth_h: int) -> Tuple[float, float]:
  return depth_w / float(color_w), depth_h / float(color_h)


def color_uv_from_depth_uv(
  u_depth: float,
  v_depth: float,
  color_w: int,
  color_h: int,
  depth_w: int,
  depth_h: int,
) -> Tuple[float, float]:
  scale_u, scale_v = _depth_scale(color_w, color_h, depth_w, depth_h)
  return u_depth / scale_u, v_depth / scale_v


def depth_uv_from_color_uv(
  u_color: float,
  v_color: float,
  color_w: int,
  color_h: int,
  depth_w: int,
  depth_h: int,
) -> Tuple[float, float]:
  scale_u, scale_v = _depth_scale(color_w, color_h, depth_w, depth_h)
  return u_color * scale_u, v_color * scale_v


def sample_depth_in_mask(
  depth_m: np.ndarray,
  mask: np.ndarray,
  min_depth: float,
  max_depth: float,
  min_pixels: int,
  sample_mode: str = 'closest',
) -> Optional[Tuple[float, float, float]]:
  """Return (u, v, depth) from one consistent pixel inside the mask.

  Older code took median(depth) and median(u/v) separately, which shifts small
  objects (e.g. flowers) when the mask spans mixed depths (petal + leaf behind).
  """
  ys, xs = np.where(mask)
  if ys.size == 0:
    return None
  vals = depth_m[ys, xs]
  valid = np.isfinite(vals) & (vals > min_depth) & (vals < max_depth)
  if int(valid.sum()) < min_pixels:
    return None
  vy = ys[valid]
  vx = xs[valid]
  vz = vals[valid]
  mode = sample_mode.strip().lower()
  if mode == 'median':
    pick = int(np.argsort(vz)[len(vz) // 2])
  else:
    pick = int(np.argmin(vz))
  return float(vx[pick]), float(vy[pick]), float(vz[pick])


def median_depth_in_mask(
  depth_m: np.ndarray,
  mask: np.ndarray,
  min_depth: float,
  max_depth: float,
  min_pixels: int,
) -> Optional[Tuple[float, float, float]]:
  return sample_depth_in_mask(
    depth_m, mask, min_depth, max_depth, min_pixels, sample_mode='median',
  )


def detections_3d_from_segmentation(
  result,
  names: Dict[int, str],
  depth_m: np.ndarray,
  camera_info: CameraInfo,
  color_h: int,
  color_w: int,
  min_depth: float,
  max_depth: float,
  min_mask_pixels: int,
  depth_sample_mode: str = 'closest',
) -> List[Detection3D]:
  detections: List[Detection3D] = []
  if result.masks is None or result.boxes is None:
    return detections

  depth_h, depth_w = depth_m.shape[:2]
  scale_u = depth_w / float(color_w)
  scale_v = depth_h / float(color_h)
  masks = result.masks.data.cpu().numpy()
  boxes_conf = result.boxes.conf.cpu().numpy()
  boxes_cls = result.boxes.cls.cpu().numpy().astype(int)

  for i in range(len(masks)):
    mask_color = mask_bool_at_image_size(masks[i], color_h, color_w)
    if scale_u != 1.0 or scale_v != 1.0:
      mask_depth = cv2.resize(
        mask_color.astype(np.uint8),
        (depth_w, depth_h),
        interpolation=cv2.INTER_NEAREST,
      ) > 0
    else:
      mask_depth = mask_color

    sampled = sample_depth_in_mask(
      depth_m, mask_depth, min_depth, max_depth, min_mask_pixels,
      sample_mode=depth_sample_mode,
    )
    if sampled is None:
      continue
    u_depth, v_depth, depth_val = sampled
    u_color, v_color = color_uv_from_depth_uv(
      u_depth, v_depth, color_w, color_h, depth_w, depth_h,
    )
    x, y, z = deproject_uv(u_color, v_color, depth_val, camera_info, color_w, color_h)
    cls_id = int(boxes_cls[i])
    detections.append(
      Detection3D(
        label=names.get(cls_id, str(cls_id)),
        cls_id=cls_id,
        conf=float(boxes_conf[i]),
        u=int(u_color),
        v=int(v_color),
        x=x,
        y=y,
        z=z,
      )
    )
  return detections


def detections_3d_from_detection(
  result,
  names: Dict[int, str],
  depth_m: np.ndarray,
  camera_info: CameraInfo,
  color_h: int,
  color_w: int,
  min_depth: float,
  max_depth: float,
) -> List[Detection3D]:
  detections: List[Detection3D] = []
  if result.boxes is None:
    return detections

  depth_h, depth_w = depth_m.shape[:2]
  scale_u = depth_w / float(color_w)
  scale_v = depth_h / float(color_h)
  boxes_xyxy = result.boxes.xyxy.cpu().numpy()
  boxes_conf = result.boxes.conf.cpu().numpy()
  boxes_cls = result.boxes.cls.cpu().numpy().astype(int)

  for i in range(len(boxes_xyxy)):
    x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
    u_color = int(0.5 * (x1 + x2))
    v_color = int(0.5 * (y1 + y2))
    u_depth, v_depth = depth_uv_from_color_uv(
      float(u_color), float(v_color), color_w, color_h, depth_w, depth_h,
    )
    u = int(round(u_depth))
    v = int(round(v_depth))
    if not (0 <= u < depth_w and 0 <= v < depth_h):
      continue
    depth_val = float(depth_m[v, u])
    if not np.isfinite(depth_val) or depth_val <= min_depth or depth_val >= max_depth:
      continue
    x, y, z = deproject_uv(
      float(u_color), float(v_color), depth_val, camera_info, color_w, color_h,
    )
    cls_id = int(boxes_cls[i])
    detections.append(
      Detection3D(
        label=names.get(cls_id, str(cls_id)),
        cls_id=cls_id,
        conf=float(boxes_conf[i]),
        u=u_color,
        v=v_color,
        x=x,
        y=y,
        z=z,
      )
    )
  return detections


def detections_to_markers(
  stamp,
  frame_id: str,
  detections: Sequence[Detection3D],
  marker_id_base: int,
) -> List[Marker]:
  markers: List[Marker] = []
  for i, det in enumerate(detections):
    color = color_for_class(det.cls_id)
    norm_color = ColorRGBA(
      r=color[2] / 255.0,
      g=color[1] / 255.0,
      b=color[0] / 255.0,
      a=0.9,
    )
    sphere = Marker()
    sphere.header = Header(stamp=stamp, frame_id=frame_id)
    sphere.ns = 'perception_object'
    sphere.id = marker_id_base + i
    sphere.type = Marker.SPHERE
    sphere.action = Marker.ADD
    sphere.pose.position = Point(x=det.x, y=det.y, z=det.z)
    sphere.pose.orientation = Quaternion(w=1.0)
    sphere.scale = Vector3(x=0.03, y=0.03, z=0.03)
    sphere.color = norm_color
    markers.append(sphere)

    text = Marker()
    text.header = sphere.header
    text.ns = 'perception_object_label'
    text.id = marker_id_base + 10000 + i
    text.type = Marker.TEXT_VIEW_FACING
    text.action = Marker.ADD
    text.pose.position = Point(x=det.x, y=det.y, z=det.z + 0.04)
    text.pose.orientation = Quaternion(w=1.0)
    text.scale.z = 0.03
    text.color = norm_color
    text.text = f'{det.label} ({det.conf:.2f})'
    markers.append(text)
  return markers


def detections_to_transforms(
  stamp,
  parent_frame: str,
  detections: Sequence[Detection3D],
) -> List[TransformStamped]:
  transforms: List[TransformStamped] = []
  for i, det in enumerate(detections):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = sanitize_tf_child_name(det.label, i)
    t.transform.translation.x = det.x
    t.transform.translation.y = det.y
    t.transform.translation.z = det.z
    t.transform.rotation.w = 1.0
    transforms.append(t)
  return transforms


@dataclass
class ManualPoint3D:
  kind: str  # 'target' | 'obstacle'
  point_id: int
  u: int
  v: int
  x: float
  y: float
  z: float

  @property
  def child_frame(self) -> str:
    return f'perception/{self.kind}_{self.point_id}'

  @property
  def label(self) -> str:
    return self.kind


def sample_depth_at_uv(
  u: int,
  v: int,
  depth_m: np.ndarray,
  color_h: int,
  color_w: int,
  camera_info: CameraInfo,
  min_depth: float,
  max_depth: float,
  patch_radius: int = 5,
  sample_mode: str = 'closest',
  depth_scale: float = 1.0,
  depth_offset_m: float = 0.0,
  depth_is_radial: bool = False,
) -> Optional[Tuple[float, float, float]]:
  depth_h, depth_w = depth_m.shape[:2]
  du, dv = depth_uv_from_color_uv(
    float(u), float(v), color_w, color_h, depth_w, depth_h,
  )
  du_i = int(round(du))
  dv_i = int(round(dv))
  r = max(int(patch_radius), 1)
  u0 = max(du_i - r, 0)
  u1 = min(du_i + r + 1, depth_w)
  v0 = max(dv_i - r, 0)
  v1 = min(dv_i + r + 1, depth_h)
  patch = depth_m[v0:v1, u0:u1]
  valid_mask = np.isfinite(patch) & (patch > min_depth) & (patch < max_depth)
  if not np.any(valid_mask):
    return None

  center_val = float('nan')
  if 0 <= du_i < depth_w and 0 <= dv_i < depth_h:
    center_val = float(depth_m[dv_i, du_i])

  mode = sample_mode.strip().lower()
  pick_du = float(du)
  pick_dv = float(dv)
  if mode == 'center':
    if np.isfinite(center_val) and min_depth < center_val < max_depth:
      depth_raw = center_val
    else:
      flat_idx = int(np.argmin(np.where(valid_mask.ravel(), patch.ravel(), np.inf)))
      local_v, local_u = np.unravel_index(flat_idx, patch.shape)
      pick_du = float(u0 + local_u)
      pick_dv = float(v0 + local_v)
      depth_raw = float(patch[local_v, local_u])
  elif mode == 'median':
    flat_vals = patch[valid_mask]
    depth_raw = float(np.median(flat_vals))
    median_idx = int(np.argmin(np.abs(flat_vals - depth_raw)))
    flat_idx = int(np.flatnonzero(valid_mask.ravel())[median_idx])
    local_v, local_u = np.unravel_index(flat_idx, patch.shape)
    pick_du = float(u0 + local_u)
    pick_dv = float(v0 + local_v)
  else:
    # closest — foreground; deproject at the actual closest pixel, not click UV
    flat_idx = int(np.argmin(np.where(valid_mask.ravel(), patch.ravel(), np.inf)))
    local_v, local_u = np.unravel_index(flat_idx, patch.shape)
    pick_du = float(u0 + local_u)
    pick_dv = float(v0 + local_v)
    depth_raw = float(patch[local_v, local_u])

  depth_val = depth_raw * float(depth_scale) + float(depth_offset_m)
  pick_u, pick_v = color_uv_from_depth_uv(
    pick_du, pick_dv, color_w, color_h, depth_w, depth_h,
  )
  x, y, z = deproject_uv(
    pick_u, pick_v, depth_val, camera_info, color_w, color_h, depth_is_radial,
  )
  return x, y, z


def manual_points_to_markers(
  stamp,
  frame_id: str,
  points: Sequence[ManualPoint3D],
) -> List[Marker]:
  markers: List[Marker] = []
  for i, pt in enumerate(points):
    if pt.kind == 'target':
      color = (0, 255, 0)
    else:
      color = (0, 0, 255)
    norm_color = ColorRGBA(
      r=color[2] / 255.0,
      g=color[1] / 255.0,
      b=color[0] / 255.0,
      a=0.95,
    )
    sphere = Marker()
    sphere.header = Header(stamp=stamp, frame_id=frame_id)
    sphere.ns = f'perception_{pt.kind}'
    sphere.id = i
    sphere.type = Marker.SPHERE
    sphere.action = Marker.ADD
    sphere.pose.position = Point(x=pt.x, y=pt.y, z=pt.z)
    sphere.pose.orientation = Quaternion(w=1.0)
    scale = 0.04 if pt.kind == 'target' else 0.05
    sphere.scale = Vector3(x=scale, y=scale, z=scale)
    sphere.color = norm_color
    markers.append(sphere)

    text = Marker()
    text.header = sphere.header
    text.ns = f'perception_{pt.kind}_label'
    text.id = i + 10000
    text.type = Marker.TEXT_VIEW_FACING
    text.action = Marker.ADD
    text.pose.position = Point(x=pt.x, y=pt.y, z=pt.z + 0.05)
    text.pose.orientation = Quaternion(w=1.0)
    text.scale.z = 0.035
    text.color = norm_color
    text.text = f'{pt.kind}_{pt.point_id}'
    markers.append(text)
  return markers


def manual_points_to_transforms(
  stamp,
  parent_frame: str,
  points: Sequence[ManualPoint3D],
) -> List[TransformStamped]:
  transforms: List[TransformStamped] = []
  for pt in points:
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = pt.child_frame
    t.transform.translation.x = pt.x
    t.transform.translation.y = pt.y
    t.transform.translation.z = pt.z
    t.transform.rotation.w = 1.0
    transforms.append(t)
  return transforms
