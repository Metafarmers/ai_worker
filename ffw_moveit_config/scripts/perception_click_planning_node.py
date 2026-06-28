#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Click on ZED image → planning frame (base_link):
#   mode 1: left target  → /target_pose
#   mode 4: right target → /target_pose_r  (dual-arm cuRobo needs both)
#   mode 2: obstacle → CUBE MarkerArray on /obstacle_markers (cuRobo cuboid)
#   mode 3: delete nearest click
#
# Run (with curobo_pathplanning_node + ZED):
#   ./scripts/run_perception_click_planning.sh
#
# Keys: 1=left target, 4=right target, 2=obstacle, 3=delete, c=clear all, q=quit

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

from curobo_obstacle_utils import (
  TARGET_BACK_WALL_CUROBO_COLLISION_X,
  TARGET_BACK_WALL_ID,
  append_ee_axes_markers,
  approach_position_from_contact,
  quat_toward_robot_y_up_arm,
  target_back_wall_center_x,
)
from perception_3d_utils import (
  depth_image_to_meters,
  depth_uv_from_color_uv,
  offset_along_camera_ray,
  project_camera_xyz_to_uv,
  sample_depth_at_uv,
)

ClickMode = Literal['target_l', 'target_r', 'obstacle', 'delete']
MODE_COLORS = {
  'target_l': (0, 255, 0),
  'target_r': (0, 220, 255),
  'obstacle': (0, 0, 255),
  'delete': (0, 165, 255),
}

OBSTACLE_MARKER_NS = 'obstacle_cuboid'
TARGET_MARKER_NS = 'click_target'
TARGET_MARKER_ID_AXES_L = 1
TARGET_MARKER_ID_SPHERE_L = 2
TARGET_MARKER_ID_AXES_R = 3
TARGET_MARKER_ID_SPHERE_R = 4
TARGET_MARKER_ID_CONTACT_L = 5
TARGET_MARKER_ID_CONTACT_R = 6


def _quat_from_yaw(yaw: float):
  from geometry_msgs.msg import Quaternion
  half = yaw * 0.5
  q = Quaternion()
  q.x = 0.0
  q.y = 0.0
  q.z = math.sin(half)
  q.w = math.cos(half)
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
    self.declare_parameter('target_pose_r_topic', '/target_pose_r')
    self.declare_parameter('target_pose_contact_topic', '/target_pose_contact')
    self.declare_parameter('target_pose_contact_r_topic', '/target_pose_contact_r')
    self.declare_parameter('enable_tool_approach_offset', True)
    self.declare_parameter('approach_retreat_tool_m', 0.08)
    self.declare_parameter('approach_tool_axis', 'z')
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
    self.declare_parameter('right_arm_roll_rad', math.pi)
    self.declare_parameter('cuboid_size_x', 0.02)
    self.declare_parameter('cuboid_size_y', 0.02)
    self.declare_parameter('cuboid_size_z', 0.02)
    self.declare_parameter('obstacle_dim_scale', 0.6)
    self.declare_parameter('obstacle_yaw', 0.0)
    self.declare_parameter('target_arrow_length', 0.12)
    self.declare_parameter('target_sphere_diameter', 0.06)
    self.declare_parameter(
      'target_view_inset_m',
      0.01,
    )
    self.declare_parameter('enable_target_back_wall', True)
    self.declare_parameter('target_back_wall_offset_m', 0.07)
    self.declare_parameter('target_back_wall_size_x', 0.08)
    self.declare_parameter('target_back_wall_size_y', 2.0)
    self.declare_parameter('target_back_wall_size_z', 1.5)
    self.declare_parameter('target_back_wall_center_y', 0.0)
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
    self._obstacle_dim_scale = max(
      float(self.get_parameter('obstacle_dim_scale').value), 0.1
    )
    self._obstacle_yaw = float(self.get_parameter('obstacle_yaw').value)
    self._publish_target_markers = bool(self.get_parameter('publish_target_markers').value)
    self._target_view_inset_m = max(
      float(self.get_parameter('target_view_inset_m').value), 0.0
    )
    self._enable_target_back_wall = bool(self.get_parameter('enable_target_back_wall').value)
    self._target_back_wall_offset_m = float(self.get_parameter('target_back_wall_offset_m').value)
    self._target_back_wall_size = (
      max(float(self.get_parameter('target_back_wall_size_x').value), 0.01),
      max(float(self.get_parameter('target_back_wall_size_y').value), 0.01),
      max(float(self.get_parameter('target_back_wall_size_z').value), 0.01),
    )
    self._target_back_wall_center_y = float(
      self.get_parameter('target_back_wall_center_y').value
    )
    self._enable_tool_approach_offset = bool(
      self.get_parameter('enable_tool_approach_offset').value
    )
    self._approach_retreat_tool_m = max(
      float(self.get_parameter('approach_retreat_tool_m').value), 0.0
    )
    self._approach_tool_axis = str(self.get_parameter('approach_tool_axis').value).strip().lower()

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

    self._mode: ClickMode = 'target_l'
    self._pending_click: Optional[_PendingClick] = None
    self._obstacles: List[PlanningClickPoint] = []
    self._target_l: Optional[PlanningClickPoint] = None
    self._target_r: Optional[PlanningClickPoint] = None
    self._next_obstacle_id = 0
    self._logged_target_key_l: Optional[tuple[float, float, float]] = None
    self._logged_target_key_r: Optional[tuple[float, float, float]] = None

    qos = QoSProfile(
      depth=10,
      durability=DurabilityPolicy.VOLATILE,
      reliability=ReliabilityPolicy.RELIABLE,
    )
    # Latched planning outputs so curobo_pathplanning_node receives last click after late start.
    planning_qos = QoSProfile(
      depth=10,
      durability=DurabilityPolicy.TRANSIENT_LOCAL,
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
    target_r_topic = str(self.get_parameter('target_pose_r_topic').value)
    contact_topic = str(self.get_parameter('target_pose_contact_topic').value)
    contact_r_topic = str(self.get_parameter('target_pose_contact_r_topic').value)

    self.create_subscription(Image, image_topic, self._on_image, sensor_qos)
    self.create_subscription(Image, image_topic, self._on_image, reliable_qos)
    self.create_subscription(Image, depth_topic, self._on_depth, reliable_qos)
    self.create_subscription(CameraInfo, depth_info_topic, self._on_camera_info, camera_info_qos)
    self.create_subscription(CameraInfo, depth_info_topic, self._on_camera_info, reliable_qos)

    self._obstacle_pub = self.create_publisher(MarkerArray, obstacle_topic, planning_qos)
    self._target_pose_pub = self.create_publisher(PoseStamped, target_topic, planning_qos)
    self._target_pose_r_pub = self.create_publisher(PoseStamped, target_r_topic, planning_qos)
    self._target_contact_pub = self.create_publisher(PoseStamped, contact_topic, planning_qos)
    self._target_contact_r_pub = self.create_publisher(
      PoseStamped, contact_r_topic, planning_qos
    )
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
    self.get_logger().info(f'Left target  PoseStamped → {target_topic}')
    self.get_logger().info(f'Right target PoseStamped → {target_r_topic}')
    self.get_logger().info(f'Left contact  PoseStamped → {contact_topic}')
    self.get_logger().info(f'Right contact PoseStamped → {contact_r_topic}')
    if self._enable_tool_approach_offset:
      self.get_logger().info(
        f'Tool approach: cuRobo → pre-grasp +{self._approach_retreat_tool_m:.3f}m '
        f'EE +{self._approach_tool_axis.upper()}; MoveIt final → contact '
        f'along EE −{self._approach_tool_axis.upper()} when ≤15cm'
      )
    self.get_logger().info(
      'Keys: 1=left target, 4=right target, 2=obstacle, 3=delete, c=clear, q=quit'
    )

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
      self._target_l = None
      self._target_r = None
      self._next_obstacle_id = 0
      self._logged_target_key_l = None
      self._logged_target_key_r = None
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

  def _goal_quat_for_position(
    self, xyz: Tuple[float, float, float], arm: Literal['l', 'r'] = 'l'
  ):
    return quat_toward_robot_y_up_arm(
      xyz,
      robot_base_xy=(self._robot_base_x, self._robot_base_y),
      arm=arm,
      right_roll_rad=float(self.get_parameter('right_arm_roll_rad').value),
    )

  def _make_contact_pose_msg(
    self, pt: PlanningClickPoint, stamp, arm: Literal['l', 'r'] = 'l'
  ) -> PoseStamped:
    q = self._goal_quat_for_position((pt.x, pt.y, pt.z), arm=arm)
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

  def _make_approach_pose_msg(
    self, pt: PlanningClickPoint, stamp, arm: Literal['l', 'r'] = 'l'
  ) -> PoseStamped:
    contact = (pt.x, pt.y, pt.z)
    q = self._goal_quat_for_position(contact, arm=arm)
    if self._enable_tool_approach_offset and self._approach_retreat_tool_m > 0.0:
      ax, ay, az = approach_position_from_contact(
        contact,
        q,
        self._approach_retreat_tool_m,
        axis=self._approach_tool_axis,
      )
    else:
      ax, ay, az = contact
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = self._planning_frame
    msg.pose.position.x = ax
    msg.pose.position.y = ay
    msg.pose.position.z = az
    msg.pose.orientation.x = float(q.x)
    msg.pose.orientation.y = float(q.y)
    msg.pose.orientation.z = float(q.z)
    msg.pose.orientation.w = float(q.w)
    return msg

  def _make_target_pose_msg(
    self, pt: PlanningClickPoint, stamp, arm: Literal['l', 'r'] = 'l'
  ) -> PoseStamped:
    """cuRobo / planning goal (pre-grasp with tool offset applied)."""
    return self._make_approach_pose_msg(pt, stamp, arm=arm)

  def _make_target_markers(
    self, pt: PlanningClickPoint, stamp, arm: Literal['l', 'r']
  ) -> MarkerArray:
    approach_msg = self._make_approach_pose_msg(pt, stamp, arm=arm)
    contact_msg = self._make_contact_pose_msg(pt, stamp, arm=arm)
    arrow_len = float(self.get_parameter('target_arrow_length').value)
    sphere_d = float(self.get_parameter('target_sphere_diameter').value)
    axes_id = TARGET_MARKER_ID_AXES_L if arm == 'l' else TARGET_MARKER_ID_AXES_R
    sphere_id = TARGET_MARKER_ID_SPHERE_L if arm == 'l' else TARGET_MARKER_ID_SPHERE_R
    contact_id = TARGET_MARKER_ID_CONTACT_L if arm == 'l' else TARGET_MARKER_ID_CONTACT_R
    color = (
      ColorRGBA(r=0.1, g=0.85, b=0.2, a=0.85)
      if arm == 'l'
      else ColorRGBA(r=0.1, g=0.75, b=1.0, a=0.85)
    )
    contact_color = (
      ColorRGBA(r=0.95, g=0.95, b=0.2, a=0.9)
      if arm == 'l'
      else ColorRGBA(r=0.95, g=0.85, b=0.2, a=0.9)
    )

    arr = MarkerArray()
    append_ee_axes_markers(
      arr,
      stamp,
      self._planning_frame,
      approach_msg.pose,
      ns=TARGET_MARKER_NS,
      marker_id=axes_id,
      axis_length=max(arrow_len, 0.02),
    )

    approach_sphere = Marker()
    approach_sphere.header = Header(stamp=stamp, frame_id=self._planning_frame)
    approach_sphere.ns = TARGET_MARKER_NS
    approach_sphere.id = sphere_id
    approach_sphere.type = Marker.SPHERE
    approach_sphere.action = Marker.ADD
    approach_sphere.pose = approach_msg.pose
    approach_sphere.scale.x = sphere_d
    approach_sphere.scale.y = sphere_d
    approach_sphere.scale.z = sphere_d
    approach_sphere.color = color
    arr.markers.append(approach_sphere)

    contact_sphere = Marker()
    contact_sphere.header = Header(stamp=stamp, frame_id=self._planning_frame)
    contact_sphere.ns = TARGET_MARKER_NS
    contact_sphere.id = contact_id
    contact_sphere.type = Marker.SPHERE
    contact_sphere.action = Marker.ADD
    contact_sphere.pose = contact_msg.pose
    contact_sphere.scale.x = sphere_d * 0.65
    contact_sphere.scale.y = sphere_d * 0.65
    contact_sphere.scale.z = sphere_d * 0.65
    contact_sphere.color = contact_color
    arr.markers.append(contact_sphere)
    return arr

  def _wall_near_face_x(self, target_x: float) -> float:
    """YZ-parallel slab face just past target (away from robot base along +X)."""
    offset = self._target_back_wall_offset_m
    if target_x >= self._robot_base_x:
      return target_x + offset
    return target_x - offset

  def _append_target_back_wall(
    self,
    arr: MarkerArray,
    stamp,
    target: PlanningClickPoint,
    wall_id: int,
  ) -> None:
    # Full YZ slab at X behind left target (both arms must not cross this plane).
    wall = Marker()
    wall.header = Header(stamp=stamp, frame_id=self._planning_frame)
    wall.ns = OBSTACLE_MARKER_NS
    wall.id = wall_id
    wall.type = Marker.CUBE
    wall.action = Marker.ADD
    # Center cuboid so the near face is at target+offset (target stays outside the wall).
    wall.pose.position.x = target_back_wall_center_x(
      self._wall_near_face_x(target.x),
    )
    wall.pose.position.y = self._target_back_wall_center_y
    wall.pose.position.z = target.z
    wall.pose.orientation.w = 1.0
    wall.scale.x = TARGET_BACK_WALL_CUROBO_COLLISION_X
    wall.scale.y = self._target_back_wall_size[1]
    wall.scale.z = self._target_back_wall_size[2]
    wall.color = ColorRGBA(r=1.0, g=0.15, b=0.15, a=0.45)
    arr.markers.append(wall)

  def _back_wall_anchor_target(
    self,
    target_l: Optional[PlanningClickPoint],
    target_r: Optional[PlanningClickPoint],
  ) -> Optional[PlanningClickPoint]:
    """Pick the target farthest along +X so the wall sits behind every goal."""
    candidates = [t for t in (target_l, target_r) if t is not None]
    if not candidates:
      return None
    if len(candidates) == 1:
      return candidates[0]
    return max(candidates, key=lambda t: float(t.x))

  def _build_obstacle_markers(self, stamp) -> MarkerArray:
    with self._state_lock:
      obstacles = list(self._obstacles)
      target_l = self._target_l
      target_r = self._target_r

    arr = MarkerArray()
    delete_all = Marker()
    delete_all.action = Marker.DELETEALL
    arr.markers.append(delete_all)

    sx, sy, sz = self._cuboid_size
    obs_scale = self._obstacle_dim_scale
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
      m.scale.x = sx * obs_scale
      m.scale.y = sy * obs_scale
      m.scale.z = sz * obs_scale
      m.color = ColorRGBA(r=1.0, g=0.4, b=0.1, a=0.85)
      arr.markers.append(m)

    # Virtual wall: YZ slab behind the deepest (+X) target; both arms must avoid it in cuRobo.
    wall_anchor = self._back_wall_anchor_target(target_l, target_r)
    if self._enable_target_back_wall and wall_anchor is not None:
      self._append_target_back_wall(arr, stamp, wall_anchor, TARGET_BACK_WALL_ID)
    return arr

  def _publish_obstacles(self) -> None:
    stamp = self.get_clock().now().to_msg()
    self._obstacle_pub.publish(self._build_obstacle_markers(stamp))

  def _publish_arm_target(self, arm: Literal['l', 'r']) -> None:
    with self._state_lock:
      pt = self._target_l if arm == 'l' else self._target_r
    if pt is None:
      return
    stamp = self.get_clock().now().to_msg()
    approach_msg = self._make_approach_pose_msg(pt, stamp, arm=arm)
    contact_msg = self._make_contact_pose_msg(pt, stamp, arm=arm)
    if arm == 'l':
      self._target_pose_pub.publish(approach_msg)
      self._target_contact_pub.publish(contact_msg)
      target_topic = str(self.get_parameter('target_pose_topic').value)
      contact_topic = str(self.get_parameter('target_pose_contact_topic').value)
    else:
      self._target_pose_r_pub.publish(approach_msg)
      self._target_contact_r_pub.publish(contact_msg)
      target_topic = str(self.get_parameter('target_pose_r_topic').value)
      contact_topic = str(self.get_parameter('target_pose_contact_r_topic').value)
    if self._target_marker_pub is not None:
      self._target_marker_pub.publish(self._make_target_markers(pt, stamp, arm))

    key = (round(pt.x, 4), round(pt.y, 4), round(pt.z, 4))
    last_key = self._logged_target_key_l if arm == 'l' else self._logged_target_key_r
    if key == last_key:
      return
    if arm == 'l':
      self._logged_target_key_l = key
    else:
      self._logged_target_key_r = key
    arm_tag = 'LEFT' if arm == 'l' else 'RIGHT'
    ap = approach_msg.pose.position
    cp = contact_msg.pose.position
    self.get_logger().info(
      f'[Perception/{arm_tag}] cuRobo → {target_topic} '
      f'pre-grasp=({ap.x:.3f},{ap.y:.3f},{ap.z:.3f})'
    )
    self.get_logger().info(
      f'[Perception/{arm_tag}] MoveIt → {contact_topic} '
      f'contact=({cp.x:.3f},{cp.y:.3f},{cp.z:.3f})'
    )

  def _publish_targets(self) -> None:
    self._publish_arm_target('l')
    self._publish_arm_target('r')

  def _republish_planning_outputs(self) -> None:
    self._publish_obstacles()
    self._publish_targets()

  def _delete_nearest(self, u: int, v: int) -> None:
    with self._state_lock:
      candidates: List[Tuple[float, PlanningClickPoint, str]] = []
      if self._target_l is not None:
        d = math.hypot(self._target_l.u - u, self._target_l.v - v)
        candidates.append((d, self._target_l, 'target_l'))
      if self._target_r is not None:
        d = math.hypot(self._target_r.u - u, self._target_r.v - v)
        candidates.append((d, self._target_r, 'target_r'))
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

      if kind == 'target_l':
        self._target_l = None
        self.get_logger().info('Deleted left target')
      elif kind == 'target_r':
        self._target_r = None
        self.get_logger().info('Deleted right target')
      else:
        self._obstacles = [o for o in self._obstacles if o.point_id != pt.point_id]
        self.get_logger().info(f'Deleted obstacle_{pt.point_id}')

    self._publish_obstacles()
    self._publish_targets()

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
    surface_cam = (cx, cy, cz)
    if mode in ('target_l', 'target_r') and self._target_view_inset_m > 0.0:
      cx, cy, cz = offset_along_camera_ray(cx, cy, cz, self._target_view_inset_m)
    xyz_plan = self._transform_camera_xyz_to_planning(cx, cy, cz, camera_frame, stamp)
    if xyz_plan is None:
      return
    x, y, z = xyz_plan

    inset_note = ''
    if mode in ('target_l', 'target_r') and self._target_view_inset_m > 0.0:
      inset_note = f', view_inset={self._target_view_inset_m:.3f}m into object'

    self.get_logger().info(
      f'Click ({click.u},{click.v}): depth_used={surface_cam[2]:.3f}m '
      f'(center={center_d:.3f} closest={float(np.min(valid)):.3f} '
      f'median={float(np.median(valid)):.3f}) '
      f'surface_cam=({surface_cam[0]:.3f},{surface_cam[1]:.3f},{surface_cam[2]:.3f}) '
      f'→ target {self._planning_frame}=({x:.3f},{y:.3f},{z:.3f}){inset_note}'
    )

    if mode in ('target_l', 'target_r'):
      arm: Literal['l', 'r'] = 'l' if mode == 'target_l' else 'r'
      pt = PlanningClickPoint(
        kind='target', point_id=0, u=click.u, v=click.v, x=x, y=y, z=z,
      )
      with self._state_lock:
        if arm == 'l':
          self._target_l = pt
        else:
          self._target_r = pt
      label = 'LEFT' if arm == 'l' else 'RIGHT'
      self.get_logger().info(
        f'[Perception/{label}] click contact=({x:.3f},{y:.3f},{z:.3f}) — '
        f'cuRobo pre-grasp +{self._approach_retreat_tool_m:.3f}m EE+'
        f'{self._approach_tool_axis.upper()}, MoveIt final → contact'
        if self._enable_tool_approach_offset and self._approach_retreat_tool_m > 0.0
        else f'[Perception/{label}] click contact=({x:.3f},{y:.3f},{z:.3f})'
      )
      self._publish_obstacles()
      self._publish_arm_target(arm)
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
      target_l = self._target_l
      target_r = self._target_r
      obstacles = list(self._obstacles)

    if target_l is not None:
      cv2.circle(out, (target_l.u, target_l.v), 10, MODE_COLORS['target_l'], 2)
      cv2.putText(
        out, 'L', (target_l.u + 12, target_l.v - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MODE_COLORS['target_l'], 2, cv2.LINE_AA,
      )
    if target_r is not None:
      cv2.circle(out, (target_r.u, target_r.v), 10, MODE_COLORS['target_r'], 2)
      cv2.putText(
        out, 'R', (target_r.u + 12, target_r.v - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MODE_COLORS['target_r'], 2, cv2.LINE_AA,
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
      targets = [t for t in (target_l, target_r) if t is not None]
      for pt in targets + obstacles:
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
      f'Mode [1=L 4=R 2=obs 3=del]: {mode.upper()}',
      f'L:{"yes" if target_l else "no"} R:{"yes" if target_r else "no"} | Obs: {len(obstacles)}',
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
    if key in (ord('1'), ord('2'), ord('3'), ord('4')):
      self._set_mode({
        ord('1'): 'target_l',
        ord('4'): 'target_r',
        ord('2'): 'obstacle',
        ord('3'): 'delete',
      }[key])
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
