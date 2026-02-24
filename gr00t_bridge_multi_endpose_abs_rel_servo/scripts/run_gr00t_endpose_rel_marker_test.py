#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Model output visualization test — RViz Marker only (NO Servo, NO robot movement)
#
# rosbag으로 observation을 재생하면서 model inference 결과를
# RViz Marker로만 시각화하여 모델 출력이 올바른지 검증.
#
# Markers:
#   - 파란 구: /end_pose_left (frame: arm_l_link1)
#   - 빨간 구: /end_pose_right (frame: arm_r_link1)
#   - 파란 화살표 경로: model이 예측한 left arm trajectory (16 steps)
#   - 빨간 화살표 경로: model이 예측한 right arm trajectory (16 steps)
#   - 초록 원: target = current + delta (end_pose와 동일 프레임)
#
# 사용법:
#   # Terminal 1: rosbag 재생
#   ros2 bag play <bag_path> --clock
#
#   # Terminal 2: end_pose 발행
#   python3 inference_time_end_pose.py --rate 50 --use-sim-time
#
#   # Terminal 3: 이 스크립트 실행
#   python3 run_gr00t_endpose_rel_marker_test.py --use-sim-time
#
#   # Terminal 4: RViz에서 MarkerArray 토픽 /model_output_markers 추가

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

ISAAC_GROOT_PATH = Path("/workspace/Isaac-GR00T")
if ISAAC_GROOT_PATH.exists() and str(ISAAC_GROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_PATH))

GROOT_BRIDGE_PATH = Path(__file__).parent.parent.parent
if str(GROOT_BRIDGE_PATH) not in sys.path:
    sys.path.insert(0, str(GROOT_BRIDGE_PATH))


def quaternion_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def apply_relative_action(current_pose, delta_pose):
    target_pos = current_pose[:3] + delta_pose[:3]
    current_quat = current_pose[3:7]
    delta_quat = delta_pose[3:7]
    target_quat = quaternion_multiply(current_quat, delta_quat)
    quat_norm = np.linalg.norm(target_quat)
    if quat_norm > 1e-8:
        target_quat = target_quat / quat_norm
    else:
        target_quat = np.array([0.0, 0.0, 0.0, 1.0])
    return np.concatenate([target_pos, target_quat])


class ModelOutputMarkerNode(Node):
    """Model inference 결과를 RViz Marker로만 시각화하는 테스트 노드.

    Servo/로봇 명령 없음. rosbag + inference_time_end_pose.py 와 함께 사용.

    Published topics:
      /model_output_markers (MarkerArray): 모델 예측 trajectory 시각화
    """

    def __init__(self, args):
        super().__init__("model_output_marker_test")

        if args.use_sim_time:
            self.set_parameters([
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            ])
            self.get_logger().info("use_sim_time enabled")

        self.checkpoint_path = args.checkpoint
        self.embodiment_tag = args.embodiment
        self.device = args.device
        self.inference_rate = args.rate
        self.task_instruction = args.task
        self.action_horizon = args.action_horizon

        # State buffers
        self.latest_images: Dict[str, Optional[np.ndarray]] = {
            "ego_view": None,
            "cam_wrist_left": None,
            "cam_wrist_right": None,
        }
        self.latest_end_pose_left: Optional[PoseStamped] = None
        self.latest_end_pose_right: Optional[PoseStamped] = None
        self.latest_joint_state: Optional[JointState] = None

        self.inference_count = 0
        self.last_status_print = time.time()
        # 마커 중 inference 결과만 캐시 (end_pose 구는 매번 최신으로 갱신)
        self._cached_markers_except_end_pose: List[Marker] = []
        self._marker_publish_rate = args.marker_rate  # end_pose 구 갱신 Hz

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribers
        self.create_subscription(
            CompressedImage, args.ego_view_topic,
            lambda msg: self._image_cb(msg, "ego_view"), sensor_qos)
        self.create_subscription(
            CompressedImage, args.wrist_left_topic,
            lambda msg: self._image_cb(msg, "cam_wrist_left"), sensor_qos)
        self.create_subscription(
            CompressedImage, args.wrist_right_topic,
            lambda msg: self._image_cb(msg, "cam_wrist_right"), sensor_qos)
        self.create_subscription(
            PoseStamped, args.end_pose_left_topic,
            self._end_pose_left_cb, sensor_qos)
        self.create_subscription(
            PoseStamped, args.end_pose_right_topic,
            self._end_pose_right_cb, sensor_qos)
        self.create_subscription(
            JointState, args.joint_topic,
            self._joint_state_cb, sensor_qos)

        # Marker publisher
        self.marker_pub = self.create_publisher(MarkerArray, "/model_output_markers", 10)

        # Load model
        self._load_model()

        # Inference timer (궤적/타깃 등은 여기서만 갱신)
        self.timer = self.create_timer(1.0 / self.inference_rate, self._inference_cb)
        # end_pose 구만 높은 주기로 갱신 (늦게 따라가는 느낌 방지)
        self.marker_timer = self.create_timer(
            1.0 / self._marker_publish_rate, self._marker_update_cb
        )

        self.get_logger().info("=" * 60)
        self.get_logger().info("Model Output Marker Test Node (NO SERVO)")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Checkpoint: {self.checkpoint_path}")
        self.get_logger().info(f"  Rate: {self.inference_rate} Hz")
        self.get_logger().info(f"  Action horizon: {self.action_horizon}")
        self.get_logger().info(f"  Marker topic: /model_output_markers")
        self.get_logger().info("  Add MarkerArray in RViz to visualize")
        self.get_logger().info("=" * 60)

    def _load_model(self):
        from gr00t_bridge_multi_endpose_abs_rel_servo.gr00t_endpose_rel_servo_node import (
            Gr00tEndPoseRelInferenceWrapper,
        )
        self.model = Gr00tEndPoseRelInferenceWrapper(
            checkpoint_path=self.checkpoint_path,
            embodiment_tag=self.embodiment_tag,
            device=self.device,
        )
        self.get_logger().info("Model loaded!")

    # === Callbacks ===

    def _image_cb(self, msg, key):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is not None:
                self.latest_images[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"Image decode error ({key}): {e}")

    def _end_pose_left_cb(self, msg):
        self.latest_end_pose_left = msg

    def _end_pose_right_cb(self, msg):
        self.latest_end_pose_right = msg

    def _joint_state_cb(self, msg):
        self.latest_joint_state = msg

    # === Helpers ===

    def _extract_pose_array(self, pose_stamped):
        p = pose_stamped.pose
        return np.array([
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
        ])

    def _extract_state(self):
        end_pose_left = self._extract_pose_array(self.latest_end_pose_left)
        end_pose_right = self._extract_pose_array(self.latest_end_pose_right)
        jd = {n: p for n, p in zip(
            self.latest_joint_state.name, self.latest_joint_state.position)}
        return {
            "end_pose_left": end_pose_left,
            "gripper_l": np.array([jd.get("gripper_l_joint1", 0.0)]),
            "end_pose_right": end_pose_right,
            "gripper_r": np.array([jd.get("gripper_r_joint1", 0.0)]),
            "head": np.array([jd.get("head_joint1", 0.0), jd.get("head_joint2", 0.0)]),
        }

    def _check_ready(self):
        missing = [k for k, v in self.latest_images.items() if v is None]
        ready = (
            len(missing) == 0
            and self.latest_end_pose_left is not None
            and self.latest_end_pose_right is not None
            and self.latest_joint_state is not None
        )
        now = time.time()
        if not ready and now - self.last_status_print > 3.0:
            self.get_logger().info(
                f"Waiting... cameras_missing={missing} "
                f"end_pose_L={'OK' if self.latest_end_pose_left else 'NO'} "
                f"end_pose_R={'OK' if self.latest_end_pose_right else 'NO'} "
                f"joints={'OK' if self.latest_joint_state else 'NO'}"
            )
            self.last_status_print = now
        return ready

    # === Inference & Marker ===

    def _inference_cb(self):
        if not self._check_ready():
            return

        try:
            from gr00t_bridge_multi_endpose_abs_rel_servo.gr00t_endpose_rel_servo_node import (
                prepare_observation_endpose_rel,
            )

            state = self._extract_state()
            observation = prepare_observation_endpose_rel(
                self.latest_images.copy(), state, self.task_instruction, self.model)

            inference_start = time.time()
            action_dict = self.model.policy.get_action(observation)
            action = self.model._parse_action_output(action_dict)
            inference_time = time.time() - inference_start

            self.inference_count += 1

            # 현재 end pose
            current_left = self._extract_pose_array(self.latest_end_pose_left)
            current_right = self._extract_pose_array(self.latest_end_pose_right)
            frame_left = self.latest_end_pose_left.header.frame_id
            frame_right = self.latest_end_pose_right.header.frame_id

            # 16 step trajectory 계산
            traj_left = self._compute_trajectory(current_left, action.rel_end_pose_left)
            traj_right = self._compute_trajectory(current_right, action.rel_end_pose_right)

            # Marker 생성 및 발행
            markers = MarkerArray()
            stamp = self.get_clock().now().to_msg()

            # 기존 marker 전부 삭제
            delete_all = Marker()
            delete_all.action = Marker.DELETEALL
            markers.markers.append(delete_all)

            mid = 0  # marker id counter

            # === /end_pose_left, /end_pose_right 마커 (기준: arm_l_link1, arm_r_link1) ===
            mid = self._add_sphere(markers, mid, stamp, "arm_l_link1",
                                   current_left[:3], [0.0, 0.3, 1.0, 1.0], 0.03,
                                   "end_pose_left")
            mid = self._add_sphere(markers, mid, stamp, "arm_r_link1",
                                   current_right[:3], [1.0, 0.2, 0.2, 1.0], 0.03,
                                   "end_pose_right")

            # === Left arm trajectory: 파란색 계열 ===
            for i, pose in enumerate(traj_left):
                alpha = 0.3 + 0.7 * (i / max(len(traj_left) - 1, 1))
                mid = self._add_sphere(markers, mid, stamp, frame_left,
                                       pose[:3], [0.2, 0.5, 1.0, alpha], 0.015,
                                       f"traj_L_{i}")

            # Left trajectory line
            mid = self._add_line(markers, mid, stamp, frame_left,
                                 [current_left[:3]] + [p[:3] for p in traj_left],
                                 [0.2, 0.5, 1.0, 0.8], 0.005, "traj_line_left")

            # === Right arm trajectory: 빨간색 계열 ===
            for i, pose in enumerate(traj_right):
                alpha = 0.3 + 0.7 * (i / max(len(traj_right) - 1, 1))
                mid = self._add_sphere(markers, mid, stamp, frame_right,
                                       pose[:3], [1.0, 0.4, 0.2, alpha], 0.015,
                                       f"traj_R_{i}")

            # Right trajectory line
            mid = self._add_line(markers, mid, stamp, frame_right,
                                 [current_right[:3]] + [p[:3] for p in traj_right],
                                 [1.0, 0.4, 0.2, 0.8], 0.005, "traj_line_right")

            # === Delta arrows (첫 step만, 크게) ===
            if len(traj_left) > 0:
                mid = self._add_arrow(markers, mid, stamp, frame_left,
                                      current_left[:3], traj_left[0][:3],
                                      [0.0, 1.0, 0.5, 1.0], 0.008, "delta_L_0")
            if len(traj_right) > 0:
                mid = self._add_arrow(markers, mid, stamp, frame_right,
                                      current_right[:3], traj_right[0][:3],
                                      [0.0, 1.0, 0.5, 1.0], 0.008, "delta_R_0")

            # === 최종 목표 위치: 초록 원 = current + delta (같은 프레임에서 표현) ===
            if len(traj_left) > 0:
                # target = current_left + accumulated delta (traj_left[-1]이 이미 그 결과)
                mid = self._add_circle(markers, mid, stamp, frame_left,
                                       traj_left[-1][:3], [0.0, 1.0, 0.0, 1.0], 0.025, 0.005,
                                       "target_left")
            if len(traj_right) > 0:
                mid = self._add_circle(markers, mid, stamp, frame_right,
                                       traj_right[-1][:3], [0.0, 1.0, 0.0, 1.0], 0.025, 0.005,
                                       "target_right")

            # === Text: inference info ===
            mid = self._add_text(markers, mid, stamp, "base_link",
                                 [0.0, 0.0, 1.5],
                                 f"Inference #{self.inference_count}  "
                                 f"{inference_time*1000:.0f}ms  "
                                 f"dL=[{action.rel_end_pose_left[0][0]:.4f}, "
                                 f"{action.rel_end_pose_left[0][1]:.4f}, "
                                 f"{action.rel_end_pose_left[0][2]:.4f}]",
                                 "info_text")

            # 궤적/타깃/텍스트만 캐시 (end_pose 구는 매 주기에서 최신으로 발행)
            self._cached_markers_except_end_pose = copy.deepcopy([
                m for m in markers.markers
                if m.action != Marker.DELETEALL
                and getattr(m, "ns", "") not in ("end_pose_left", "end_pose_right")
            ])

            self.marker_pub.publish(markers)

            # 로그 출력
            dl = action.rel_end_pose_left[0][:3] if len(action.rel_end_pose_left.shape) > 1 else action.rel_end_pose_left[:3]
            dr = action.rel_end_pose_right[0][:3] if len(action.rel_end_pose_right.shape) > 1 else action.rel_end_pose_right[:3]
            self.get_logger().info(
                f"[#{self.inference_count}] {inference_time*1000:.0f}ms | "
                f"dL=[{dl[0]:+.5f}, {dl[1]:+.5f}, {dl[2]:+.5f}] "
                f"dR=[{dr[0]:+.5f}, {dr[1]:+.5f}, {dr[2]:+.5f}] | "
                f"16-step total dL=[{sum(action.rel_end_pose_left[:,0]):.4f}, "
                f"{sum(action.rel_end_pose_left[:,1]):.4f}, "
                f"{sum(action.rel_end_pose_left[:,2]):.4f}]"
            )

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _marker_update_cb(self):
        """end_pose 구만 높은 주기로 갱신하여 늦게 따라가는 느낌 방지."""
        if self.latest_end_pose_left is None or self.latest_end_pose_right is None:
            return
        stamp = self.get_clock().now().to_msg()
        current_left = self._extract_pose_array(self.latest_end_pose_left)
        current_right = self._extract_pose_array(self.latest_end_pose_right)

        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        mid = 0
        mid = self._add_sphere(markers, mid, stamp, "arm_l_link1",
                               current_left[:3], [0.0, 0.3, 1.0, 1.0], 0.03,
                               "end_pose_left")
        mid = self._add_sphere(markers, mid, stamp, "arm_r_link1",
                               current_right[:3], [1.0, 0.2, 0.2, 1.0], 0.03,
                               "end_pose_right")

        for m in self._cached_markers_except_end_pose:
            m_copy = copy.deepcopy(m)
            m_copy.header.stamp = stamp
            markers.markers.append(m_copy)

        self.marker_pub.publish(markers)

    def _compute_trajectory(self, current_pose, rel_deltas):
        """현재 위치에서 16 step의 delta를 누적 적용하여 trajectory 계산."""
        trajectory = []
        pose = current_pose.copy()
        n_steps = rel_deltas.shape[0] if len(rel_deltas.shape) > 1 else 1
        for i in range(n_steps):
            delta = rel_deltas[i].flatten() if len(rel_deltas.shape) > 1 else rel_deltas.flatten()
            pose = apply_relative_action(pose, delta)
            trajectory.append(pose.copy())
        return trajectory

    # === Marker helpers ===

    def _add_sphere(self, ma, mid, stamp, frame, pos, rgba, scale, ns):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = frame
        m.ns = ns
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(pos[0]), float(pos[1]), float(pos[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
        m.lifetime.sec = 1
        ma.markers.append(m)
        return mid + 1

    def _add_circle(self, ma, mid, stamp, frame, center, rgba, radius, line_width, ns):
        """Draw a circle in XY plane at center (LINE_STRIP)."""
        n_pts = 32
        points = []
        for i in range(n_pts + 1):
            th = 2.0 * np.pi * i / n_pts
            points.append([
                float(center[0]) + radius * np.cos(th),
                float(center[1]) + radius * np.sin(th),
                float(center[2]),
            ])
        return self._add_line(ma, mid, stamp, frame, points, rgba, line_width, ns)

    def _add_line(self, ma, mid, stamp, frame, points, rgba, width, ns):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = frame
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = width
        m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
        for p in points:
            pt = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            m.points.append(pt)
        m.lifetime.sec = 1
        ma.markers.append(m)
        return mid + 1

    def _add_arrow(self, ma, mid, stamp, frame, start, end, rgba, shaft_d, ns):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = frame
        m.ns = ns
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.points.append(Point(x=float(start[0]), y=float(start[1]), z=float(start[2])))
        m.points.append(Point(x=float(end[0]), y=float(end[1]), z=float(end[2])))
        m.scale.x = shaft_d
        m.scale.y = shaft_d * 2
        m.scale.z = 0.0
        m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
        m.lifetime.sec = 1
        ma.markers.append(m)
        return mid + 1

    def _add_text(self, ma, mid, stamp, frame, pos, text, ns):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = frame
        m.ns = ns
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(pos[0]), float(pos[1]), float(pos[2])
        m.pose.orientation.w = 1.0
        m.scale.z = 0.05
        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        m.text = text
        m.lifetime.sec = 1
        ma.markers.append(m)
        return mid + 1


def parse_args():
    ap = argparse.ArgumentParser(
        description="Model output visualization with RViz Markers (NO Servo)")

    ap.add_argument("--checkpoint", type=str,
                    default="/Isaac-GR00T/results/260214_endpose_abs_rel_1frame_shift/checkpoint-7000")
    ap.add_argument("--embodiment", type=str, default="new_embodiment")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--rate", type=float, default=2.0,
                    help="Inference rate in Hz (default: 2.0, slow for visualization)")
    ap.add_argument("--marker-rate", type=float, default=30.0,
                    help="End-pose sphere marker publish rate in Hz (default: 30, reduces lag)")
    ap.add_argument("--task", type=str, default="Execute the task")
    ap.add_argument("--action-horizon", type=int, default=16)

    ap.add_argument("--ego-view-topic", type=str,
                    default="/zed/zed_node/left/image_rect_color/compressed")
    ap.add_argument("--wrist-left-topic", type=str,
                    default="/camera_left/camera_left/color/image_rect_raw/compressed")
    ap.add_argument("--wrist-right-topic", type=str,
                    default="/camera_right/camera_right/color/image_rect_raw/compressed")
    ap.add_argument("--end-pose-left-topic", type=str, default="/end_pose_left")
    ap.add_argument("--end-pose-right-topic", type=str, default="/end_pose_right")
    ap.add_argument("--joint-topic", type=str, default="/joint_states")

    ap.add_argument("--use-sim-time", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Model Output Marker Test (NO SERVO, NO ROBOT MOVEMENT)")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Rate: {args.rate} Hz")
    print(f"  Use sim time: {args.use_sim_time}")
    print()
    print("RViz Setup:")
    print("  1. Add MarkerArray display")
    print("  2. Set topic to /model_output_markers")
    print("  3. Markers:")
    print("     - Blue sphere:  /end_pose_left  (frame: arm_l_link1)")
    print("     - Red sphere:   /end_pose_right (frame: arm_r_link1)")
    print("     - Blue path:    predicted left trajectory (16 steps)")
    print("     - Red path:     predicted right trajectory (16 steps)")
    print("     - Green circle: target = current + delta (same frame as end_pose)")
    print("     - Green arrow:  first delta direction")
    print("=" * 60)

    rclpy.init()

    try:
        node = ModelOutputMarkerNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutdown")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
