#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""
Gripper bridge: subscribes to Float64 gripper commands and publishes gripper-only
JointTrajectory to the leader joint_trajectory topics using local now() so the
controller (same container) never sees 'trajectory ends in the past'.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration


class GripperBridgeNode(Node):
    LEFT_GRIPPER_JOINTS = ["gripper_l_joint1"]
    RIGHT_GRIPPER_JOINTS = ["gripper_r_joint1"]

    def __init__(self):
        super().__init__("gripper_bridge_node")
        self.declare_parameter("gripper_time_from_start_sec", 0.05)

        self.gripper_time_from_start_sec = self.get_parameter(
            "gripper_time_from_start_sec"
        ).get_parameter_value().double_value

        self.sub_left = self.create_subscription(
            Float64,
            "/gripper_left_command",
            self._cb_left,
            10,
        )
        self.sub_right = self.create_subscription(
            Float64,
            "/gripper_right_command",
            self._cb_right,
            10,
        )
        self.pub_left = self.create_publisher(
            JointTrajectory,
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            10,
        )
        self.pub_right = self.create_publisher(
            JointTrajectory,
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            10,
        )
        self.get_logger().info(
            "Gripper bridge: /gripper_*_command (Float64) -> leader joint_trajectory (gripper only, stamp=now())"
        )

    def _publish_gripper_trajectory(self, value: float, side: str):
        msg = JointTrajectory()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""

        if side == "left":
            msg.joint_names = self.LEFT_GRIPPER_JOINTS
            pub = self.pub_left
        else:
            msg.joint_names = self.RIGHT_GRIPPER_JOINTS
            pub = self.pub_right

        duration_ns = int(self.gripper_time_from_start_sec * 1_000_000_000)
        point = JointTrajectoryPoint()
        point.positions = [value]
        point.time_from_start = Duration(sec=0, nanosec=duration_ns)
        msg.points = [point]
        pub.publish(msg)

    def _cb_left(self, msg: Float64):
        self._publish_gripper_trajectory(msg.data, "left")

    def _cb_right(self, msg: Float64):
        self._publish_gripper_trajectory(msg.data, "right")


def main(args=None):
    rclpy.init(args=args)
    node = GripperBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
