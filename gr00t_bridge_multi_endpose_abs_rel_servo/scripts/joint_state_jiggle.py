#!/usr/bin/env python3
"""
One-shot workaround for MoveIt2 Issue #3040.

CurrentStateMonitor::jointStateCallback() ignores joint_states when positions
haven't changed from the previous value. If the robot starts with all joints
at 0.0, the monitor never registers a state update → Servo hangs at
"Waiting to receive robot state update."

This script publishes a single jiggled message (epsilon offset on the first
joint) followed by the original message, forcing the monitor to register.
"""
import sys
import time
import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from sensor_msgs.msg import JointState


class JointStateJiggle(Node):
    def __init__(self):
        super().__init__("joint_state_jiggle")
        self._done = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(JointState, "/joint_states", qos)
        self.sub = self.create_subscription(
            JointState, "/joint_states", self._callback, qos
        )
        self.get_logger().info("Waiting for /joint_states message...")

    def _callback(self, msg: JointState):
        if self._done:
            return
        self._done = True

        self.get_logger().info(
            f"Received {len(msg.name)} joints. Applying epsilon jiggle..."
        )

        jiggle_msg = copy.deepcopy(msg)
        positions = list(jiggle_msg.position)
        if positions:
            positions[0] += 1e-10
        jiggle_msg.position = positions
        jiggle_msg.header.stamp = self.get_clock().now().to_msg()

        self.pub.publish(jiggle_msg)
        self.get_logger().info("Published jiggled message")

        time.sleep(0.05)

        original_msg = copy.deepcopy(msg)
        original_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(original_msg)
        self.get_logger().info("Published original message (restored)")
        self.get_logger().info("Done. Servo should now pass 'Waiting' check.")

        raise SystemExit(0)


def main():
    rclpy.init()
    node = JointStateJiggle()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
