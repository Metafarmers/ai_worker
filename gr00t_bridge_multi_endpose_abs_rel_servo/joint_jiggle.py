#!/usr/bin/env python3
"""
Joint State Jiggle — MoveIt Servo unblock utility.

MoveIt2's CurrentStateMonitor ignores /joint_states when all positions
are unchanged between consecutive messages. This script reads the current
state, publishes a tiny perturbation (+0.0001 rad), then restores the
original values so Servo recognises "state received".

Usage:
    python3 joint_state_jiggle.py          # one-shot jiggle
    python3 joint_state_jiggle.py --loop   # keep jiggling every 2s (for debugging)
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState

JIGGLE_AMOUNT = 0.0001  # rad — small enough to be invisible


class JiggleNode(Node):
    def __init__(self):
        super().__init__("joint_state_jiggle")
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE

        self.current_msg = None
        self.sub = self.create_subscription(
            JointState, "/joint_states", self._cb, qos
        )
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

    def _cb(self, msg: JointState):
        self.current_msg = msg

    def wait_for_msg(self, timeout: float = 5.0) -> bool:
        t0 = time.time()
        while self.current_msg is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.current_msg is not None

    def jiggle(self):
        if self.current_msg is None:
            self.get_logger().error("No /joint_states received")
            return False

        orig = self.current_msg

        # 1) Publish perturbed
        jig = JointState()
        jig.header.stamp = self.get_clock().now().to_msg()
        jig.name = list(orig.name)
        jig.position = [p + JIGGLE_AMOUNT for p in orig.position]
        jig.velocity = list(orig.velocity) if orig.velocity else []
        jig.effort = list(orig.effort) if orig.effort else []
        self.pub.publish(jig)
        self.get_logger().info(
            f"Jiggle published (+{JIGGLE_AMOUNT} rad) for {len(jig.name)} joints"
        )

        time.sleep(0.05)

        # 2) Restore original
        restore = JointState()
        restore.header.stamp = self.get_clock().now().to_msg()
        restore.name = list(orig.name)
        restore.position = list(orig.position)
        restore.velocity = list(orig.velocity) if orig.velocity else []
        restore.effort = list(orig.effort) if orig.effort else []
        self.pub.publish(restore)
        self.get_logger().info("Original values restored")
        return True


def main():
    rclpy.init()
    node = JiggleNode()

    loop = "--loop" in sys.argv

    node.get_logger().info("Waiting for /joint_states ...")
    if not node.wait_for_msg():
        node.get_logger().error("Timeout — /joint_states not received")
        node.destroy_node()
        rclpy.shutdown()
        return

    if loop:
        node.get_logger().info("Loop mode — Ctrl+C to stop")
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                node.jiggle()
                time.sleep(2.0)
        except KeyboardInterrupt:
            pass
    else:
        node.jiggle()
        time.sleep(0.2)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()