#!/usr/bin/env python3
"""
QoS bridge: /joint_states (RELIABLE+TRANSIENT_LOCAL) → /joint_states_servo (BEST_EFFORT+VOLATILE)

MoveIt Servo's PlanningSceneMonitor (CurrentStateMonitor) subscribes with
BEST_EFFORT+VOLATILE, but joint_state_broadcaster publishes with
RELIABLE+TRANSIENT_LOCAL. On FastDDS this combination can fail silently.

This node bridges the gap by subscribing with the publisher's QoS and
republishing with a QoS that CurrentStateMonitor can receive.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from sensor_msgs.msg import JointState


class JointStateQoSBridge(Node):
    def __init__(self):
        super().__init__("joint_state_qos_bridge")

        self.declare_parameter("input_topic", "/joint_states")
        self.declare_parameter("output_topic", "/joint_states_servo")

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(JointState, output_topic, pub_qos)
        self.sub = self.create_subscription(
            JointState, input_topic, self._callback, sub_qos
        )

        self.get_logger().info(
            f"QoS bridge: {input_topic} (RELIABLE) → {output_topic} (BEST_EFFORT)"
        )

    def _callback(self, msg: JointState):
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = JointStateQoSBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
