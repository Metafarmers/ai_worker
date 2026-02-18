#!/usr/bin/env python3
"""
Publish missing gripper mimic joints to /joint_states.
Fixes MoveIt Servo "Waiting to receive robot state update" by ensuring
all URDF active joints appear in /joint_states.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState


class MissingJointPublisher(Node):
    def __init__(self):
        super().__init__("missing_joint_publisher")

        self.missing_joints = [
            "gripper_l_joint2",
            "gripper_l_joint3",
            "gripper_l_joint4",
            "gripper_r_joint2",
            "gripper_r_joint3",
            "gripper_r_joint4",
        ]

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(JointState, "/joint_states", qos)
        # 50Hz로 publish
        self.timer = self.create_timer(0.02, self._publish)
        self.get_logger().info(
            f"Publishing {len(self.missing_joints)} missing joints to /joint_states"
        )

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.missing_joints
        msg.position = [0.0] * len(self.missing_joints)
        msg.velocity = [0.0] * len(self.missing_joints)
        msg.effort = [0.0] * len(self.missing_joints)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = MissingJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()