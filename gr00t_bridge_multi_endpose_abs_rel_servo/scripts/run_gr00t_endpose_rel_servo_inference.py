#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# End-pose RELATIVE ACTION inference script — MoveIt Servo version
#
# Instead of calling /compute_ik, this publishes PoseStamped to MoveIt Servo.
# Servo handles IK, smoothing, singularity, and collision internally.
#
# Prerequisites:
#   1. Robot: ros2 launch ffw_bringup ffw_sg2_follower_ai.launch.py
#   2. End-pose publisher: python3 inference_time_end_pose.py --rate 50
#   3. MoveIt Servo (dual arm):
#      ros2 launch gr00t_bridge_multi_endpose_abs_rel_servo servo_dual_arm.launch.py
#
# Usage:
#   python3 run_gr00t_endpose_rel_servo_inference.py --dry-run
#   python3 run_gr00t_endpose_rel_servo_inference.py --publish

import argparse
import sys
from pathlib import Path

ISAAC_GROOT_PATH = Path("/workspace/Isaac-GR00T")
if ISAAC_GROOT_PATH.exists() and str(ISAAC_GROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_PATH))

GROOT_BRIDGE_PATH = Path(__file__).parent.parent.parent
if str(GROOT_BRIDGE_PATH) not in sys.path:
    sys.path.insert(0, str(GROOT_BRIDGE_PATH))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GR00T end-pose RELATIVE ACTION inference with MoveIt Servo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run mode (no publishing, just print)
  python3 run_gr00t_endpose_rel_servo_inference.py --dry-run

  # With action publishing via Servo
  python3 run_gr00t_endpose_rel_servo_inference.py --publish

  # Custom servo namespaces
  python3 run_gr00t_endpose_rel_servo_inference.py --publish \\
      --servo-left-ns servo_left --servo-right-ns servo_right

Prerequisites:
  1. Robot: ros2 launch ffw_bringup ffw_sg2_follower_ai.launch.py
  2. End-pose publisher: python3 inference_time_end_pose.py --rate 50
  3. MoveIt Servo:
     ros2 launch gr00t_bridge_multi_endpose_abs_rel_servo servo_dual_arm.launch.py
"""
    )

    parser.add_argument(
        "--checkpoint", type=str,
        default="/Isaac-GR00T/results/260222_bugfix_abs_rel/checkpoint-1000",
        # default="/Isaac-GR00T/results/260214_endpose_abs_rel_1frame_shift/checkpoint-7000",
        help="Path to GR00T end-pose relative action checkpoint",
    )
    parser.add_argument("--embodiment", type=str, default="new_embodiment")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--rate", type=float, default=20.0, help="Inference rate in Hz")

    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--publish", action="store_true", default=False)
    parser.add_argument("--step-by-step", action="store_true", default=False,
                        help="Wait for Enter key before each inference step (safe for real robot)")
    parser.add_argument("--task", type=str, default="Execute the task")

    parser.add_argument(
        "--ego-view-topic", type=str,
        default="/zed/zed_node/left/image_rect_color/compressed",
    )
    parser.add_argument(
        "--wrist-left-topic", type=str,
        default="/camera_left/camera_left/color/image_rect_raw/compressed",
    )
    parser.add_argument(
        "--wrist-right-topic", type=str,
        default="/camera_right/camera_right/color/image_rect_raw/compressed",
    )
    parser.add_argument("--end-pose-left-topic", type=str, default="/end_pose_left")
    parser.add_argument("--end-pose-right-topic", type=str, default="/end_pose_right")
    parser.add_argument("--joint-topic", type=str, default="/joint_states")

    parser.add_argument("--servo-left-ns", type=str, default="servo_left/servo_node",
                        help="Namespace/name of the left arm MoveIt Servo node")
    parser.add_argument("--servo-right-ns", type=str, default="servo_right/servo_node",
                        help="Namespace/name of the right arm MoveIt Servo node")

    parser.add_argument("--use-sim-time", action="store_true")
    parser.add_argument("--print-rate", type=float, default=1.0)

    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--action-horizon-execute", type=int, default=4)
    parser.add_argument("--action-horizon-step-delay", type=float, default=0.02)
    parser.add_argument("--gripper-step-duration", type=float, default=0.05,
                        help="time_from_start for gripper JointTrajectory points (sec)")

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    if args.publish:
        dry_run = False
    elif args.dry_run:
        dry_run = True
    else:
        dry_run = True
        print("[INFO] Neither --publish nor --dry-run specified. Using dry-run mode.")

    print("=" * 70)
    print("GR00T End-Pose Relative Action + MoveIt Servo Inference")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Embodiment: {args.embodiment}")
    print(f"  Device: {args.device}")
    print(f"  Inference rate: {args.rate} Hz")
    print(f"  Dry-run mode: {dry_run}")
    print(f"  Step-by-step: {args.step_by_step}")
    print(f"  Use sim time: {args.use_sim_time}")
    print(f"  Action horizon: {args.action_horizon}")
    print("-" * 70)
    print("Output: MoveIt Servo PoseCommand (non-blocking)")
    print("  Model → relative delta → absolute PoseStamped → Servo")
    print("  Servo handles: IK + smoothing + singularity + collision")
    print(f"  Servo left ns:  {args.servo_left_ns}")
    print(f"  Servo right ns: {args.servo_right_ns}")
    print("-" * 70)
    print("Input Topics:")
    print(f"  ego_view:        {args.ego_view_topic}")
    print(f"  cam_wrist_left:  {args.wrist_left_topic}")
    print(f"  cam_wrist_right: {args.wrist_right_topic}")
    print(f"  end_pose_left:   {args.end_pose_left_topic}")
    print(f"  end_pose_right:  {args.end_pose_right_topic}")
    print(f"  joint_states:    {args.joint_topic}")
    print("=" * 70)

    import rclpy
    from rclpy.parameter import Parameter

    rclpy.init()

    try:
        from gr00t_bridge_multi_endpose_abs_rel_servo.gr00t_endpose_rel_servo_node import (
            Gr00tEndPoseRelServoNode,
        )

        param_overrides = [
            Parameter("checkpoint_path", Parameter.Type.STRING, args.checkpoint),
            Parameter("embodiment_tag", Parameter.Type.STRING, args.embodiment),
            Parameter("device", Parameter.Type.STRING, args.device),
            Parameter("inference_rate", Parameter.Type.DOUBLE, float(args.rate)),
            Parameter("dry_run", Parameter.Type.BOOL, dry_run),
            Parameter("task_instruction", Parameter.Type.STRING, args.task),
            Parameter("ego_view_topic", Parameter.Type.STRING, args.ego_view_topic),
            Parameter("wrist_left_topic", Parameter.Type.STRING, args.wrist_left_topic),
            Parameter("wrist_right_topic", Parameter.Type.STRING, args.wrist_right_topic),
            Parameter("end_pose_left_topic", Parameter.Type.STRING, args.end_pose_left_topic),
            Parameter("end_pose_right_topic", Parameter.Type.STRING, args.end_pose_right_topic),
            Parameter("joint_state_topic", Parameter.Type.STRING, args.joint_topic),
            Parameter("print_rate", Parameter.Type.DOUBLE, float(args.print_rate)),
            Parameter("servo_left_ns", Parameter.Type.STRING, args.servo_left_ns),
            Parameter("servo_right_ns", Parameter.Type.STRING, args.servo_right_ns),
            Parameter("action_horizon", Parameter.Type.INTEGER, int(args.action_horizon)),
            Parameter("action_horizon_execute", Parameter.Type.INTEGER, int(args.action_horizon_execute)),
            Parameter("action_horizon_step_delay", Parameter.Type.DOUBLE, float(args.action_horizon_step_delay)),
            Parameter("gripper_step_duration_sec", Parameter.Type.DOUBLE, float(args.gripper_step_duration)),
            Parameter("step_by_step", Parameter.Type.BOOL, args.step_by_step),
        ]

        if args.use_sim_time:
            param_overrides.append(
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            )

        inference_node = Gr00tEndPoseRelServoNode(parameter_overrides=param_overrides)

        print("\n[INFO] Node started. Waiting for messages from ALL inputs...")
        print("[INFO] Required: 3 cameras + 2 end-poses + joint states")
        print("[INFO] Make sure MoveIt Servo is running:")
        print("[INFO]   ros2 launch gr00t_bridge_multi_endpose_abs_rel_servo servo_dual_arm.launch.py")
        if args.step_by_step:
            print("[INFO] *** STEP-BY-STEP MODE: Press Enter to execute each inference step ***")
        print("[INFO] Press Ctrl+C to stop\n")

        rclpy.spin(inference_node)

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
