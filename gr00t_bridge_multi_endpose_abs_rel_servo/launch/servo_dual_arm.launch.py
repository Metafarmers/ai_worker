#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Launch file for dual-arm MoveIt Servo nodes (arm_l + arm_r).
#
# Each Servo node is a primary planning scene monitor — subscribes directly
# to /joint_states. No separate move_group node needed.
#
# Prerequisites:
#   1. Robot bringup must be running (robot_state_publisher, controllers)
#   2. ros-jazzy-moveit-servo must be installed

import os
import yaml
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "use_sim",
            default_value="false",
            description="Whether to use simulation time",
        ),
    ]

    use_sim = LaunchConfiguration("use_sim")

    moveit_config = (
        MoveItConfigsBuilder("ffw", package_name="ffw_moveit_config")
        .robot_description_semantic(file_path="config/ffw.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )

    config_dir = str(Path(__file__).resolve().parent.parent / "config")

    servo_params_left = load_yaml(
        os.path.join(config_dir, "servo_params_left.yaml")
    )
    servo_params_right = load_yaml(
        os.path.join(config_dir, "servo_params_right.yaml")
    )

    # Each Servo node is a primary planning scene monitor — subscribes directly to /joint_states
    servo_params_left["moveit_servo"]["is_primary_planning_scene_monitor"] = True
    servo_params_right["moveit_servo"]["is_primary_planning_scene_monitor"] = True

    acceleration_filter_update_period = {"update_period": 0.02}

    servo_node_left = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        namespace="servo_left",
        parameters=[
            servo_params_left,
            acceleration_filter_update_period,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim},
        ],
        remappings=[
            ("joint_states", "/joint_states"),
        ],
        output="screen",
    )

    servo_node_right = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        namespace="servo_right",
        parameters=[
            servo_params_right,
            acceleration_filter_update_period,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim},
        ],
        remappings=[
            ("joint_states", "/joint_states"),
        ],
        output="screen",
    )

    return LaunchDescription(
        declared_arguments
        + [
            servo_node_left,
            servo_node_right,
        ]
    )
