#!/usr/bin/env python3
# Copyright 2026
# Docker: ros2 launch ffw_moveit_config obstacle.launch.py (from ~/ros2_ws after colcon build)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
  return LaunchDescription([
    DeclareLaunchArgument('use_sim', default_value='true'),
    DeclareLaunchArgument('seed', default_value='-1'),
    DeclareLaunchArgument('use_fixed_pose', default_value='false'),
    DeclareLaunchArgument('fixed_x', default_value='0.11'),
    DeclareLaunchArgument('fixed_y', default_value='0.13'),
    DeclareLaunchArgument('fixed_z', default_value='1.35'),
    DeclareLaunchArgument('obstacle_x_min', default_value='0.10'),
    DeclareLaunchArgument('obstacle_x_max', default_value='0.12'),
    DeclareLaunchArgument('obstacle_y_min', default_value='0.12'),
    DeclareLaunchArgument('obstacle_y_max', default_value='0.14'),
    DeclareLaunchArgument('obstacle_z_min', default_value='1.28'),
    DeclareLaunchArgument('obstacle_z_max', default_value='1.48'),
    Node(
      package='ffw_moveit_config',
      executable='obstacle_node.py',
      output='screen',
      parameters=[{
        'use_sim_time': LaunchConfiguration('use_sim'),
        'seed': LaunchConfiguration('seed'),
        'use_fixed_pose': LaunchConfiguration('use_fixed_pose'),
        'fixed_x': LaunchConfiguration('fixed_x'),
        'fixed_y': LaunchConfiguration('fixed_y'),
        'fixed_z': LaunchConfiguration('fixed_z'),
        'obstacle_x_min': LaunchConfiguration('obstacle_x_min'),
        'obstacle_x_max': LaunchConfiguration('obstacle_x_max'),
        'obstacle_y_min': LaunchConfiguration('obstacle_y_min'),
        'obstacle_y_max': LaunchConfiguration('obstacle_y_max'),
        'obstacle_z_min': LaunchConfiguration('obstacle_z_min'),
        'obstacle_z_max': LaunchConfiguration('obstacle_z_max'),
      }],
    ),
  ])
