#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# FFW Docker: bridges /obstacle_markers -> cuRobo HTTP -> ExecuteTrajectory.
# cuRobo Docker: run curobo_plan_server.py on 0.0.0.0:8000 and publish the port.
#
# Separate containers — set CUROBO_URL to reach the cuRobo container, e.g.:
#   export CUROBO_URL=http://172.17.0.1:8000/plan          # host gateway (Linux)
#   export CUROBO_URL=http://curobo:8000/plan              # shared docker network
#   export CUROBO_URL=http://host.docker.internal:8000/plan  # + --add-host=host-gateway
#
# Build once in FFW workspace:
#   cd ~/ros2_ws && colcon build --packages-select ffw_moveit_config && source install/setup.bash
#
# Real robot example (RViz check, then Enter in terminal to execute):
#   ros2 launch ffw_moveit_config moveit.launch.py use_sim:=false
#   ros2 launch ffw_moveit_config curobo_planning.launch.py use_sim:=false \
#     velocity_scale:=0.15 confirm_before_execute:=true send_obstacles_to_curobo:=false

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node

_DEFAULT_CUROBO_URL = os.environ.get('CUROBO_URL', 'http://127.0.0.1:8000/plan')


def generate_launch_description():
  use_sim = LaunchConfiguration('use_sim')
  curobo_url = LaunchConfiguration('curobo_url')
  auto_plan = LaunchConfiguration('auto_plan')
  velocity_scale = LaunchConfiguration('velocity_scale')
  confirm_before_execute = LaunchConfiguration('confirm_before_execute')
  send_obstacles_to_curobo = LaunchConfiguration('send_obstacles_to_curobo')
  move_group_namespace = LaunchConfiguration('move_group_namespace')

  return LaunchDescription([
    DeclareLaunchArgument('use_sim', default_value='true'),
    DeclareLaunchArgument(
      'curobo_url',
      default_value=[EnvironmentVariable('CUROBO_URL', default_value=_DEFAULT_CUROBO_URL)],
      description='cuRobo plan server base URL (must be reachable from this container)',
    ),
    DeclareLaunchArgument('auto_plan', default_value='true'),
    DeclareLaunchArgument(
      'velocity_scale',
      default_value='0.2',
      description='Trajectory speed scale in (0,1]; 0.2 = 20%% of cuRobo plan speed',
    ),
    DeclareLaunchArgument(
      'confirm_before_execute',
      default_value='true',
      description='Publish planned path to RViz; wait for Enter before ExecuteTrajectory',
    ),
    DeclareLaunchArgument(
      'send_obstacles_to_curobo',
      default_value='true',
      description='Send obstacle cuboids to cuRobo HTTP /plan',
    ),
    DeclareLaunchArgument(
      'move_group_namespace',
      default_value='root',
      description=(
        'MoveIt action namespace. Use root for ffw_moveit_config moveit.launch.py '
        '(ExecuteTrajectory at /execute_trajectory).'
      ),
    ),
    Node(
      package='ffw_moveit_config',
      executable='curobo_pathplanning_node.py',
      output='screen',
      emulate_tty=True,
      parameters=[{
        'use_sim_time': use_sim,
        'curobo_url': curobo_url,
        'auto_plan': auto_plan,
        'move_group_namespace': move_group_namespace,
        'velocity_scale': velocity_scale,
        'confirm_before_execute': confirm_before_execute,
        'send_obstacles_to_curobo': send_obstacles_to_curobo,
      }],
    ),
  ])
