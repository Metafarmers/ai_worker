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
# Example:
#   ros2 launch ffw_moveit_config curobo_planning.launch.py use_sim:=true \
#     curobo_url:=http://172.17.0.1:8000/plan

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

  return LaunchDescription([
    DeclareLaunchArgument('use_sim', default_value='true'),
    DeclareLaunchArgument(
      'curobo_url',
      default_value=[EnvironmentVariable('CUROBO_URL', default_value=_DEFAULT_CUROBO_URL)],
      description='cuRobo plan server base URL (must be reachable from this container)',
    ),
    DeclareLaunchArgument('auto_plan', default_value='true'),
    Node(
      package='ffw_moveit_config',
      executable='curobo_pathplanning_node.py',
      output='screen',
      parameters=[{
        'use_sim_time': use_sim,
        'curobo_url': curobo_url,
        'auto_plan': auto_plan,
        'move_group_namespace': 'root',
      }],
    ),
  ])
