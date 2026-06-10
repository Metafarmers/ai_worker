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
#     velocity_scale:=0.15 confirm_before_execute:=true skip_start_feasibility_check:=false

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node

_DEFAULT_CUROBO_URL = os.environ.get('CUROBO_URL', 'http://127.0.0.1:8000/plan')


def _truthy(value: str) -> bool:
  return value.strip().lower() in ('true', '1', 'yes')


def _launch_setup(context):
  use_sim = _truthy(LaunchConfiguration('use_sim').perform(context))
  skip_arg = LaunchConfiguration('skip_start_feasibility_check').perform(context).strip()
  if skip_arg == '':
    skip_start = use_sim
  else:
    skip_start = _truthy(skip_arg)

  return [
    Node(
      package='ffw_moveit_config',
      executable='curobo_pathplanning_node.py',
      output='screen',
      emulate_tty=True,
      parameters=[{
        'use_sim_time': use_sim,
        'curobo_url': LaunchConfiguration('curobo_url').perform(context),
        'auto_plan': _truthy(LaunchConfiguration('auto_plan').perform(context)),
        'move_group_namespace': LaunchConfiguration('move_group_namespace').perform(context),
        'velocity_scale': float(LaunchConfiguration('velocity_scale').perform(context)),
        'confirm_before_execute': _truthy(
          LaunchConfiguration('confirm_before_execute').perform(context)
        ),
        'send_obstacles_to_curobo': _truthy(
          LaunchConfiguration('send_obstacles_to_curobo').perform(context)
        ),
        'curobo_obstacle_dim_scale': float(
          LaunchConfiguration('curobo_obstacle_dim_scale').perform(context)
        ),
        'skip_start_feasibility_check': skip_start,
        'use_passive_defaults_for_start': True,
        'publish_curobo_obstacle_debug': True,
      }],
    ),
  ]


def generate_launch_description():
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
    DeclareLaunchArgument(
      'curobo_obstacle_dim_scale',
      default_value='0.92',
      description='Shrink obstacle cuboids sent to cuRobo (reduces start false positives)',
    ),
    DeclareLaunchArgument(
      'skip_start_feasibility_check',
      default_value='',
      description=(
        'Skip cuRobo start self-collision gate. Empty = true when use_sim:=true.'
      ),
    ),
    OpaqueFunction(function=_launch_setup),
  ])
