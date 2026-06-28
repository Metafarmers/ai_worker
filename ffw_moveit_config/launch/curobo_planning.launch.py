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
# Real robot example (RViz check, then confirm_execute):
#   ros2 launch ffw_moveit_config moveit.launch.py use_sim:=false
#   ros2 launch ffw_moveit_config curobo_planning.launch.py use_sim:=false \
#     dual_sequence:=lift_midpoint send_obstacles_to_curobo:=true \
#     require_back_wall_for_dual:=false max_attempts:=30 orientation_weight:=0.1
# Perception back wall off (separate node): enable_target_back_wall:=false on perception_click_planning_node
# Return to initial pose after a move (cuRobo joint plan + confirm):
#   ros2 topic pub --once /curobo/plan_home std_msgs/msg/Empty "{}"
#   ros2 topic pub --once /curobo/confirm_execute std_msgs/msg/Empty "{}"
# Return to initial pose via direct joint move (no cuRobo / no confirm):
#   ros2 topic pub --once /curobo/joint_home std_msgs/msg/Empty "{}"

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node

_DEFAULT_CUROBO_URL = os.environ.get('CUROBO_URL', 'http://127.0.0.1:8000/plan')


def _truthy(value: str) -> bool:
  return value.strip().lower() in ('true', '1', 'yes')


def _launch_setup(context):
  pkg_share = get_package_share_directory('ffw_moveit_config')
  joint_home_config = os.path.join(pkg_share, 'config', 'robot_joint_home.yaml')
  use_sim = _truthy(LaunchConfiguration('use_sim').perform(context))
  skip_arg = LaunchConfiguration('skip_start_feasibility_check').perform(context).strip()
  if skip_arg == '':
    skip_start = use_sim
  else:
    skip_start = _truthy(skip_arg)
  planning_mode = LaunchConfiguration('planning_mode').perform(context).strip().lower()

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
        'min_trajectory_duration_sec': float(
          LaunchConfiguration('min_trajectory_duration_sec').perform(context)
        ),
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
        'planning_mode': planning_mode,
        'goal_orientation_mode': LaunchConfiguration('goal_orientation_mode').perform(
          context
        ),
        'orientation_weight': float(
          LaunchConfiguration('orientation_weight').perform(context)
        ),
        'right_arm_roll_rad': float(
          LaunchConfiguration('right_arm_roll_rad').perform(context)
        ),
        'dual_sequence': LaunchConfiguration('dual_sequence').perform(context),
        'dual_sequence_settle_sec': float(
          LaunchConfiguration('dual_sequence_settle_sec').perform(context)
        ),
        'dual_sequence_joint_wait_sec': float(
          LaunchConfiguration('dual_sequence_joint_wait_sec').perform(context)
        ),
        'dual_require_right_confirm': _truthy(
          LaunchConfiguration('dual_require_right_confirm').perform(context)
        ),
        'dual_plan_lift': _truthy(
          LaunchConfiguration('dual_plan_lift').perform(context)
        ),
        'require_back_wall_for_dual': _truthy(
          LaunchConfiguration('require_back_wall_for_dual').perform(context)
        ),
        'max_attempts': int(LaunchConfiguration('max_attempts').perform(context)),
        'use_passive_defaults_for_start': False,
        'publish_curobo_obstacle_debug': True,
        'enable_final_tool_approach': True,
        'final_approach_tool_m': 0.08,
        'final_approach_max_distance_m': 0.35,
        'final_approach_use_move_group': True,
        'final_approach_settle_sec': 1.0,
        'joint_home_config': joint_home_config,
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
      default_value='20',
      description=(
        'Speed: 20 = 20%% (percent) or use fraction 0.2. '
        'Values >1 are percent; do NOT pass 100 unless you want full speed.'
      ),
    ),
    DeclareLaunchArgument(
      'min_trajectory_duration_sec',
      default_value='8.0',
      description='Minimum execute duration (s) after velocity_scale; prevents snap-like moves',
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
      default_value='0.6',
      description=(
        'Scale click obstacle cuboid dims for cuRobo and MoveIt '
        '(match obstacle_dim_scale on perception_click_planning_node). '
        '0.5–0.7 reduces EE gap vs clicked obstacles; 1.0 = full cuboid_size_*.'
      ),
    ),
    DeclareLaunchArgument(
      'skip_start_feasibility_check',
      default_value='',
      description=(
        'Skip cuRobo start self-collision gate. Empty = true when use_sim:=true.'
      ),
    ),
    DeclareLaunchArgument(
      'planning_mode',
      default_value='dual',
      description='left | right | dual (dual plans both arms when /target_pose and /target_pose_r are set)',
    ),
    DeclareLaunchArgument(
      'dual_sequence',
      default_value='plan_merged',
      description=(
        'dual only: plan_merged = plan L+lift, plan R at L goal, merge, 1 confirm | '
        'parallel = same-time plan | left_then_right = execute L then plan R | '
        'lift_midpoint = probe L+lift & R+lift, lift to midpoint, replan L/R | '
        'right_lift_first = alias for lift_midpoint'
      ),
    ),
    DeclareLaunchArgument(
      'dual_sequence_settle_sec',
      default_value='1.0',
      description='Pause after left+lift before planning right (dual_sequence:=left_then_right)',
    ),
    DeclareLaunchArgument(
      'dual_sequence_joint_wait_sec',
      default_value='5.0',
      description='Max wait for /joint_states after left+lift before right plan',
    ),
    DeclareLaunchArgument(
      'dual_require_right_confirm',
      default_value='false',
      description=(
        'dual left_then_right: false = confirm left only, then auto plan+execute right'
      ),
    ),
    DeclareLaunchArgument(
      'dual_plan_lift',
      default_value='true',
      description=(
        'dual mode: plan lift_joint with left arm so EE can reach target z. '
        'false = lift fixed at start (EE may stop above low targets).'
      ),
    ),
    DeclareLaunchArgument(
      'require_back_wall_for_dual',
      default_value='true',
      description=(
        'dual mode: wait for left-target back wall cuboid on /obstacle_markers before planning. '
        'Set false when enable_target_back_wall:=false on perception_click_planning_node.'
      ),
    ),
    DeclareLaunchArgument(
      'max_attempts',
      default_value='15',
      description='cuRobo HTTP /plan max_attempts (graph retries from attempt 1)',
    ),
    DeclareLaunchArgument(
      'goal_orientation_mode',
      default_value='toward_robot',
      description=(
        'toward_robot: +X→base (+ right roll pi) soft-tracked | free: position only | '
        'from_message: PoseStamped quat | fixed: full orientation'
      ),
    ),
    DeclareLaunchArgument(
      'orientation_weight',
      default_value='0.2',
      description=(
        'Soft orientation strength for toward_robot/from_message (0=ignore, 1=hard). '
        '0.2 = loose; right arm still uses roll pi hint.'
      ),
    ),
    DeclareLaunchArgument(
      'right_arm_roll_rad',
      default_value='3.141592653589793',
      description='Extra roll (rad) about EE +X for right arm toward_robot (default pi)',
    ),
    OpaqueFunction(function=_launch_setup),
  ])
