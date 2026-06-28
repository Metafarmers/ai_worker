#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Woojin Wie

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

_RVIZ_BY_PLANNING_GROUP = {
    'dual_arms': 'moveit.rviz',
    'arm_l': 'moveit_arm_l.rviz',
    'arm_r': 'moveit_arm_r.rviz',
}


def _launch_setup(context, *args, **kwargs):
    use_sim = LaunchConfiguration('use_sim')
    start_rviz = LaunchConfiguration('start_rviz')
    warehouse_sqlite_path = LaunchConfiguration('warehouse_sqlite_path')
    publish_robot_description_semantic = LaunchConfiguration(
        'publish_robot_description_semantic'
    )
    planning_group = LaunchConfiguration('planning_group').perform(context).strip()
    rviz_file = _RVIZ_BY_PLANNING_GROUP.get(planning_group, 'moveit.rviz')
    if planning_group not in _RVIZ_BY_PLANNING_GROUP:
        print(
            f'[moveit.launch.py] Unknown planning_group={planning_group!r}; '
            f'using {_RVIZ_BY_PLANNING_GROUP["dual_arms"]}. '
            f'Valid: {sorted(_RVIZ_BY_PLANNING_GROUP)}'
        )

    moveit_config = (
        MoveItConfigsBuilder(robot_name='ffw', package_name='ffw_moveit_config')
        .robot_description_semantic(file_path='config/ffw.srdf')
        .joint_limits(str(Path('config') / 'joint_limits.yaml'))
        .trajectory_execution(str(Path('config') / 'moveit_controllers.yaml'))
        .robot_description_kinematics(str(Path('config') / 'kinematics.yaml'))
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )

    warehouse_ros_config = {
        'warehouse_plugin': 'warehouse_ros_sqlite::DatabaseConnection',
        'warehouse_host': warehouse_sqlite_path,
    }

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            warehouse_ros_config,
            {
                'use_sim_time': use_sim,
                'publish_robot_description_semantic': publish_robot_description_semantic,
            },
        ],
    )

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare('ffw_moveit_config'), 'config', rviz_file]
    )
    rviz_node = Node(
        package='rviz2',
        condition=IfCondition(start_rviz),
        executable='rviz2',
        name='rviz2_moveit',
        output='log',
        arguments=['-d', rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            warehouse_ros_config,
            {
                'use_sim_time': use_sim,
            },
        ],
    )

    return [move_group_node, rviz_node]


def generate_launch_description():
    # Declare launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            'start_rviz', default_value='true', description='Whether to execute rviz2'
        ),
        DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Whether to use simulation time',
        ),
        DeclareLaunchArgument(
            'warehouse_sqlite_path',
            default_value=os.path.expanduser('~/.ros/warehouse_ros.sqlite'),
            description='Path where the warehouse database should be stored',
        ),
        DeclareLaunchArgument(
            'publish_robot_description_semantic',
            default_value='true',
            description='Whether to publish robot description semantic',
        ),
        DeclareLaunchArgument(
            'planning_group',
            default_value='dual_arms',
            description='RViz Motion Planning default group: dual_arms | arm_l | arm_r',
        ),
    ]

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=_launch_setup)])
