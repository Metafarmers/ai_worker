#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0

from glob import glob

from setuptools import find_packages
from setuptools import setup

package_name = 'gr00t_bridge_multi_endpose_abs_rel_servo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='ROBOTIS',
    author_email='support@robotis.com',
    maintainer='ROBOTIS',
    maintainer_email='support@robotis.com',
    description='GR00T End-Pose Relative Action Inference Bridge with MoveIt Servo',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gr00t_endpose_rel_servo_node = gr00t_bridge_multi_endpose_abs_rel_servo.gr00t_endpose_rel_servo_node:main',
        ],
    },
)
