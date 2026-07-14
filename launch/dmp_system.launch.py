#!/usr/bin/env python3
"""
ROS2 Launch file for DMP system
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    dmp_type_arg = DeclareLaunchArgument(
        'dmp_type',
        default_value='discrete',
        description='DMP type: discrete or rhythmic'
    )
    
    n_dmps_arg = DeclareLaunchArgument(
        'n_dmps',
        default_value='3',
        description='Number of DMPs'
    )
    
    n_bfs_arg = DeclareLaunchArgument(
        'n_bfs',
        default_value='10',
        description='Number of basis functions'
    )
    
    # Get config files
    config_dir = get_package_share_directory('ros2_dmp_node')
    params_file = os.path.join(config_dir, 'config', 'dmp_params.yaml')
    
    # DMP Node
    dmp_node = Node(
        package='ros2_dmp_node',
        executable='dmp_node',
        name='dmp_node',
        output='screen',
        parameters=[{
            'dmp_type': LaunchConfiguration('dmp_type'),
            'n_dmps': LaunchConfiguration('n_dmps'),
            'n_bfs': LaunchConfiguration('n_bfs'),
        }]
    )
    
    # UI Bridge (optional)
    ui_bridge = Node(
        package='ros2_dmp_node',
        executable='ui_bridge',
        name='ui_bridge',
        output='screen'
    )
    
    return LaunchDescription([
        dmp_type_arg,
        n_dmps_arg,
        n_bfs_arg,
        dmp_node,
    ])
