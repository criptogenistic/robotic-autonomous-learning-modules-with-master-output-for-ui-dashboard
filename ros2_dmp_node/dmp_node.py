#!/usr/bin/env python3
"""
ROS2 Node for Dynamic Movement Primitives (PyDMPs) Integration
Provides ROS2 interfaces for DMP learning, execution, and trajectory publishing
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional
import threading

from pydmps.dmp_discrete import DMPs_discrete
from pydmps.dmp_rhythmic import DMPs_rhythmic

from std_msgs.msg import String, Float32MultiArray, Float64
from geometry_msgs.msg import Twist, PoseStamped
from std_srvs.srv import SetBool, Trigger
from custom_interfaces.srv import DMPService, TrajectoryService
from custom_interfaces.msg import DMPState, TrajectoryData


class DMPNode(Node):
    """ROS2 Node for managing Dynamic Movement Primitives"""

    def __init__(self):
        super().__init__('dmp_node')
        
        # Declare parameters
        self.declare_parameter('dmp_type', 'discrete')  # or 'rhythmic'
        self.declare_parameter('n_dmps', 3)
        self.declare_parameter('n_bfs', 10)
        self.declare_parameter('dt', 0.01)
        self.declare_parameter('y0', 0.0)
        self.declare_parameter('goal', 1.0)
        self.declare_parameter('ui_port', 5000)
        
        self.dmp_type = self.get_parameter('dmp_type').value
        self.n_dmps = self.get_parameter('n_dmps').value
        self.n_bfs = self.get_parameter('n_bfs').value
        self.dt = self.get_parameter('dt').value
        self.y0 = self.get_parameter('y0').value
        self.goal = self.get_parameter('goal').value
        
        # Initialize DMP instances
        self.dmp_instances: Dict[str, Dict] = {}
        self.current_trajectory = None
        self.trajectory_lock = threading.Lock()
        self.callback_group = ReentrantCallbackGroup()
        
        # Publishers
        self.trajectory_pub = self.create_publisher(
            Float32MultiArray,
            '/dmp/trajectory',
            10,
            callback_group=self.callback_group
        )
        
        self.state_pub = self.create_publisher(
            DMPState,
            '/dmp/state',
            10,
            callback_group=self.callback_group
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/dmp/status',
            10,
            callback_group=self.callback_group
        )
        
        # Subscribers
        self.create_subscription(
            String,
            '/dmp/command',
            self.command_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Services
        self.create_service(
            DMPService,
            '/dmp/create_dmp',
            self.create_dmp_callback,
            callback_group=self.callback_group
        )
        
        self.create_service(
            TrajectoryService,
            '/dmp/learn_trajectory',
            self.learn_trajectory_callback,
            callback_group=self.callback_group
        )
        
        self.create_service(
            Trigger,
            '/dmp/execute',
            self.execute_callback,
            callback_group=self.callback_group
        )
        
        self.create_service(
            SetBool,
            '/dmp/pause',
            self.pause_callback,
            callback_group=self.callback_group
        )
        
        # Timer for status publishing
        self.create_timer(
            1.0,
            self.publish_status,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('DMP Node initialized successfully')
        self.publish_status_message('DMP Node ready')

    def create_dmp_callback(self, request, response):
        """Service callback to create a new DMP instance"""
        try:
            dmp_id = request.dmp_id
            dmp_type = request.dmp_type if request.dmp_type else self.dmp_type
            n_dmps = request.n_dmps if request.n_dmps > 0 else self.n_dmps
            n_bfs = request.n_bfs if request.n_bfs > 0 else self.n_bfs
            
            if dmp_type == 'discrete':
                dmp = DMPs_discrete(n_dmps=n_dmps, n_bfs=n_bfs, dt=self.dt)
            elif dmp_type == 'rhythmic':
                dmp = DMPs_rhythmic(n_dmps=n_dmps, n_bfs=n_bfs, dt=self.dt)
            else:
                response.success = False
                response.message = f"Unknown DMP type: {dmp_type}"
                return response
            
            self.dmp_instances[dmp_id] = {
                'dmp': dmp,
                'type': dmp_type,
                'created_at': datetime.now().isoformat(),
                'trajectory_history': []
            }
            
            response.success = True
            response.message = f"DMP '{dmp_id}' created successfully"
            self.get_logger().info(response.message)
            self.publish_status_message(response.message)
            
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(response.message)
        
        return response

    def learn_trajectory_callback(self, request, response):
        """Service callback to learn from trajectory data"""
        try:
            dmp_id = request.dmp_id
            y_desired = np.array(request.trajectory_data)
            
            if dmp_id not in self.dmp_instances:
                response.success = False
                response.message = f"DMP '{dmp_id}' not found"
                return response
            
            dmp = self.dmp_instances[dmp_id]['dmp']
            dmp.imitate_path(y_desired)
            
            self.dmp_instances[dmp_id]['trajectory_history'].append({
                'timestamp': datetime.now().isoformat(),
                'trajectory': y_desired.tolist()
            })
            
            response.success = True
            response.message = f"Trajectory learned for DMP '{dmp_id}'"
            self.get_logger().info(response.message)
            self.publish_status_message(response.message)
            
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(response.message)
        
        return response

    def execute_callback(self, request, response):
        """Service callback to execute DMP"""
        try:
            with self.trajectory_lock:
                # Get first DMP instance (can be extended for multiple)
                if not self.dmp_instances:
                    response.success = False
                    response.message = "No DMP instances available"
                    return response
                
                dmp_id = list(self.dmp_instances.keys())[0]
                dmp = self.dmp_instances[dmp_id]['dmp']
                
                y_track, dy_track, ddy_track = dmp.rollout()
                self.current_trajectory = {
                    'y': y_track,
                    'dy': dy_track,
                    'ddy': ddy_track,
                    'dmp_id': dmp_id,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Publish trajectory
                self.publish_trajectory(y_track)
                
                response.success = True
                response.message = f"Trajectory executed for DMP '{dmp_id}'"
                self.get_logger().info(response.message)
                self.publish_status_message(response.message)
            
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(response.message)
        
        return response

    def pause_callback(self, request, response):
        """Service callback to pause execution"""
        response.success = True
        response.message = "DMP execution paused" if request.data else "DMP execution resumed"
        self.get_logger().info(response.message)
        self.publish_status_message(response.message)
        return response

    def command_callback(self, msg):
        """Subscription callback for commands"""
        try:
            command = json.loads(msg.data)
            command_type = command.get('type')
            
            if command_type == 'update_params':
                self.update_dmp_parameters(command)
            elif command_type == 'set_goal':
                self.set_dmp_goal(command)
            
            self.get_logger().info(f"Command processed: {command_type}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")

    def update_dmp_parameters(self, command):
        """Update DMP parameters dynamically"""
        dmp_id = command.get('dmp_id')
        if dmp_id in self.dmp_instances:
            dmp = self.dmp_instances[dmp_id]['dmp']
            
            if 'goal' in command:
                dmp.goal = np.array(command['goal'])
            if 'y0' in command:
                dmp.y0 = np.array(command['y0'])
            
            self.publish_status_message(f"Parameters updated for DMP '{dmp_id}'")

    def set_dmp_goal(self, command):
        """Set goal position for DMP"""
        dmp_id = command.get('dmp_id')
        goal = command.get('goal')
        
        if dmp_id in self.dmp_instances:
            dmp = self.dmp_instances[dmp_id]['dmp']
            dmp.goal = np.array(goal) if isinstance(goal, list) else np.ones(dmp.n_dmps) * goal
            self.publish_status_message(f"Goal set to {goal} for DMP '{dmp_id}'")

    def publish_trajectory(self, trajectory):
        """Publish trajectory data"""
        msg = Float32MultiArray()
        msg.data = trajectory.flatten().tolist()
        self.trajectory_pub.publish(msg)

    def publish_status_message(self, message):
        """Publish status message"""
        msg = String()
        msg.data = message
        self.status_pub.publish(msg)

    def publish_status(self):
        """Publish periodic status updates"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'active_dmps': len(self.dmp_instances),
            'dmp_ids': list(self.dmp_instances.keys()),
            'current_trajectory': 'executing' if self.current_trajectory else 'idle'
        }
        self.publish_status_message(json.dumps(status))


def main(args=None):
    rclpy.init(args=args)
    dmp_node = DMPNode()
    executor = MultiThreadedExecutor()
    executor.add_node(dmp_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        dmp_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
