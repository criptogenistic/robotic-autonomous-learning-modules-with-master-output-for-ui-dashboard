#!/usr/bin/env python3
"""
Web-based UI Dashboard for PyDMPs-ROS2 Integration
Provides real-time control and visualization of DMP parameters and trajectories
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import threading
import numpy as np
from datetime import datetime
import os
from pathlib import Path

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global state
app.config['DMP_STATE'] = {
    'instances': {},
    'current_trajectory': None,
    'status_log': [],
    'parameters': {},
    'execution_state': 'idle'
}

app.config['ROS_INITIALIZED'] = False
app.config['ROS_NODE'] = None


class ROSBridge(Node):
    """Bridge between Flask and ROS2"""
    
    def __init__(self, app_config):
        super().__init__('ui_bridge_node')
        self.app_config = app_config
        
        # Subscribers
        self.trajectory_sub = self.create_subscription(
            Float32MultiArray,
            '/dmp/trajectory',
            self.trajectory_callback,
            10
        )
        
        self.status_sub = self.create_subscription(
            String,
            '/dmp/status',
            self.status_callback,
            10
        )
        
        self.get_logger().info('ROS Bridge initialized')
    
    def trajectory_callback(self, msg):
        """Callback for trajectory messages"""
        self.app_config['current_trajectory'] = {
            'data': msg.data,
            'timestamp': datetime.now().isoformat()
        }
    
    def status_callback(self, msg):
        """Callback for status messages"""
        try:
            status_data = json.loads(msg.data)
        except:
            status_data = {'message': msg.data}
        
        status_entry = {
            'timestamp': datetime.now().isoformat(),
            'data': status_data
        }
        self.app_config['status_log'].append(status_entry)
        
        # Keep only last 100 status messages
        if len(self.app_config['status_log']) > 100:
            self.app_config['status_log'].pop(0)


def init_ros():
    """Initialize ROS2 node"""
    if not app.config['ROS_INITIALIZED']:
        rclpy.init()
        app.config['ROS_NODE'] = ROSBridge(app.config['DMP_STATE'])
        app.config['ROS_INITIALIZED'] = True
        
        # Start ROS spinning in background thread
        def ros_spin():
            while app.config['ROS_INITIALIZED']:
                rclpy.spin_once(app.config['ROS_NODE'], timeout_sec=0.1)
        
        ros_thread = threading.Thread(target=ros_spin, daemon=True)
        ros_thread.start()


# Routes

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'instances': app.config['DMP_STATE']['instances'],
        'execution_state': app.config['DMP_STATE']['execution_state'],
        'recent_status': app.config['DMP_STATE']['status_log'][-5:] if app.config['DMP_STATE']['status_log'] else []
    })


@app.route('/api/trajectory')
def get_trajectory():
    """Get current trajectory"""
    trajectory = app.config['DMP_STATE']['current_trajectory']
    if trajectory:
        return jsonify({
            'success': True,
            'trajectory': trajectory['data'][:100],  # Return first 100 points for visualization
            'timestamp': trajectory['timestamp']
        })
    return jsonify({'success': False, 'message': 'No trajectory available'})


@app.route('/api/dmp/create', methods=['POST'])
def create_dmp():
    """Create a new DMP instance"""
    data = request.json
    dmp_id = data.get('dmp_id', f"dmp_{len(app.config['DMP_STATE']['instances'])}")
    dmp_type = data.get('type', 'discrete')
    n_dmps = data.get('n_dmps', 3)
    n_bfs = data.get('n_bfs', 10)
    
    app.config['DMP_STATE']['instances'][dmp_id] = {
        'type': dmp_type,
        'n_dmps': n_dmps,
        'n_bfs': n_bfs,
        'created_at': datetime.now().isoformat(),
        'parameters': {
            'goal': data.get('goal', 1.0),
            'y0': data.get('y0', 0.0),
            'dt': data.get('dt', 0.01)
        }
    }
    
    app.config['DMP_STATE']['status_log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': f"DMP instance '{dmp_id}' created"
    })
    
    return jsonify({
        'success': True,
        'dmp_id': dmp_id,
        'message': f"DMP '{dmp_id}' created successfully"
    })


@app.route('/api/dmp/<dmp_id>/parameters', methods=['GET', 'PUT'])
def manage_parameters(dmp_id):
    """Get or update DMP parameters"""
    if dmp_id not in app.config['DMP_STATE']['instances']:
        return jsonify({'success': False, 'message': 'DMP not found'}), 404
    
    if request.method == 'GET':
        params = app.config['DMP_STATE']['instances'][dmp_id]['parameters']
        return jsonify({'success': True, 'parameters': params})
    
    elif request.method == 'PUT':
        data = request.json
        app.config['DMP_STATE']['instances'][dmp_id]['parameters'].update(data)
        
        app.config['DMP_STATE']['status_log'].append({
            'timestamp': datetime.now().isoformat(),
            'message': f"Parameters updated for DMP '{dmp_id}'"
        })
        
        return jsonify({
            'success': True,
            'message': 'Parameters updated',
            'parameters': app.config['DMP_STATE']['instances'][dmp_id]['parameters']
        })


@app.route('/api/dmp/<dmp_id>/execute', methods=['POST'])
def execute_dmp(dmp_id):
    """Execute a DMP instance"""
    if dmp_id not in app.config['DMP_STATE']['instances']:
        return jsonify({'success': False, 'message': 'DMP not found'}), 404
    
    app.config['DMP_STATE']['execution_state'] = 'executing'
    
    app.config['DMP_STATE']['status_log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': f"DMP '{dmp_id}' execution started"
    })
    
    return jsonify({
        'success': True,
        'message': f"DMP '{dmp_id}' execution started",
        'dmp_id': dmp_id
    })


@app.route('/api/dmp/<dmp_id>/pause', methods=['POST'])
def pause_dmp(dmp_id):
    """Pause DMP execution"""
    app.config['DMP_STATE']['execution_state'] = 'paused'
    
    app.config['DMP_STATE']['status_log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': f"DMP '{dmp_id}' execution paused"
    })
    
    return jsonify({
        'success': True,
        'message': 'DMP execution paused'
    })


@app.route('/api/dmp/<dmp_id>/delete', methods=['DELETE'])
def delete_dmp(dmp_id):
    """Delete a DMP instance"""
    if dmp_id in app.config['DMP_STATE']['instances']:
        del app.config['DMP_STATE']['instances'][dmp_id]
        
        app.config['DMP_STATE']['status_log'].append({
            'timestamp': datetime.now().isoformat(),
            'message': f"DMP '{dmp_id}' deleted"
        })
        
        return jsonify({
            'success': True,
            'message': f"DMP '{dmp_id}' deleted"
        })
    
    return jsonify({'success': False, 'message': 'DMP not found'}), 404


@app.route('/api/dmp/list')
def list_dmps():
    """List all DMP instances"""
    return jsonify({
        'success': True,
        'instances': app.config['DMP_STATE']['instances']
    })


@app.route('/api/logs')
def get_logs():
    """Get status logs"""
    return jsonify({
        'success': True,
        'logs': app.config['DMP_STATE']['status_log'][-50:]
    })


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ros_initialized': app.config['ROS_INITIALIZED']
    })


if __name__ == '__main__':
    init_ros()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
