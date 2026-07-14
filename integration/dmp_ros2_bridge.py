#!/usr/bin/env python3
"""
Bridge module connecting PyDMPs with ROS2
Provides high-level abstractions and utilities
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import json
from datetime import datetime

from pydmps.dmp_discrete import DMPs_discrete
from pydmps.dmp_rhythmic import DMPs_rhythmic


class DMPType(Enum):
    """Enumeration of DMP types"""
    DISCRETE = 'discrete'
    RHYTHMIC = 'rhythmic'


@dataclass
class DMPConfig:
    """Configuration for DMP initialization"""
    dmp_type: DMPType = DMPType.DISCRETE
    n_dmps: int = 3
    n_bfs: int = 10
    dt: float = 0.01
    y0: float = 0.0
    goal: float = 1.0
    
    def to_dict(self):
        return {
            'dmp_type': self.dmp_type.value,
            'n_dmps': self.n_dmps,
            'n_bfs': self.n_bfs,
            'dt': self.dt,
            'y0': self.y0,
            'goal': self.goal
        }


class DMPManager:
    """High-level manager for DMP instances"""
    
    def __init__(self):
        self.instances: Dict[str, Dict] = {}
        self.lock = threading.RLock()
        self.callbacks: Dict[str, List[Callable]] = {
            'on_create': [],
            'on_execute': [],
            'on_error': [],
            'on_parameter_change': []
        }
    
    def create_dmp(self, dmp_id: str, config: DMPConfig) -> bool:
        """Create a new DMP instance"""
        with self.lock:
            if dmp_id in self.instances:
                return False
            
            try:
                if config.dmp_type == DMPType.DISCRETE:
                    dmp = DMPs_discrete(
                        n_dmps=config.n_dmps,
                        n_bfs=config.n_bfs,
                        dt=config.dt
                    )
                elif config.dmp_type == DMPType.RHYTHMIC:
                    dmp = DMPs_rhythmic(
                        n_dmps=config.n_dmps,
                        n_bfs=config.n_bfs,
                        dt=config.dt
                    )
                else:
                    raise ValueError(f"Unknown DMP type: {config.dmp_type}")
                
                self.instances[dmp_id] = {
                    'dmp': dmp,
                    'config': config,
                    'created_at': datetime.now().isoformat(),
                    'trajectories': [],
                    'state': 'ready'
                }
                
                self._trigger_callback('on_create', dmp_id, config)
                return True
            
            except Exception as e:
                self._trigger_callback('on_error', dmp_id, str(e))
                return False
    
    def delete_dmp(self, dmp_id: str) -> bool:
        """Delete a DMP instance"""
        with self.lock:
            if dmp_id in self.instances:
                del self.instances[dmp_id]
                return True
            return False
    
    def get_dmp(self, dmp_id: str) -> Optional[Dict]:
        """Get DMP instance data"""
        with self.lock:
            return self.instances.get(dmp_id)
    
    def update_parameters(self, dmp_id: str, **kwargs) -> bool:
        """Update DMP parameters"""
        with self.lock:
            if dmp_id not in self.instances:
                return False
            
            dmp = self.instances[dmp_id]['dmp']
            
            if 'goal' in kwargs:
                dmp.goal = np.array(kwargs['goal']) if isinstance(kwargs['goal'], list) else np.ones(dmp.n_dmps) * kwargs['goal']
            
            if 'y0' in kwargs:
                dmp.y0 = np.array(kwargs['y0']) if isinstance(kwargs['y0'], list) else np.ones(dmp.n_dmps) * kwargs['y0']
            
            self._trigger_callback('on_parameter_change', dmp_id, kwargs)
            return True
    
    def execute_dmp(self, dmp_id: str, **kwargs) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Execute DMP and return trajectory"""
        with self.lock:
            if dmp_id not in self.instances:
                return None
            
            try:
                instance = self.instances[dmp_id]
                dmp = instance['dmp']
                
                instance['state'] = 'executing'
                y_track, dy_track, ddy_track = dmp.rollout(**kwargs)
                instance['state'] = 'ready'
                
                # Store trajectory
                trajectory_data = {
                    'timestamp': datetime.now().isoformat(),
                    'y': y_track.tolist(),
                    'dy': dy_track.tolist(),
                    'ddy': ddy_track.tolist()
                }
                instance['trajectories'].append(trajectory_data)
                
                self._trigger_callback('on_execute', dmp_id, trajectory_data)
                return y_track, dy_track, ddy_track
            
            except Exception as e:
                self.instances[dmp_id]['state'] = 'error'
                self._trigger_callback('on_error', dmp_id, str(e))
                return None
    
    def learn_from_trajectory(self, dmp_id: str, y_desired: np.ndarray) -> bool:
        """Learn DMP parameters from desired trajectory"""
        with self.lock:
            if dmp_id not in self.instances:
                return False
            
            try:
                dmp = self.instances[dmp_id]['dmp']
                dmp.imitate_path(y_desired)
                return True
            
            except Exception as e:
                self._trigger_callback('on_error', dmp_id, str(e))
                return False
    
    def list_instances(self) -> Dict[str, Dict]:
        """List all DMP instances"""
        with self.lock:
            return {k: {'config': v['config'].to_dict(), 'state': v['state']} for k, v in self.instances.items()}
    
    def register_callback(self, event: str, callback: Callable) -> bool:
        """Register callback for events"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            return True
        return False
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger all callbacks for an event"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Error in callback {event}: {e}")


class TrajectoryProcessor:
    """Utilities for trajectory processing and analysis"""
    
    @staticmethod
    def smooth_trajectory(trajectory: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Smooth trajectory using moving average"""
        if len(trajectory) < window_size:
            return trajectory
        
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(trajectory, kernel, mode='same')
        return smoothed
    
    @staticmethod
    def scale_trajectory(trajectory: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """Scale trajectory to specified range"""
        if trajectory.max() == trajectory.min():
            return trajectory
        
        scaled = (trajectory - trajectory.min()) / (trajectory.max() - trajectory.min())
        scaled = scaled * (max_val - min_val) + min_val
        return scaled
    
    @staticmethod
    def compute_statistics(trajectory: np.ndarray) -> Dict:
        """Compute trajectory statistics"""
        return {
            'mean': float(np.mean(trajectory)),
            'std': float(np.std(trajectory)),
            'min': float(np.min(trajectory)),
            'max': float(np.max(trajectory)),
            'range': float(np.max(trajectory) - np.min(trajectory))
        }


class StatePublisher:
    """Manages state publishing to external systems"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, topic: str, callback: Callable) -> bool:
        """Subscribe to state updates"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        return True
    
    def publish(self, topic: str, data: any) -> bool:
        """Publish data to all subscribers"""
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error publishing to {topic}: {e}")
                    return False
        return True
