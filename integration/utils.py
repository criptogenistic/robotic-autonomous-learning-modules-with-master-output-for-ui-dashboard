#!/usr/bin/env python3
"""
Utility functions for PyDMPs-ROS2 integration
"""

import numpy as np
import json
import yaml
from typing import Any, Dict, List
from pathlib import Path
import logging
from datetime import datetime


class ConfigManager:
    """Manages configuration loading and saving"""
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict:
        """Load YAML configuration file"""
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading YAML config: {e}")
            return {}
    
    @staticmethod
    def save_yaml(data: Dict, filepath: str) -> bool:
        """Save configuration to YAML file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                yaml.dump(data, f)
            return True
        except Exception as e:
            logging.error(f"Error saving YAML config: {e}")
            return False
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load JSON configuration file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading JSON config: {e}")
            return {}
    
    @staticmethod
    def save_json(data: Dict, filepath: str) -> bool:
        """Save configuration to JSON file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving JSON config: {e}")
            return False


class Logger:
    """Custom logging utility"""
    
    def __init__(self, name: str, log_file: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)


class PerformanceMonitor:
    """Monitors and reports system performance"""
    
    def __init__(self):
        self.timings = {}
        self.counters = {}
    
    def start_timer(self, name: str):
        """Start a timer"""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append({'start': datetime.now()})
    
    def end_timer(self, name: str):
        """End a timer and record duration"""
        if name in self.timings and self.timings[name]:
            start_time = self.timings[name][-1]['start']
            duration = (datetime.now() - start_time).total_seconds()
            self.timings[name][-1]['duration'] = duration
    
    def increment_counter(self, name: str):
        """Increment a counter"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += 1
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        
        # Timing stats
        for name, timers in self.timings.items():
            durations = [t['duration'] for t in timers if 'duration' in t]
            if durations:
                stats[f"{name}_avg"] = np.mean(durations)
                stats[f"{name}_min"] = np.min(durations)
                stats[f"{name}_max"] = np.max(durations)
        
        # Counter stats
        for name, count in self.counters.items():
            stats[f"{name}_count"] = count
        
        return stats


class DataValidator:
    """Validates input data for DMPs"""
    
    @staticmethod
    def validate_trajectory(trajectory: np.ndarray, min_length: int = 10) -> bool:
        """Validate trajectory data"""
        if not isinstance(trajectory, np.ndarray):
            return False
        if len(trajectory) < min_length:
            return False
        if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
            return False
        return True
    
    @staticmethod
    def validate_parameters(params: Dict) -> bool:
        """Validate DMP parameters"""
        required = ['n_dmps', 'n_bfs']
        for req in required:
            if req not in params or params[req] <= 0:
                return False
        return True
    
    @staticmethod
    def sanitize_trajectory(trajectory: np.ndarray, remove_nan: bool = True) -> np.ndarray:
        """Sanitize trajectory data"""
        if remove_nan:
            trajectory = trajectory[~np.isnan(trajectory)]
        trajectory = np.clip(trajectory, -1e6, 1e6)  # Clip extreme values
        return trajectory
