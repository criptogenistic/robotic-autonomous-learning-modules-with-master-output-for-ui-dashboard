# PyDMPs-ROS2 Integration Framework

## Overview
This framework integrates Dynamic Movement Primitives (PyDMPs) with ROS2 to enable autonomous learning modules with a master UI dashboard for real-time monitoring and parameter modification.

## Architecture

### Components
1. **ROS2 DMP Node** - Wraps PyDMPs functionality
2. **UI Dashboard** - Web-based interface for parameter control
3. **ROS2-UI Bridge** - Real-time communication layer
4. **Trajectory Manager** - Handles DMP execution and publishing
5. **Parameter Server** - Central configuration management

### Features
- Real-time parameter modification
- Live trajectory visualization
- Multiple DMP instance management
- ROS2 topic publishing/subscribing
- Persistent configuration storage
- Historical data logging

## Directory Structure
```
.
├── ros2_dmp_node/          # ROS2 node implementation
├── ui_dashboard/           # Web-based dashboard
├── integration/            # Core integration modules
├── launch/                 # ROS2 launch files
├── config/                 # Configuration files
└── docs/                   # Documentation
```

## Quick Start

### Prerequisites
- ROS2 (rolling or latest stable)
- Python 3.8+
- PyDMPs
- Flask or FastAPI (for UI backend)

### Installation
```bash
# Clone and setup
cd robotic-autonomous-learning-modules-with-master-output-for-ui-dashboard

# Install dependencies
pip install pydmps flask flask-cors

# Build ROS2 packages
colcon build

# Source setup
source install/setup.bash
```

### Running the System
```bash
# Terminal 1: Start ROS2 DMP Node
ros2 launch ros2_dmp_node dmp_system.launch.py

# Terminal 2: Start UI Dashboard
python ui_dashboard/app.py

# Open browser: http://localhost:5000
```

## Dashboard Features
- **DMP Parameter Control**: Adjust n_dmps, n_bfs, goal, initial position
- **Trajectory Visualization**: Real-time plot of executed trajectories
- **Multi-Instance Management**: Control multiple DMP instances
- **Live ROS2 Integration**: Monitor topics and services
- **Data Export**: Save trajectories and logs
- **System Status**: View node health and performance metrics
