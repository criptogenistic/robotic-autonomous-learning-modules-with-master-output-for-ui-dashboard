# Robotic Autonomous Learning Modules with Gymnasium v1.3.0

A comprehensive framework for training robotic agents using reinforcement learning with Gymnasium v1.3.0 integration and real-time UI dashboard.

## Features

- **Gymnasium v1.3.0 Integration**: Fully compatible with the latest Gymnasium API
- **Real-time UI Dashboard**: WebSocket-based dashboard for monitoring and control
- **Voice Command Rerouting**: Support for voice-controlled robot behavior modification
- **Multi-Environment Support**: Compatible with Atari, Box2D, Classic Control, MuJoCo, and Toy Text environments
- **Async Architecture**: FastAPI-based server with WebSocket support for real-time updates
- **Modular Design**: Easy to extend and customize for different robotic platforms

## Installation

```bash
git clone https://github.com/criptogenistic/robotic-autonomous-learning-modules-with-master-output-for-ui-dashboard.git
cd robotic-autonomous-learning-modules-with-master-output-for-ui-dashboard
pip install -e ".[ui]"
```

For additional environment support:

```bash
pip install -e ".[atari,box2d,mujoco]"
```

## Quick Start

### Basic Usage

```python
from src.training_loop import TrainingLoop

# Create training loop
training = TrainingLoop(
    env_config={
        "motion_type": "continuous",
        "render_mode": None,
    },
    dashboard_port=8000
)

# Run training with dashboard
training.run_training(
    num_episodes=100,
    run_dashboard_thread=True
)
```

### Accessing the Dashboard

Once training starts, navigate to `http://localhost:8000` to access the real-time dashboard.

## Project Structure

```
.
├── src/
│   ├── __init__.py              # Package initialization
│   ├── environment.py           # RoboticEnvironment class
│   ├── ui_sync.py               # UI synchronization layer
│   ├── dashboard.py             # FastAPI dashboard server
│   └── training_loop.py         # Main training loop
├── tests/                       # Unit tests
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Architecture

### Components

1. **RoboticEnvironment**: Gymnasium-compatible environment wrapping robotic dynamics
   - Supports both continuous and discrete action spaces
   - Implements voice command rerouting
   - Generates observations including position, velocity, and acceleration

2. **UISync**: Synchronization layer for dashboard updates
   - Publishes environment state at regular intervals
   - Manages event subscriptions and broadcasting
   - Handles metrics and error reporting

3. **DashboardServer**: FastAPI-based web server
   - Real-time WebSocket connection for live updates
   - REST API endpoints for status and commands
   - Supports voice commands and action specifications

4. **TrainingLoop**: Main orchestration layer
   - Manages episode execution
   - Coordinates environment, UI sync, and dashboard
   - Tracks training metrics and progress

## Gymnasium v1.3.0 Features

This project leverages the following features from Gymnasium v1.3.0:

- **RepeatAction Wrapper**: Frame-skipping with action repetition
- **pygame-ce Integration**: Cross-platform rendering with Python 3.14 support
- **Vector Environments**: Efficient parallel environment execution
- **Optimized Box Space**: Reduced overhead through lazy evaluation

## Configuration

### Environment Configuration

```python
env_config = {
    "motion_type": "continuous",  # or "discrete"
    "render_mode": "rgb_array",    # "human", "rgb_array", or None
}
```

### Dashboard Configuration

```python
dashboard = DashboardServer(
    host="0.0.0.0",
    port=8000
)
```

## API Endpoints

### Health Check
- `GET /health` - Check server health

### Status
- `GET /status` - Get current environment status

### Commands
- `POST /command` - Send command to robot
  ```json
  {
    "command_type": "voice",
    "voice_command": "move forward"
  }
  ```

### WebSocket
- `WS /ws` - Real-time updates and bi-directional communication

## Development

### Running Tests

```bash
pip install -e ".[testing]"
pytest tests/
```

### Code Quality

```bash
ruff check src/
ruff format src/
```

## Performance Considerations

- UI update interval: 16ms (60 FPS)
- WebSocket broadcasting: Async with error handling
- Environment stepping: GPU-accelerated with JAX/PyTorch support
- Observation buffering: Minimal memory footprint

## Voice Command Examples

```python
environment.set_voice_command("move forward")
environment.set_voice_command("turn left")
environment.set_voice_command("speed up")
environment.set_voice_command("stop")
```

## Troubleshooting

### Dashboard not connecting
- Ensure port 8000 is not in use
- Check firewall settings
- Verify WebSocket support in your network

### Environment not responding
- Check that Gymnasium is properly installed
- Verify environment configuration
- Review error messages in dashboard console

## License

MIT License - See LICENSE file for details

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Reinforcement Learning Basics](https://en.wikipedia.org/wiki/Reinforcement_learning)
