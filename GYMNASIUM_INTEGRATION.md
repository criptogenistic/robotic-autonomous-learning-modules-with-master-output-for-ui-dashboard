# Gymnasium v1.3.0 Integration Guide

## Overview

This document outlines the integration of Gymnasium v1.3.0 with the Robotic Autonomous Learning Modules and UI Dashboard.

## Key Integration Points

### 1. Environment API

The `RoboticEnvironment` class extends `gymnasium.Env` and implements the required interface:

```python
class RoboticEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        # Environment-specific reset logic
        pass

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Environment step: obs, reward, terminated, truncated, info
        pass

    def render(self) -> Optional[np.ndarray]:
        # Rendering support
        pass
```

### 2. Action and Observation Spaces

**Action Space** (Continuous):
- Type: `Box`
- Shape: (6,) - 6 robotic joints
- Range: [-1.0, 1.0]

**Observation Space**:
- Type: `Box`
- Shape: (18,) - Position (6), Velocity (6), Acceleration (6)
- Range: [-inf, inf]

### 3. Wrappers and Utilities

Utilize Gymnasium wrappers for enhanced functionality:

```python
from gymnasium.wrappers import TimeLimit, RecordVideo

env = RoboticEnvironment()
env = TimeLimit(env, max_episode_steps=1000)
env = RecordVideo(env, video_folder="videos")
```

### 4. Vector Environments

For parallel training:

```python
from gymnasium.vector import AsyncVectorEnv

def make_env():
    return RoboticEnvironment()

env = AsyncVectorEnv([make_env for _ in range(4)])
```

## UI Dashboard Integration

### Real-time State Updates

The UISync layer bridges environment state and the dashboard:

```python
# In training loop
obs, reward, terminated, truncated, info = env.step(action)
ui_sync.update_state(obs, reward, info)
```

### WebSocket Broadcasting

State updates are broadcast to connected clients:

```python
# Dashboard receives updates via WebSocket
{
    "event": "state_update",
    "observation": [0.1, 0.2, ...],
    "reward": 0.5,
    "info": {"step": 42},
    "timestamp": "2026-07-14T12:30:45.123456"
}
```

## Voice Command Integration

### Setting Voice Commands

```python
env.set_voice_command("move forward")
# Environment behavior is modified based on voice command
```

### Voice Command Processing

```python
# In training loop, voice commands are propagated
if info.get("voice_controlled"):
    voice_cmd = info.get("voice_command")
    # Adjust action or reward based on voice command
```

## Performance Optimizations

### 1. Lazy Evaluation (Gymnasium v1.3.0 feature)

The `Box` space now uses lazy evaluation:

```python
action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(6,),
    dtype=np.float32
)
# Reduced initialization overhead
```

### 2. Frame Skipping with RepeatAction

```python
from gymnasium.wrappers import RepeatAction

env = RepeatAction(env, repeat_count=4)
# Action repeated 4 times, reduces computation
```

### 3. pygame-ce Integration

Replaces `pygame` with `pygame-ce` for:
- Python 3.14 compatibility
- Better performance
- Drop-in replacement

```python
# Rendering works seamlessly with pygame-ce
env.render_mode = "rgb_array"
frame = env.render()
```

## Testing

### Unit Tests

```python
import gymnasium as gym
from src.environment import RoboticEnvironment

def test_environment():
    env = RoboticEnvironment()
    obs, info = env.reset()
    assert obs in env.observation_space

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs in env.observation_space
```

## Compatibility

- **Gymnasium**: >= 1.3.0
- **Python**: >= 3.10
- **NumPy**: >= 1.21.0
- **pygame-ce**: >= 2.1.3

## Migration from OpenAI Gym

If migrating from older Gym versions:

```python
# Old Gym API
done = terminated or truncated  # Combine boolean flags

# New Gymnasium API
obs, reward, terminated, truncated, info = env.step(action)
```

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Gymnasium v1.3.0 Release Notes](https://github.com/Farama-Foundation/Gymnasium/releases/tag/v1.3.0)
- [pygame-ce Documentation](https://pyga.me/)
