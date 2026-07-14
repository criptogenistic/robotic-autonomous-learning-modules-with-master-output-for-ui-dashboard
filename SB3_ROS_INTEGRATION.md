# Stable Baselines3 Integration with ROS UI

Detailed integration guide for Stable Baselines3 with ROS ecosystem support.

## Overview

This integration provides:
- Full Stable Baselines3 (PPO, A2C, DDPG, SAC, TD3, DQN) support
- ROS bridge for topic publishing/subscribing
- Real-time UI dashboard synchronized with training
- Voice command integration for dynamic agent control

## Components

### 1. RLAgent (`rl_agent.py`)

Wrapper for all SB3 algorithms:

```python
from src.rl_agent import RLAgent

agent = RLAgent(
    environment=env,
    algorithm="PPO",
    policy="MlpPolicy",
    learning_rate=3e-4,
)

# Train
agent.learn(total_timesteps=10000)

# Predict
action, _ = agent.predict(observation, deterministic=True)

# Save/Load
agent.save("model.zip")
agent.load("model.zip")
```

### 2. ROS Integration (`ros_integration.py`)

#### ROSBridge

Publish/subscribe to ROS topics:

```python
from src.ros_integration import ROSBridge

ros_bridge = ROSBridge(namespace="/robotic_learning")

# Subscribe
ros_bridge.subscribe("sensor_data", "Float32", callback)

# Publish
ros_bridge.publish("agent_action", {"action": [0.5, -0.2]})
```

#### ROSUIAdapter

Sync UI state with ROS topics:

```python
from src.ros_integration import ROSUIAdapter

ros_adapter = ROSUIAdapter(ros_bridge)

# Update state
ros_adapter.update_state(observation, action, reward, info)

# Publish metrics
ros_adapter.publish_metrics({
    "episode": 42,
    "reward": 150.5,
    "success_rate": 0.85,
})
```

### 3. Integrated Training (`sb3_training_loop.py`)

Combines everything:

```python
from src.sb3_training_loop import SB3RLTrainingLoop

training = SB3RLTrainingLoop(
    environment=env,
    algorithm="PPO",
    dashboard_port=8000,
    enable_ros=True,
)

training.train(
    total_timesteps=100000,
    run_dashboard=True,
)

training.save_model("trained_agent.zip")
```

## Algorithm Selection

### Policy Gradient Methods
- **PPO (Proximal Policy Optimization)**: Recommended for robotics, good performance and stability
- **A2C (Advantage Actor-Critic)**: Faster, good for continuous control
- **TRPO**: Similar to PPO, more conservative

### Off-Policy Methods
- **SAC (Soft Actor-Critic)**: Excellent for continuous control, sample efficient
- **TD3 (Twin Delayed DDPG)**: Improved DDPG, better for continuous control
- **DDPG**: Legacy continuous control algorithm

### Value-Based Methods
- **DQN**: For discrete action spaces

## ROS Topic Mapping

### Published Topics
```
/robotic_learning/ui/state        # Current state observation/action/reward
/robotic_learning/metrics         # Training metrics
/robotic_learning/errors          # Error events
```

### Subscribed Topics
```
/robotic_learning/commands        # External commands
/robotic_learning/sensor_data     # Sensor inputs (custom)
```

## Training Example

```python
import gymnasium as gym
from src.sb3_training_loop import SB3RLTrainingLoop

# Create environment
env = gym.make("CartPole-v1")

# Initialize training with all integrations
training = SB3RLTrainingLoop(
    environment=env,
    algorithm="PPO",
    policy="MlpPolicy",
    dashboard_port=8000,
    enable_ros=True,
    ros_namespace="/cartpole_learning",
)

# Train
training.train(
    total_timesteps=100000,
    run_dashboard=True,
    dashboard_thread=True,
)

# Access results
print(f"Episodes: {training.episode_count}")
print(f"Steps: {training.total_steps}")

# Save model
training.save_model("cartpole_ppo.zip")

# Later: Load and continue training
training.load_model("cartpole_ppo.zip")
training.train(total_timesteps=50000)

training.close()
```

## Dashboard Integration

The dashboard provides:
- Real-time training metrics
- Episode statistics
- Action/observation visualization
- Voice command input
- Model save/load controls
- ROS topic monitoring

Access at: `http://localhost:8000`

## Voice Command Processing

Voice commands are propagated through:
1. Dashboard captures voice input
2. Publishes to `/robotic_learning/commands`
3. ROS adapter receives and processes
4. Environment receives modified action/reward

## Performance Tips

1. **Algorithm Selection**:
   - Use PPO for most robotics tasks
   - Use SAC/TD3 for continuous control with high precision
   - Use DQN for discrete action spaces

2. **Hyperparameters**:
   ```python
   learning_rate=3e-4       # Standard for PPO
   n_steps=2048            # PPO rollout buffer size
   batch_size=64           # Batch size for updates
   n_epochs=10             # Training epochs per rollout
   ```

3. **Vectorized Environments**:
   ```python
   from stable_baselines3.common.vec_env import SubprocVecEnv
   
   env = SubprocVecEnv([make_env for _ in range(4)])
   ```

## Advanced Features

### Custom Callbacks

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Called at each training step
        if self.num_timesteps % 1000 == 0:
            print(f"Step: {self.num_timesteps}")
        return True

callback = CustomCallback()
agent.learn(total_timesteps=10000, callback=callback)
```

### Monitoring

```python
from stable_baselines3.common.monitor import Monitor

env = gym.make("CartPole-v1")
env = Monitor(env, filename="logs/cartpole")
```

## Troubleshooting

### Model not learning
- Check learning rate (try 1e-4 to 1e-3)
- Verify environment reward signal
- Check observation space normalization

### ROS connection issues
- Verify ROS is running
- Check namespace configuration
- Review ros_bridge.py for topic names

### Dashboard not updating
- Check WebSocket connection
- Verify dashboard is running on correct port
- Check browser console for errors

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [ROS Documentation](https://wiki.ros.org/)
