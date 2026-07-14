# RL Zoo and SB3 Contrib Learning Integration

Integration of RL Baselines3 Zoo v2.9.1 and Stable Baselines3 Contrib v2.9.0 with the UI dashboard for comprehensive RL training and hyperparameter optimization.

## Overview

This integration provides:
- **RL Zoo v2.9.1**: Hyperparameter database for 200+ trained agents
- **SB3 Contrib v2.9.0**: Experimental algorithms (ARS, QR-DQN, Maskable PPO, Recurrent PPO, TQC, TRPO, CrossQ)
- **Learning UI Panel**: Configure and execute training from dashboard
- **Hyperparameter Management**: Automatic suggestions based on environment type
- **Training Monitoring**: Real-time metrics and evaluation

## Components

### 1. HyperparameterManager (`advanced_learner.py`)

Manages hyperparameters from RL Zoo and provides algorithm recommendations:

```python
from src.advanced_learner import HyperparameterManager

manager = HyperparameterManager()

# Get hyperparameters for algorithm
params = manager.get_hyperparams("PPO")

# Get algorithm suggestions
suggested = manager.suggest_algorithm("continuous")  # → ["PPO", "SAC", "TD3", "DDPG"]

# Load custom hyperparameters
manager = HyperparameterManager("custom_hyperparams.json")
```

### 2. AdvancedLearner (`advanced_learner.py`)

Unified interface for all SB3 and SB3-Contrib algorithms:

```python
from src.advanced_learner import AdvancedLearner

learner = AdvancedLearner(
    environment=env,
    algorithm="PPO",  # or any from RL Zoo
    policy="MlpPolicy",
    tensorboard_log="logs/",
)

# Train with RL Zoo hyperparameters
metrics = learner.learn(
    total_timesteps=100000,
    eval_env=eval_env,
    eval_freq=10000,
    n_eval_episodes=5,
)

# Save/Load
learner.save("model.zip")
learner.load("model.zip")
```

### 3. LearningRouter (`learning_ui_router.py`)

FastAPI routes for learning control:

**Endpoints**:
- `GET /api/learning/algorithms` - Get available algorithms
- `POST /api/learning/suggest-algorithm` - Suggest algorithms
- `POST /api/learning/hyperparameters` - Get hyperparameters
- `POST /api/learning/train` - Start training
- `GET /api/learning/info` - Get learner info
- `POST /api/learning/save` - Save model
- `POST /api/learning/load` - Load model

### 4. EnhancedDashboard (`enhanced_dashboard.py`)

Dashboard with integrated learning panel:

```python
from src.enhanced_dashboard import EnhancedDashboard
from src.advanced_learner import AdvancedLearner

learner = AdvancedLearner(env, algorithm="PPO")
dashboard = EnhancedDashboard(advanced_learner=learner)
dashboard.run()
```

## Supported Algorithms

### Stable Baselines3 (Core)
- PPO - Proximal Policy Optimization
- A2C - Advantage Actor-Critic
- DDPG - Deep Deterministic Policy Gradient
- SAC - Soft Actor-Critic
- TD3 - Twin Delayed DDPG
- DQN - Deep Q-Network

### Stable Baselines3 Contrib (Experimental)
- ARS - Augmented Random Search
- QR-DQN - Quantile Regression DQN
- Maskable PPO - PPO with invalid action masking
- Recurrent PPO - PPO with LSTM policy
- TQC - Truncated Quantile Critics
- TRPO - Trust Region Policy Optimization
- CrossQ - Batch Normalization in Deep RL

## Default Hyperparameters

### From RL Zoo v2.9.1

**PPO**:
```python
{
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}
```

**SAC**:
```python
{
    "learning_rate": 3e-4,
    "buffer_size": 1000000,
    "learning_starts": 10000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
}
```

**TD3**:
```python
{
    "learning_rate": 1e-3,
    "buffer_size": 1000000,
    "learning_starts": 10000,
    "batch_size": 100,
    "tau": 0.005,
    "gamma": 0.99,
    "policy_delay": 2,
}
```

### From SB3 Contrib v2.9.0

**Recurrent PPO** (for sequential data):
```python
{
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
}
```

**TQC** (improved SAC):
```python
{
    "learning_rate": 3e-4,
    "buffer_size": 1000000,
    "learning_starts": 10000,
    "batch_size": 256,
    "tau": 0.005,
}
```

## Usage Example

```python
import gymnasium as gym
from src.advanced_learner import AdvancedLearner, HyperparameterManager
from src.enhanced_dashboard import EnhancedDashboard

# Create environment
env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

# Get hyperparameter suggestions
manager = HyperparameterManager()
suggested = manager.suggest_algorithm("discrete")
print(f"Suggested: {suggested}")

# Create learner with RL Zoo hyperparameters
learner = AdvancedLearner(
    environment=env,
    algorithm="PPO",
    policy="MlpPolicy",
)

# Create dashboard
dashboard = EnhancedDashboard(advanced_learner=learner)

# Train (from code or via learning UI)
metrics = learner.learn(
    total_timesteps=100000,
    eval_env=eval_env,
    eval_freq=10000,
    n_eval_episodes=5,
)

# Save
learner.save("trained_models/cartpole")

print(f"Best reward: {learner.best_reward:.2f}")
```

## Learning UI Panel

Access the learning panel at `http://localhost:8000`:

1. **Algorithm Selection**: Choose from 13+ algorithms
2. **Hyperparameter Configuration**: Pre-tuned defaults from RL Zoo
3. **Training Control**: Start, stop, and monitor training
4. **Evaluation**: Built-in evaluation with configurable frequency
5. **Model Management**: Save/load trained models
6. **Metrics Dashboard**: Real-time training statistics
7. **Recommendations**: Algorithm suggestions based on environment

## API Usage

```bash
# Get available algorithms
curl http://localhost:8000/api/learning/algorithms

# Get hyperparameters for PPO
curl -X POST http://localhost:8000/api/learning/hyperparameters \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "ppo"}'

# Start training
curl -X POST http://localhost:8000/api/learning/train \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "ppo",
    "policy": "MlpPolicy",
    "total_timesteps": 100000
  }'

# Get learner info
curl http://localhost:8000/api/learning/info
```

## Environment Type Recommendations

- **Continuous Control**: PPO, SAC, TD3, TQC (RL Zoo tuned)
- **Discrete Actions**: PPO, DQN, QR-DQN, A2C
- **Atari Games**: PPO, DQN, A2C
- **Robotics**: PPO, SAC, TD3, HER+TQC
- **Sequential Data**: Recurrent PPO
- **Exploration-Heavy**: ARS
- **Action Masking**: Maskable PPO

## Performance Considerations

1. **RL Zoo Hyperparameters**: Already tuned for common environments
2. **Tensorboard Logging**: Automatically logs to `logs/` directory
3. **Evaluation**: Periodic evaluation prevents overfitting
4. **Model Checkpointing**: Save best models automatically
5. **Vectorized Environments**: Supported for faster training

## Troubleshooting

### Algorithm not available
- Install: `pip install sb3-contrib`
- Check: `AdvancedLearner.ALGORITHM_MAP`

### Hyperparameters not loading
- Verify JSON/YAML format
- Check file path
- Use `HyperparameterManager` for validation

### Training not improving
- Try different algorithm recommendations
- Increase `n_eval_episodes`
- Check environment reward scaling
- Review hyperparameters in logs

## References

- [RL Baselines3 Zoo v2.9.1](https://github.com/DLR-RM/rl-baselines3-zoo/releases/tag/v2.9.1)
- [SB3 Contrib v2.9.0](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/releases/tag/v2.9.0)
- [RL Zoo Documentation](https://rl-baselines3-zoo.readthedocs.io/)
- [SB3 Contrib Documentation](https://sb3-contrib.readthedocs.io/)
