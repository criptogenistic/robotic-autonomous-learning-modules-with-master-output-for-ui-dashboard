"""Advanced RL learning utilities from RL Zoo and SB3 Contrib."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3, DQN
from stable_baselines3.common.utils import get_latest_run_id

try:
    from sb3_contrib import ARS, QRDQN, MaskablePPO, RecurrentPPO, TQC, TRPO, CrossQ
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    SB3_CONTRIB_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


class HyperparameterManager:
    """Manage hyperparameters using RL Zoo configuration."""

    # Default hyperparameters from RL Zoo v2.9.1
    DEFAULT_HYPERPARAMS = {
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "max_grad_norm": 0.5,
        },
        "a2c": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
        },
        "sac": {
            "learning_rate": 3e-4,
            "buffer_size": 1000000,
            "learning_starts": 10000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
        },
        "td3": {
            "learning_rate": 1e-3,
            "buffer_size": 1000000,
            "learning_starts": 10000,
            "batch_size": 100,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "policy_delay": 2,
        },
        "ddpg": {
            "learning_rate": 1e-3,
            "buffer_size": 1000000,
            "learning_starts": 10000,
            "batch_size": 100,
            "tau": 0.001,
            "gamma": 0.99,
            "train_freq": -1,
        },
        "dqn": {
            "learning_rate": 1e-4,
            "buffer_size": 1000000,
            "learning_starts": 50000,
            "batch_size": 32,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "target_update_interval": 10000,
            "exploration_fraction": 0.1,
        },
    }

    # SB3-Contrib algorithms (v2.9.0)
    if SB3_CONTRIB_AVAILABLE:
        DEFAULT_HYPERPARAMS.update({
            "ars": {
                "learning_rate": 0.02,
                "population_size": 32,
                "n_episodes_rollout": 2,
            },
            "qrdqn": {
                "learning_rate": 1e-4,
                "buffer_size": 1000000,
                "learning_starts": 50000,
                "batch_size": 32,
                "gamma": 0.99,
                "train_freq": 4,
            },
            "maskable_ppo": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
            },
            "recurrent_ppo": {
                "learning_rate": 3e-4,
                "n_steps": 512,
                "batch_size": 256,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
            },
            "tqc": {
                "learning_rate": 3e-4,
                "buffer_size": 1000000,
                "learning_starts": 10000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
            },
            "trpo": {
                "learning_rate": 1e-3,
                "n_steps": 2048,
                "batch_size": 128,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "cg_max_steps": 15,
                "line_search_coef": 0.5,
            },
            "crossq": {
                "learning_rate": 3e-4,
                "buffer_size": 1000000,
                "learning_starts": 10000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
            },
        })

    def __init__(self, hyperparams_file: Optional[str] = None):
        """Initialize hyperparameter manager.

        Args:
            hyperparams_file: Path to custom hyperparameters JSON/YAML
        """
        self.hyperparams = self.DEFAULT_HYPERPARAMS.copy()
        if hyperparams_file and os.path.exists(hyperparams_file):
            self._load_hyperparams(hyperparams_file)

    def _load_hyperparams(self, path: str) -> None:
        """Load hyperparameters from file.

        Args:
            path: File path (JSON or YAML)
        """
        try:
            with open(path, "r") as f:
                if path.endswith(".json"):
                    custom = json.load(f)
                else:
                    import yaml
                    custom = yaml.safe_load(f)
                self.hyperparams.update(custom)
                logger.info(f"Loaded custom hyperparameters from {path}")
        except Exception as e:
            logger.warning(f"Failed to load hyperparameters: {e}")

    def get_hyperparams(
        self, algorithm: str, env_id: str = None
    ) -> Dict[str, Any]:
        """Get hyperparameters for algorithm and environment.

        Args:
            algorithm: Algorithm name
            env_id: Optional environment ID

        Returns:
            Hyperparameters dictionary
        """
        algo_lower = algorithm.lower()
        
        if algo_lower in self.hyperparams:
            return self.hyperparams[algo_lower].copy()
        
        logger.warning(f"No hyperparameters found for {algorithm}")
        return {}

    def suggest_algorithm(
        self, env_type: str
    ) -> List[str]:
        """Suggest best algorithms for environment type.

        Args:
            env_type: Environment type (continuous, discrete, atari, etc.)

        Returns:
            List of recommended algorithms
        """
        recommendations = {
            "continuous": ["PPO", "SAC", "TD3", "DDPG"],
            "discrete": ["PPO", "DQN", "A2C"],
            "atari": ["PPO", "DQN", "A2C"],
            "robotics": ["PPO", "SAC", "TD3"],
        }
        return recommendations.get(env_type, ["PPO", "A2C"])


class AdvancedLearner:
    """Advanced learner using RL Zoo and SB3 Contrib algorithms."""

    ALGORITHM_MAP = {
        "ppo": PPO,
        "a2c": A2C,
        "ddpg": DDPG,
        "sac": SAC,
        "td3": TD3,
        "dqn": DQN,
    }

    if SB3_CONTRIB_AVAILABLE:
        ALGORITHM_MAP.update({
            "ars": ARS,
            "qrdqn": QRDQN,
            "maskable_ppo": MaskablePPO,
            "recurrent_ppo": RecurrentPPO,
            "tqc": TQC,
            "trpo": TRPO,
            "crossq": CrossQ,
        })

    def __init__(
        self,
        environment: Any,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        hyperparams_file: Optional[str] = None,
        tensorboard_log: Optional[str] = None,
    ):
        """Initialize advanced learner.

        Args:
            environment: Gymnasium environment
            algorithm: Algorithm name
            policy: Policy network type
            hyperparams_file: Path to custom hyperparameters
            tensorboard_log: Tensorboard log directory
        """
        self.environment = environment
        self.algorithm_name = algorithm.lower()
        self.policy = policy
        self.tensorboard_log = tensorboard_log

        # Initialize hyperparameter manager
        self.hyperparam_manager = HyperparameterManager(hyperparams_file)

        # Get hyperparameters
        self.hyperparams = self.hyperparam_manager.get_hyperparams(
            self.algorithm_name
        )

        # Create model
        self.model = None
        self._create_model()

        # Training metrics
        self.training_steps = 0
        self.episode_count = 0
        self.best_reward = -np.inf

    def _create_model(self) -> None:
        """Create the RL model using specified algorithm."""
        if self.algorithm_name not in self.ALGORITHM_MAP:
            raise ValueError(
                f"Algorithm {self.algorithm_name} not supported. "
                f"Available: {list(self.ALGORITHM_MAP.keys())}"
            )

        AlgorithmClass = self.ALGORITHM_MAP[self.algorithm_name]

        model_kwargs = self.hyperparams.copy()
        model_kwargs["tensorboard_log"] = self.tensorboard_log

        self.model = AlgorithmClass(
            self.policy,
            self.environment,
            **model_kwargs,
        )
        logger.info(
            f"Created {self.algorithm_name.upper()} agent with {self.policy} policy"
        )

    def learn(
        self,
        total_timesteps: int = 100000,
        callback=None,
        log_interval: int = 4,
        eval_env=None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
    ) -> Dict[str, float]:
        """Train the agent.

        Args:
            total_timesteps: Total training steps
            callback: Custom callback
            log_interval: Logging interval
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes

        Returns:
            Training metrics
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
        )
        self.training_steps += total_timesteps

        # Evaluate if eval_env provided
        metrics = {"total_steps": self.training_steps}
        if eval_env:
            mean_reward, std_reward = self._evaluate(
                eval_env, n_eval_episodes
            )
            metrics["eval_reward_mean"] = mean_reward
            metrics["eval_reward_std"] = std_reward
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                metrics["best_reward"] = self.best_reward

        return metrics

    def _evaluate(
        self,
        eval_env: Any,
        n_episodes: int = 5,
    ) -> Tuple[float, float]:
        """Evaluate agent performance.

        Args:
            eval_env: Evaluation environment
            n_episodes: Number of episodes

        Returns:
            Mean reward and standard deviation
        """
        rewards = []
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            rewards.append(episode_reward)

        return float(np.mean(rewards)), float(np.std(rewards))

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        state: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Predict action.

        Args:
            observation: Observation from environment
            deterministic: Use deterministic policy
            state: RNN state for recurrent policies

        Returns:
            Action and state
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model.predict(observation, deterministic=deterministic, state=state)

    def save(self, path: str) -> None:
        """Save model and config.

        Args:
            path: Save path
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        self.model.save(path)
        
        # Save configuration
        config = {
            "algorithm": self.algorithm_name,
            "policy": self.policy,
            "hyperparameters": self.hyperparams,
            "training_steps": self.training_steps,
            "best_reward": float(self.best_reward),
        }
        config_path = f"{path}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model and config saved to {path}")

    def load(self, path: str, environment: Optional[Any] = None) -> None:
        """Load model and config.

        Args:
            path: Load path
            environment: Optional environment
        """
        if self.algorithm_name not in self.ALGORITHM_MAP:
            raise ValueError(f"Algorithm {self.algorithm_name} not supported")

        AlgorithmClass = self.ALGORITHM_MAP[self.algorithm_name]
        self.model = AlgorithmClass.load(path, env=environment or self.environment)

        # Load configuration
        config_path = f"{path}.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self.training_steps = config.get("training_steps", 0)
                self.best_reward = config.get("best_reward", -np.inf)

        logger.info(f"Model and config loaded from {path}")

    def get_info(self) -> Dict[str, Any]:
        """Get learner information.

        Returns:
            Information dictionary
        """
        return {
            "algorithm": self.algorithm_name,
            "policy": self.policy,
            "training_steps": self.training_steps,
            "best_reward": float(self.best_reward) if self.best_reward > -np.inf else None,
            "hyperparameters": self.hyperparams,
        }
