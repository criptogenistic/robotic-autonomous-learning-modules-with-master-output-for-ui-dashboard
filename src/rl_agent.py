"""Stable Baselines3 integration with ROS support for robotic control."""

import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3, DQN
from stable_baselines3.common.base_class import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import logging

logger = logging.getLogger(__name__)


class RLAgent:
    """RL Agent wrapper supporting multiple algorithms."""

    SUPPORTED_ALGORITHMS = {
        "PPO": PPO,
        "A2C": A2C,
        "DDPG": DDPG,
        "SAC": SAC,
        "TD3": TD3,
        "DQN": DQN,
    }

    def __init__(
        self,
        environment: Any,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        verbose: int = 0,
        device: str = "auto",
        **kwargs,
    ):
        """Initialize RL agent.

        Args:
            environment: Gymnasium-compatible environment
            algorithm: Algorithm name (PPO, A2C, DDPG, SAC, TD3, DQN)
            policy: Policy network type
            learning_rate: Learning rate for optimization
            verbose: Verbosity level
            device: Device for PyTorch ("cpu", "cuda", "auto")
            **kwargs: Additional algorithm-specific parameters
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Algorithm {algorithm} not supported. "
                f"Choose from {list(self.SUPPORTED_ALGORITHMS.keys())}"
            )

        self.algorithm_name = algorithm
        self.environment = environment
        self.policy = policy
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.device = device
        self.kwargs = kwargs

        self.model = None
        self.training_steps = 0
        self.episode_count = 0

        self._create_model()

    def _create_model(self) -> None:
        """Create the RL model."""
        AlgorithmClass = self.SUPPORTED_ALGORITHMS[self.algorithm_name]
        
        # Algorithm-specific parameters
        algo_kwargs = {
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "device": self.device,
        }
        algo_kwargs.update(self.kwargs)

        self.model = AlgorithmClass(
            self.policy,
            self.environment,
            **algo_kwargs,
        )
        logger.info(f"Created {self.algorithm_name} agent")

    def learn(
        self,
        total_timesteps: int = 10000,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 4,
    ) -> None:
        """Train the agent.

        Args:
            total_timesteps: Total steps to train
            callback: Training callback
            log_interval: Logging interval
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
        )
        self.training_steps += total_timesteps
        logger.info(
            f"Training complete. Total steps: {self.training_steps}"
        )

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        state: Optional[Tuple] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """Predict action from observation.

        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            state: RNN state (for recurrent policies)

        Returns:
            Action and RNN state
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        action, state = self.model.predict(
            observation,
            deterministic=deterministic,
            state=state,
        )
        return action, state

    def save(self, path: str) -> None:
        """Save model to file.

        Args:
            path: File path to save model
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str, environment: Optional[Any] = None) -> None:
        """Load model from file.

        Args:
            path: File path to load model from
            environment: Optional environment to set
        """
        AlgorithmClass = self.SUPPORTED_ALGORITHMS[self.algorithm_name]
        self.model = AlgorithmClass.load(path, env=environment)
        logger.info(f"Model loaded from {path}")

    def get_policy(self) -> Optional[BasePolicy]:
        """Get the policy network.

        Returns:
            Policy network or None
        """
        if self.model is None:
            return None
        return self.model.policy

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update agent parameters.

        Args:
            parameters: Dictionary of parameters to update
        """
        if "learning_rate" in parameters:
            self.learning_rate = parameters["learning_rate"]
            if self.model:
                self.model.learning_rate = self.learning_rate
        if "verbose" in parameters:
            self.verbose = parameters["verbose"]
            if self.model:
                self.model.verbose = self.verbose
