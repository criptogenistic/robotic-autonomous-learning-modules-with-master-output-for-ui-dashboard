"""Integrated training loop with Stable Baselines3 and ROS UI support."""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any
from .rl_agent import RLAgent
from .ros_integration import ROSBridge, ROSUIAdapter
from .ui_sync import UISync
from .dashboard import DashboardServer
from stable_baselines3.common.callbacks import BaseCallback
import logging
import threading

logger = logging.getLogger(__name__)


class RLCallback(BaseCallback):
    """Custom callback for training with UI updates."""

    def __init__(
        self,
        ui_sync: UISync,
        ros_adapter: Optional['ROSUIAdapter'] = None,
    ):
        """Initialize callback.

        Args:
            ui_sync: UISync instance
            ros_adapter: Optional ROSUIAdapter instance
        """
        super().__init__()
        self.ui_sync = ui_sync
        self.ros_adapter = ros_adapter
        self.episode_steps = 0
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            Whether to continue training
        """
        # Get current metrics from model
        if hasattr(self.model, "env"):
            # Update step count
            self.episode_steps += 1
            
            # Broadcast metrics
            metrics = {
                "total_steps": self.num_timesteps,
                "episode_steps": self.episode_steps,
                "episode_reward": self.episode_reward,
            }
            self.ui_sync.broadcast_metrics(metrics)

            # Publish to ROS if adapter is available
            if self.ros_adapter:
                self.ros_adapter.publish_metrics(metrics)

        return True

    def _on_training_end(self) -> None:
        """Called when training ends."""
        logger.info(f"Training ended after {self.num_timesteps} steps")
        self.ui_sync.broadcast_metrics(
            {"training_complete": True, "total_steps": self.num_timesteps}
        )


class SB3RLTrainingLoop:
    """Advanced training loop integrating Stable Baselines3, ROS, and UI."""

    def __init__(
        self,
        environment: gym.Env,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        dashboard_port: int = 8000,
        enable_ros: bool = True,
        ros_namespace: str = "/robotic_learning",
    ):
        """Initialize integrated training loop.

        Args:
            environment: Gymnasium environment
            algorithm: RL algorithm (PPO, A2C, DDPG, SAC, TD3, DQN)
            policy: Policy network type
            dashboard_port: Dashboard server port
            enable_ros: Enable ROS integration
            ros_namespace: ROS namespace
        """
        self.environment = environment
        self.algorithm = algorithm
        self.policy = policy
        self.dashboard_port = dashboard_port

        # Initialize UI sync
        self.ui_sync = UISync()

        # Initialize ROS bridge if enabled
        self.ros_bridge = None
        self.ros_adapter = None
        if enable_ros:
            self.ros_bridge = ROSBridge(namespace=ros_namespace)
            self.ros_adapter = ROSUIAdapter(self.ros_bridge)
            logger.info(f"ROS bridge initialized with namespace {ros_namespace}")

        # Initialize RL agent
        self.agent = RLAgent(
            environment=self.environment,
            algorithm=self.algorithm,
            policy=self.policy,
        )

        # Initialize dashboard
        self.dashboard = DashboardServer(port=dashboard_port)
        self.dashboard.set_environment(self.environment, self.ui_sync)

        # Training state
        self.is_training = False
        self.episode_count = 0
        self.total_steps = 0

    def run_episode(
        self,
        max_steps: Optional[int] = None,
    ) -> float:
        """Run a single training episode.

        Args:
            max_steps: Maximum steps per episode

        Returns:
            Total episode reward
        """
        observation, info = self.environment.reset()
        episode_reward = 0.0
        episode_steps = 0

        while True:
            # Get action from agent
            action, _ = self.agent.predict(observation, deterministic=False)
            
            # Step environment
            observation, reward, terminated, truncated, info = self.environment.step(
                action
            )

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # Update UIs
            self.ui_sync.update_state(observation, reward, info)
            if self.ros_adapter:
                self.ros_adapter.update_state(observation, action, reward, info)

            # Render if needed
            self.environment.render()

            if terminated or truncated:
                break

            if max_steps and episode_steps >= max_steps:
                break

        self.episode_count += 1
        return episode_reward

    def train(
        self,
        total_timesteps: int = 100000,
        num_episodes: Optional[int] = None,
        run_dashboard: bool = True,
        dashboard_thread: bool = True,
    ) -> None:
        """Run training with integrated UI and ROS support.

        Args:
            total_timesteps: Total training steps
            num_episodes: Alternative: number of episodes
            run_dashboard: Whether to run dashboard
            dashboard_thread: Run dashboard in separate thread
        """
        self.is_training = True

        # Start dashboard if requested
        if run_dashboard:
            if dashboard_thread:
                dashboard_thread_obj = threading.Thread(
                    target=self.dashboard.run,
                    kwargs={"debug": False},
                )
                dashboard_thread_obj.daemon = True
                dashboard_thread_obj.start()
                logger.info(
                    f"Dashboard started on http://localhost:{self.dashboard_port}"
                )
            else:
                self.dashboard.run(debug=False)

        # Create training callback
        callback = RLCallback(
            ui_sync=self.ui_sync,
            ros_adapter=self.ros_adapter,
        )

        try:
            # Train the agent
            if num_episodes:
                for episode in range(num_episodes):
                    logger.info(f"Episode {episode + 1}/{num_episodes}")
                    reward = self.run_episode()
                    logger.info(f"Episode reward: {reward:.2f}")
            else:
                logger.info(f"Training for {total_timesteps} timesteps")
                self.agent.learn(
                    total_timesteps=total_timesteps,
                    callback=callback,
                )

            logger.info("Training completed successfully")
            self.ui_sync.broadcast_metrics(
                {
                    "training_complete": True,
                    "total_episodes": self.episode_count,
                    "total_steps": self.total_steps,
                }
            )

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            if self.ros_adapter:
                self.ros_adapter.publish_error("training_error", str(e))
        finally:
            self.is_training = False
            self.environment.close()

    def save_model(self, path: str) -> None:
        """Save trained model.

        Args:
            path: Path to save model
        """
        self.agent.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load trained model.

        Args:
            path: Path to load model from
        """
        self.agent.load(path, environment=self.environment)
        logger.info(f"Model loaded from {path}")

    def close(self) -> None:
        """Close all resources."""
        self.environment.close()
        if self.ros_bridge:
            self.ros_bridge.shutdown()
        logger.info("Training loop closed")
