"""Main training loop with Gymnasium and UI synchronization."""

import gymnasium as gym
import numpy as np
from typing import Optional
from .environment import RoboticEnvironment
from .ui_sync import UISync
from .dashboard import DashboardServer
import threading


class TrainingLoop:
    """Training loop that synchronizes with UI dashboard."""

    def __init__(
        self,
        env_config: dict = None,
        dashboard_port: int = 8000,
    ):
        """Initialize training loop.

        Args:
            env_config: Environment configuration
            dashboard_port: Port for dashboard server
        """
        self.env_config = env_config or {}
        self.environment = RoboticEnvironment(**self.env_config)
        self.ui_sync = UISync()
        self.dashboard = DashboardServer(port=dashboard_port)
        self.dashboard.set_environment(self.environment, self.ui_sync)

        self.is_running = False
        self.episode_count = 0
        self.total_steps = 0

    def run_episode(self, max_steps: Optional[int] = None) -> float:
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
            # Simple random policy for now
            action = self.environment.action_space.sample()
            observation, reward, terminated, truncated, info = self.environment.step(
                action
            )

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # Update UI with state
            self.ui_sync.update_state(observation, reward, info)

            # Render if needed
            if self.env_config.get("render_mode"):
                self.environment.render()

            if terminated or truncated:
                break

            if max_steps and episode_steps >= max_steps:
                break

        self.episode_count += 1
        metrics = {
            "episode": self.episode_count,
            "total_steps": self.total_steps,
            "episode_reward": episode_reward,
            "episode_length": episode_steps,
        }
        self.ui_sync.broadcast_metrics(metrics)

        return episode_reward

    def run_training(
        self,
        num_episodes: int = 100,
        dashboard_port: int = 8000,
        run_dashboard_thread: bool = True,
    ) -> None:
        """Run training loop with dashboard.

        Args:
            num_episodes: Number of episodes to train
            dashboard_port: Port for dashboard server
            run_dashboard_thread: Whether to run dashboard in separate thread
        """
        self.is_running = True

        # Start dashboard in separate thread if requested
        if run_dashboard_thread:
            dashboard_thread = threading.Thread(
                target=self.dashboard.run, kwargs={"debug": False}
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()
            print(f"Dashboard started on http://localhost:{dashboard_port}")

        try:
            for episode in range(num_episodes):
                print(f"Running episode {episode + 1}/{num_episodes}")
                reward = self.run_episode()
                print(
                    f"Episode {episode + 1} completed with reward: {reward:.2f}"
                )
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            self.is_running = False
            self.environment.close()

    def close(self) -> None:
        """Close training loop resources."""
        self.environment.close()
