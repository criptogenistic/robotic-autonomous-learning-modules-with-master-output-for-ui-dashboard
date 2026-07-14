"""Example usage of Stable Baselines3 with ROS UI integration."""

import gymnasium as gym
from src.sb3_training_loop import SB3RLTrainingLoop
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    """Run training with all integrations."""
    # Create environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Create training loop with SB3, ROS, and UI
    training = SB3RLTrainingLoop(
        environment=env,
        algorithm="PPO",  # or A2C, DDPG, SAC, TD3, DQN
        policy="MlpPolicy",
        dashboard_port=8000,
        enable_ros=True,
        ros_namespace="/cartpole_learning",
    )

    # Train the agent
    training.train(
        total_timesteps=100000,
        run_dashboard=True,
        dashboard_thread=True,
    )

    # Save the model
    training.save_model("cartpole_ppo.zip")

    # Close resources
    training.close()

    print("Training complete!")
    print(f"Total episodes: {training.episode_count}")
    print(f"Total steps: {training.total_steps}")
    print(f"Dashboard: http://localhost:8000")


if __name__ == "__main__":
    main()
