"""Complete training example with RL Zoo and SB3 Contrib."""

import gymnasium as gym
from src.advanced_learner import AdvancedLearner, HyperparameterManager
from src.enhanced_dashboard import EnhancedDashboard
from src.sb3_training_loop import SB3RLTrainingLoop
from src.ui_sync import UISync
from src.ros_integration import ROSBridge, ROSUIAdapter
import logging
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run complete training with all integrations."""
    
    # Create environment
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    logger.info(f"Created environment: {env_id}")

    # Initialize hyperparameter manager
    hyperparam_manager = HyperparameterManager()
    
    # Get algorithm suggestions
    suggested_algos = hyperparam_manager.suggest_algorithm("discrete")
    logger.info(f"Suggested algorithms for discrete env: {suggested_algos}")

    # Create advanced learner with RL Zoo hyperparameters
    learner = AdvancedLearner(
        environment=env,
        algorithm="PPO",  # Can use any from RL Zoo or SB3 Contrib
        policy="MlpPolicy",
        tensorboard_log="logs/",
    )
    logger.info(f"Created {learner.algorithm_name.upper()} learner")
    logger.info(f"Hyperparameters: {learner.hyperparams}")

    # Initialize UI components
    ui_sync = UISync()
    ros_bridge = ROSBridge(namespace="/robotic_learning")
    ros_adapter = ROSUIAdapter(ros_bridge)

    # Create enhanced dashboard with learning panel
    dashboard = EnhancedDashboard(
        host="0.0.0.0",
        port=8000,
        advanced_learner=learner,
    )

    # Run dashboard in separate thread
    dashboard_thread = threading.Thread(
        target=dashboard.run,
        kwargs={"debug": False},
    )
    dashboard_thread.daemon = True
    dashboard_thread.start()
    logger.info("Dashboard started on http://localhost:8000")

    # Train the agent
    logger.info("Starting training...")
    metrics = learner.learn(
        total_timesteps=100000,
        eval_env=eval_env,
        eval_freq=10000,
        n_eval_episodes=5,
    )
    logger.info(f"Training metrics: {metrics}")

    # Save the model
    model_path = "trained_models/cartpole_ppo"
    learner.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Get learner info
    info = learner.get_info()
    logger.info(f"Final learner info: {info}")

    # Publish to ROS and UI
    ros_adapter.publish_metrics(info)
    ui_sync.broadcast_metrics(info)

    # Close environments
    env.close()
    eval_env.close()

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Algorithm: {learner.algorithm_name.upper()}")
    print(f"Total Steps: {learner.training_steps}")
    print(f"Best Reward: {learner.best_reward:.2f}")
    print(f"Model saved to: {model_path}")
    print(f"Dashboard: http://localhost:8000")
    print(f"Learning API: http://localhost:8000/api/learning")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
