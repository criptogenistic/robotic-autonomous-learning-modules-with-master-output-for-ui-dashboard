"""Tests for Stable Baselines3 integration."""

import unittest
import gymnasium as gym
import numpy as np
from src.rl_agent import RLAgent
from src.ros_integration import ROSBridge, ROSUIAdapter


class TestRLAgent(unittest.TestCase):
    """Test RLAgent class."""

    def setUp(self):
        """Set up test environment."""
        self.env = gym.make("CartPole-v1")

    def tearDown(self):
        """Tear down test environment."""
        self.env.close()

    def test_agent_creation(self):
        """Test agent creation."""
        agent = RLAgent(
            environment=self.env,
            algorithm="PPO",
            policy="MlpPolicy",
        )
        self.assertIsNotNone(agent.model)
        self.assertEqual(agent.algorithm_name, "PPO")

    def test_agent_prediction(self):
        """Test agent prediction."""
        agent = RLAgent(
            environment=self.env,
            algorithm="PPO",
            policy="MlpPolicy",
        )
        obs, _ = self.env.reset()
        action, _ = agent.predict(obs)
        self.assertIsNotNone(action)

    def test_unsupported_algorithm(self):
        """Test unsupported algorithm error."""
        with self.assertRaises(ValueError):
            RLAgent(
                environment=self.env,
                algorithm="UnsupportedAlgorithm",
            )


class TestROSIntegration(unittest.TestCase):
    """Test ROS integration."""

    def test_ros_bridge_creation(self):
        """Test ROSBridge creation."""
        bridge = ROSBridge(namespace="/test")
        self.assertEqual(bridge.namespace, "/test")

    def test_topic_subscription(self):
        """Test topic subscription."""
        bridge = ROSBridge()
        called = []

        def callback(msg):
            called.append(msg)

        bridge.subscribe("test_topic", "String", callback)
        self.assertIn("/test_topic", bridge.subscribers)

    def test_ros_ui_adapter(self):
        """Test ROSUIAdapter."""
        bridge = ROSBridge()
        adapter = ROSUIAdapter(bridge)
        
        obs = np.array([0.1, 0.2, 0.3, 0.4])
        action = np.array([1.0])
        reward = 1.0
        info = {"step": 1}
        
        adapter.update_state(obs, action, reward, info)
        state = adapter.get_state()
        
        self.assertIsNotNone(state["observation"])
        self.assertIsNotNone(state["action"])
        self.assertEqual(state["reward"], reward)


if __name__ == "__main__":
    unittest.main()
