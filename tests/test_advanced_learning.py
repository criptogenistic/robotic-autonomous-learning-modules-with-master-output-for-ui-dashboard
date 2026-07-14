"""Tests for advanced learning integration."""

import unittest
import gymnasium as gym
import numpy as np
from src.advanced_learner import AdvancedLearner, HyperparameterManager


class TestHyperparameterManager(unittest.TestCase):
    """Test HyperparameterManager."""

    def setUp(self):
        """Set up test."""
        self.manager = HyperparameterManager()

    def test_get_hyperparams(self):
        """Test getting hyperparameters."""
        params = self.manager.get_hyperparams("ppo")
        self.assertIsNotNone(params)
        self.assertIn("learning_rate", params)

    def test_suggest_algorithm(self):
        """Test algorithm suggestions."""
        suggestions = self.manager.suggest_algorithm("continuous")
        self.assertIsNotNone(suggestions)
        self.assertGreater(len(suggestions), 0)

    def test_suggest_discrete(self):
        """Test discrete algorithm suggestions."""
        suggestions = self.manager.suggest_algorithm("discrete")
        self.assertIn("PPO", suggestions)
        self.assertIn("DQN", suggestions)


class TestAdvancedLearner(unittest.TestCase):
    """Test AdvancedLearner."""

    def setUp(self):
        """Set up test."""
        self.env = gym.make("CartPole-v1")

    def tearDown(self):
        """Tear down test."""
        self.env.close()

    def test_learner_creation(self):
        """Test learner creation."""
        learner = AdvancedLearner(
            environment=self.env,
            algorithm="PPO",
            policy="MlpPolicy",
        )
        self.assertIsNotNone(learner.model)
        self.assertEqual(learner.algorithm_name, "ppo")

    def test_learner_prediction(self):
        """Test learner prediction."""
        learner = AdvancedLearner(
            environment=self.env,
            algorithm="PPO",
        )
        obs, _ = self.env.reset()
        action, _ = learner.predict(obs)
        self.assertIsNotNone(action)

    def test_learner_info(self):
        """Test learner info."""
        learner = AdvancedLearner(
            environment=self.env,
            algorithm="A2C",
        )
        info = learner.get_info()
        self.assertEqual(info["algorithm"], "a2c")
        self.assertIn("training_steps", info)


if __name__ == "__main__":
    unittest.main()
