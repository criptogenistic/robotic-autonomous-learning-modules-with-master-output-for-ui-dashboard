"""Robotic environment wrapper with Gymnasium v1.3.0 integration."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Tuple, Optional


class RoboticEnvironment(gym.Env):
    """Gymnasium-compatible robotic environment with voice command rerouting."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        motion_type: str = "continuous",
    ):
        """Initialize the robotic environment.

        Args:
            render_mode: The rendering mode ("human", "rgb_array", or None)
            motion_type: Type of motion control ("continuous" or "discrete")
        """
        self.render_mode = render_mode
        self.motion_type = motion_type
        self.is_voice_controlled = False
        self.voice_command = None

        # Action space: 6D continuous for robot joints + 1D for voice override flag
        if motion_type == "continuous":
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(6,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(64)  # 2^6 for 6 binary joints

        # Observation space: robot state (position, velocity, acceleration, force)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),  # 6*3 for position, velocity, acceleration
            dtype=np.float32,
        )

        self.robot_state = np.zeros(18, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 1000

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Observation and info dict
        """
        super().reset(seed=seed)
        self.robot_state = np.zeros(18, dtype=np.float32)
        self.step_count = 0
        self.is_voice_controlled = False
        self.voice_command = None
        return self.robot_state.copy(), {}

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step of the environment.

        Args:
            action: Action to execute

        Returns:
            Observation, reward, terminated, truncated, info dict
        """
        self.step_count += 1

        # Apply action to robot state
        if self.motion_type == "continuous":
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self.robot_state[:6] += action * 0.1
        else:
            # Decode discrete action to continuous
            action_continuous = self._decode_discrete_action(action)
            self.robot_state[:6] += action_continuous * 0.1

        # Update velocity and acceleration
        self.robot_state[6:12] = self.robot_state[:6] * 0.5
        self.robot_state[12:18] = self.robot_state[6:12] * 0.5

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "voice_controlled": self.is_voice_controlled,
            "voice_command": self.voice_command,
        }

        return self.robot_state.copy(), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "human":
            print(f"Robot State at step {self.step_count}: {self.robot_state}")
        elif self.render_mode == "rgb_array":
            return self._get_rgb_frame()
        return None

    def set_voice_command(self, command: str) -> None:
        """Set voice command for robot rerouting.

        Args:
            command: Voice command string
        """
        self.voice_command = command
        self.is_voice_controlled = True

    def _decode_discrete_action(self, action: int) -> np.ndarray:
        """Decode discrete action to continuous.

        Args:
            action: Discrete action index

        Returns:
            Continuous action array
        """
        action_bits = [(action >> i) & 1 for i in range(6)]
        return np.array(action_bits, dtype=np.float32) * 2 - 1

    def _calculate_reward(self) -> float:
        """Calculate reward based on robot state.

        Returns:
            Reward value
        """
        # Simple reward: minimize position deviation from origin
        position = self.robot_state[:6]
        reward = -np.sum(np.square(position)) * 0.01
        return float(reward)

    def _check_termination(self) -> bool:
        """Check if episode should terminate.

        Returns:
            True if episode should terminate
        """
        # Terminate if robot goes too far from origin
        position = self.robot_state[:6]
        if np.max(np.abs(position)) > 10.0:
            return True
        return False

    def _get_rgb_frame(self) -> np.ndarray:
        """Get RGB frame for rendering.

        Returns:
            RGB frame array
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Simple visualization: draw robot state as points
        for i, val in enumerate(self.robot_state[:6]):
            x = int((val + 10) / 20 * 640)
            y = int((i + 1) * 480 / 7)
            x = np.clip(x, 0, 639)
            y = np.clip(y, 0, 479)
            frame[y, x] = [255, 0, 0]
        return frame

    def close(self) -> None:
        """Close the environment."""
        pass
