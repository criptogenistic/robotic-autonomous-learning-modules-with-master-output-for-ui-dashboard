"""ROS integration for robotic RL agents."""

import threading
from typing import Any, Dict, Optional, Callable
from queue import Queue
import json
import logging

logger = logging.getLogger(__name__)


class ROSBridge:
    """Bridge between RL agent and ROS ecosystem."""

    def __init__(
        self,
        node_name: str = "rl_agent",
        namespace: str = "/robotic_learning",
    ):
        """Initialize ROS bridge.

        Args:
            node_name: ROS node name
            namespace: ROS namespace
        """
        self.node_name = node_name
        self.namespace = namespace
        self.message_queue = Queue()
        self.subscribers: Dict[str, list] = {}
        self.publishers: Dict[str, Callable] = {}
        self.is_running = False

    def subscribe(
        self,
        topic: str,
        message_type: str,
        callback: Callable[[Dict], None],
    ) -> None:
        """Subscribe to ROS topic.

        Args:
            topic: Topic name
            message_type: Message type
            callback: Callback function
        """
        full_topic = f"{self.namespace}/{topic}"
        if full_topic not in self.subscribers:
            self.subscribers[full_topic] = []
        self.subscribers[full_topic].append(
            {"callback": callback, "type": message_type}
        )
        logger.info(f"Subscribed to {full_topic}")

    def publish(
        self,
        topic: str,
        message: Dict[str, Any],
    ) -> None:
        """Publish to ROS topic.

        Args:
            topic: Topic name
            message: Message data
        """
        full_topic = f"{self.namespace}/{topic}"
        self.message_queue.put({"topic": full_topic, "data": message})
        logger.debug(f"Published to {full_topic}: {message}")

    def get_full_topic_name(self, topic: str) -> str:
        """Get full ROS topic name.

        Args:
            topic: Topic name

        Returns:
            Full topic path
        """
        return f"{self.namespace}/{topic}"

    def shutdown(self) -> None:
        """Shutdown ROS bridge."""
        self.is_running = False
        logger.info("ROS bridge shutdown")


class ROSUIAdapter:
    """Adapter for ROS-compatible UI updates."""

    def __init__(self, ros_bridge: ROSBridge):
        """Initialize ROS UI adapter.

        Args:
            ros_bridge: ROSBridge instance
        """
        self.ros_bridge = ros_bridge
        self.ui_state: Dict[str, Any] = {}

    def update_state(
        self,
        observation: Any,
        action: Any,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Update UI state and publish to ROS.

        Args:
            observation: Current observation
            action: Agent action
            reward: Reward value
            info: Additional info
        """
        # Convert to serializable format
        try:
            if hasattr(observation, "tolist"):
                obs_data = observation.tolist()
            else:
                obs_data = list(observation) if isinstance(observation, (list, tuple)) else observation

            if hasattr(action, "tolist"):
                action_data = action.tolist()
            else:
                action_data = list(action) if isinstance(action, (list, tuple)) else action
        except Exception as e:
            logger.error(f"Error converting observation/action: {e}")
            obs_data = None
            action_data = None

        self.ui_state = {
            "observation": obs_data,
            "action": action_data,
            "reward": float(reward),
            "info": info,
        }

        # Publish to ROS
        self.ros_bridge.publish("ui/state", self.ui_state)

    def publish_metrics(
        self, metrics: Dict[str, float]
    ) -> None:
        """Publish training metrics to ROS.

        Args:
            metrics: Metrics dictionary
        """
        self.ros_bridge.publish("metrics", metrics)

    def publish_error(
        self, error_type: str, error_message: str
    ) -> None:
        """Publish error to ROS.

        Args:
            error_type: Error type
            error_message: Error message
        """
        self.ros_bridge.publish(
            "errors",
            {"type": error_type, "message": error_message},
        )

    def subscribe_to_commands(self, callback: Callable) -> None:
        """Subscribe to ROS commands.

        Args:
            callback: Command callback
        """
        self.ros_bridge.subscribe("commands", "String", callback)

    def get_state(self) -> Dict[str, Any]:
        """Get current UI state.

        Returns:
            UI state dictionary
        """
        return self.ui_state.copy()
