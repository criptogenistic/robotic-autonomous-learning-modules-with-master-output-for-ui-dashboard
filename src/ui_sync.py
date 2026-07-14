"""UI synchronization layer for real-time dashboard updates."""

import asyncio
from typing import Any, Dict, Optional, Callable
import json
from datetime import datetime


class UISync:
    """Handles synchronization between robot environment and UI dashboard."""

    def __init__(self, update_interval: float = 0.016):
        """Initialize UI sync.

        Args:
            update_interval: Time between UI updates in seconds (default: 60 FPS)
        """
        self.update_interval = update_interval
        self.subscribers: Dict[str, list] = {}
        self.state_buffer: Dict[str, Any] = {}
        self.last_update_time = datetime.now()
        self.is_running = False

    def subscribe(
        self, event_name: str, callback: Callable[[Dict], None]
    ) -> None:
        """Subscribe to environment events.

        Args:
            event_name: Name of event to subscribe to
            callback: Callback function to execute on event
        """
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(callback)

    def unsubscribe(
        self, event_name: str, callback: Callable[[Dict], None]
    ) -> None:
        """Unsubscribe from environment events.

        Args:
            event_name: Name of event to unsubscribe from
            callback: Callback function to remove
        """
        if event_name in self.subscribers:
            if callback in self.subscribers[event_name]:
                self.subscribers[event_name].remove(callback)

    def publish(
        self, event_name: str, data: Dict[str, Any]
    ) -> None:
        """Publish an event to all subscribers.

        Args:
            event_name: Name of event
            data: Event data
        """
        data["timestamp"] = datetime.now().isoformat()
        data["event"] = event_name

        for callback in self.subscribers.get(event_name, []):
            try:
                callback(data)
            except Exception as e:
                print(f"Error in subscriber callback: {e}")

    def update_state(
        self,
        observation: Any,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Update internal state from environment step.

        Args:
            observation: Environment observation
            reward: Reward value
            info: Additional info from environment
        """
        self.state_buffer["observation"] = observation.tolist()
        self.state_buffer["reward"] = float(reward)
        self.state_buffer["info"] = info

        # Publish update if interval has passed
        now = datetime.now()
        elapsed = (now - self.last_update_time).total_seconds()
        if elapsed >= self.update_interval:
            self.publish("state_update", self.state_buffer.copy())
            self.last_update_time = now

    def broadcast_metrics(
        self, metrics: Dict[str, float]
    ) -> None:
        """Broadcast performance metrics.

        Args:
            metrics: Dictionary of metrics to broadcast
        """
        self.publish("metrics_update", metrics)

    def broadcast_error(
        self, error_type: str, error_message: str
    ) -> None:
        """Broadcast error event.

        Args:
            error_type: Type of error
            error_message: Error message
        """
        self.publish(
            "error",
            {
                "type": error_type,
                "message": error_message,
            },
        )

    def get_state_json(self) -> str:
        """Get current state as JSON.

        Returns:
            JSON string of state buffer
        """
        return json.dumps(self.state_buffer)
