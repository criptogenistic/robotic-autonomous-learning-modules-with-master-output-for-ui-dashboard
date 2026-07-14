"""Enhanced dashboard with learning panel."""

from fastapi import FastAPI
from src.dashboard import DashboardServer
from src.learning_ui_router import LearningRouter
from src.advanced_learner import AdvancedLearner, HyperparameterManager
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class EnhancedDashboard(DashboardServer):
    """Enhanced dashboard with RL Zoo and SB3 Contrib learning UI."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        advanced_learner: Optional[AdvancedLearner] = None,
    ):
        """Initialize enhanced dashboard.

        Args:
            host: Server host
            port: Server port
            advanced_learner: AdvancedLearner instance
        """
        super().__init__(host=host, port=port)
        self.advanced_learner = advanced_learner
        
        # Add learning router
        learning_router = LearningRouter(advanced_learner)
        self.app.include_router(learning_router.get_router())

        logger.info("Enhanced dashboard initialized with learning panel")

    def set_advanced_learner(self, learner: AdvancedLearner) -> None:
        """Set the advanced learner.

        Args:
            learner: AdvancedLearner instance
        """
        self.advanced_learner = learner
        logger.info("Advanced learner set for dashboard")

    def get_dashboard_info(self) -> dict:
        """Get dashboard information.

        Returns:
            Dashboard info dictionary
        """
        info = {
            "status": "healthy",
            "port": self.port,
            "host": self.host,
            "learning_available": self.advanced_learner is not None,
        }
        
        if self.advanced_learner:
            info["learner"] = self.advanced_learner.get_info()
        
        return info
