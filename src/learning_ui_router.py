"""UI learning panel for advanced training configuration."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class AlgorithmConfig(BaseModel):
    """Algorithm configuration model."""
    algorithm: str
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    total_timesteps: int = 100000
    eval_freq: Optional[int] = 10000
    n_eval_episodes: int = 5


class HyperparameterSuggestion(BaseModel):
    """Hyperparameter suggestion model."""
    env_type: str
    algorithms: List[str]
    hyperparams: Dict[str, Any]


class LearningRouter:
    """Router for learning UI endpoints."""

    def __init__(self, advanced_learner=None):
        """Initialize learning router.

        Args:
            advanced_learner: AdvancedLearner instance
        """
        self.advanced_learner = advanced_learner
        self.router = APIRouter(prefix="/api/learning", tags=["learning"])
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.router.get("/algorithms")
        async def get_available_algorithms():
            """Get list of available algorithms."""
            from src.advanced_learner import AdvancedLearner
            return {
                "algorithms": list(AdvancedLearner.ALGORITHM_MAP.keys()),
                "sb3_contrib_available": "ars" in AdvancedLearner.ALGORITHM_MAP,
            }

        @self.router.post("/suggest-algorithm")
        async def suggest_algorithm(env_type: str):
            """Get algorithm suggestions for environment type."""
            if not self.advanced_learner:
                raise HTTPException(status_code=503, detail="Learner not initialized")

            suggestions = self.advanced_learner.hyperparam_manager.suggest_algorithm(
                env_type
            )
            return {"suggestions": suggestions, "env_type": env_type}

        @self.router.post("/hyperparameters")
        async def get_hyperparameters(algorithm: str, env_id: Optional[str] = None):
            """Get hyperparameters for algorithm."""
            if not self.advanced_learner:
                raise HTTPException(status_code=503, detail="Learner not initialized")

            hyperparams = self.advanced_learner.hyperparam_manager.get_hyperparams(
                algorithm, env_id
            )
            return {
                "algorithm": algorithm,
                "hyperparameters": hyperparams,
            }

        @self.router.post("/train")
        async def start_training(config: AlgorithmConfig):
            """Start training with specified configuration."""
            if not self.advanced_learner:
                raise HTTPException(status_code=503, detail="Learner not initialized")

            try:
                # This would be called in a background task in production
                metrics = self.advanced_learner.learn(
                    total_timesteps=config.total_timesteps,
                    eval_freq=config.eval_freq,
                    n_eval_episodes=config.n_eval_episodes,
                )
                return {
                    "status": "training_started",
                    "config": config.dict(),
                    "metrics": metrics,
                }
            except Exception as e:
                logger.error(f"Training error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/info")
        async def get_learner_info():
            """Get current learner information."""
            if not self.advanced_learner:
                raise HTTPException(status_code=503, detail="Learner not initialized")

            return self.advanced_learner.get_info()

        @self.router.post("/save")
        async def save_model(path: str):
            """Save current model."""
            if not self.advanced_learner:
                raise HTTPException(status_code=503, detail="Learner not initialized")

            try:
                self.advanced_learner.save(path)
                return {"status": "saved", "path": path}
            except Exception as e:
                logger.error(f"Save error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/load")
        async def load_model(path: str):
            """Load model from path."""
            if not self.advanced_learner:
                raise HTTPException(status_code=503, detail="Learner not initialized")

            try:
                self.advanced_learner.load(path)
                return {"status": "loaded", "path": path, "info": self.advanced_learner.get_info()}
            except Exception as e:
                logger.error(f"Load error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def get_router(self) -> APIRouter:
        """Get the router.

        Returns:
            FastAPI router
        """
        return self.router
