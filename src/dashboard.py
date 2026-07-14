"""FastAPI dashboard server for real-time monitoring and control."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import json
import logging


logger = logging.getLogger(__name__)


class RobotCommand(BaseModel):
    """Robot command model."""

    command_type: str
    action: Optional[Dict] = None
    voice_command: Optional[str] = None


class EnvironmentConfig(BaseModel):
    """Environment configuration model."""

    motion_type: str = "continuous"
    max_steps: int = 1000
    render_mode: Optional[str] = None


class DashboardServer:
    """FastAPI server for robot control dashboard."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """Initialize dashboard server.

        Args:
            host: Server host
            port: Server port
        """
        self.app = FastAPI(
            title="Robotic Learning Dashboard",
            description="Real-time monitoring and control of robotic RL agents",
            version="1.0.0",
        )
        self.host = host
        self.port = port
        self.active_connections: List[WebSocket] = []
        self.environment = None
        self.ui_sync = None

        self._setup_routes()
        self._setup_middleware()

    def _setup_middleware(self) -> None:
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "version": "1.0.0"}

        @self.app.get("/status")
        async def get_status():
            """Get current environment status."""
            if self.ui_sync is None:
                raise HTTPException(status_code=503, detail="Environment not initialized")
            return json.loads(self.ui_sync.get_state_json())

        @self.app.post("/command")
        async def send_command(command: RobotCommand):
            """Send command to robot."""
            if self.environment is None:
                raise HTTPException(status_code=503, detail="Environment not initialized")

            if command.command_type == "voice":
                self.environment.set_voice_command(command.voice_command)
                await self.broadcast_message(
                    {"type": "voice_command", "command": command.voice_command}
                )
            elif command.command_type == "action":
                await self.broadcast_message(
                    {"type": "action_received", "action": command.action}
                )

            return {"status": "accepted", "command": command.command_type}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    await self.broadcast_message(json.loads(data))
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.active_connections.remove(websocket)

    async def broadcast_message(self, message: Dict) -> None:
        """Broadcast message to all connected clients.

        Args:
            message: Message to broadcast
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.active_connections.remove(connection)

    def set_environment(self, environment, ui_sync) -> None:
        """Set the environment and UI sync instance.

        Args:
            environment: RoboticEnvironment instance
            ui_sync: UISync instance
        """
        self.environment = environment
        self.ui_sync = ui_sync
        # Subscribe to UI sync events
        ui_sync.subscribe("state_update", self._on_state_update)

    async def _on_state_update(self, data: Dict) -> None:
        """Handle state update from UI sync.

        Args:
            data: State update data
        """
        await self.broadcast_message(data)

    def run(self, debug: bool = False) -> None:
        """Run the dashboard server.

        Args:
            debug: Enable debug mode
        """
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
        )
