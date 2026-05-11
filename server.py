#!/usr/bin/env python3
"""
FastAPI server entry point for Medical Diagnosis AI

Local Development:
  python server.py
  Server will start on http://localhost:8000
  API docs: http://localhost:8000/docs

Hugging Face Spaces / Production:
  python server.py
  Server will start on http://0.0.0.0:7860
  Serves both API and static React frontend
"""

import uvicorn
import logging
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_server_config():
    """Determine server configuration based on environment"""
    # Check if running in Hugging Face Spaces
    in_space = os.getenv("SPACE_ID") is not None

    host = "0.0.0.0"
    port = 7860 if in_space else 8000
    reload = not in_space  # Disable reload in production (HF Spaces)

    return host, port, reload, in_space


if __name__ == "__main__":
    host, port, reload, in_space = get_server_config()

    # Import FastAPI app AFTER configuration
    from app.api import app

    # Configure static file serving for the React frontend
    frontend_dist = Path(__file__).parent / "frontend" / "dist"

    if frontend_dist.exists():
        logger.info(f"📁 Mounting static files from: {frontend_dist}")
        # Mount static files at root, but BEFORE API routes are checked
        # This way /api/* routes are handled by FastAPI, everything else by static files
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
    else:
        logger.warning(f"⚠️  Frontend dist directory not found: {frontend_dist}")
        logger.warning("    Run 'cd frontend && npm run build' to build the frontend")

    # Log startup info
    environment = "🚀 Hugging Face Spaces" if in_space else "💻 Local Development"
    logger.info(f"Starting Medical Diagnosis AI Server ({environment})")
    logger.info(f"📍 Server: http://{host}:{port}")
    logger.info(f"🌐 Frontend: http://{host}:{port}")
    logger.info(f"📊 API Base: http://{host}:{port}/api")
    logger.info(f"📖 API Docs: http://{host}:{port}/docs")

    uvicorn.run(
        "app.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
