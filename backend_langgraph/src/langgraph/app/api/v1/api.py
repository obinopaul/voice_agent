"""API v1 router configuration.

This module sets up the main API router and includes all sub-routers for different
endpoints like authentication and chatbot functionality.
"""

from fastapi import APIRouter

from src.langgraph.app.api.v1.auth import router as auth_router
from src.langgraph.app.api.v1.chatbot import router as chatbot_router
from src.langgraph.app.core.logging import logger

api_router = APIRouter()

# Include core routers
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])

# Conditionally include the LiveKit router if the library is installed
try:
    from src.langgraph.app.api.v1.livekit import router as livekit_router

    api_router.include_router(livekit_router, prefix="/livekit", tags=["livekit"])
    logger.info("livekit_router_loaded")
except ImportError:
    logger.warning("livekit library not found, skipping LiveKit router.")


@api_router.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        dict: Health status information.
    """
    logger.info("health_check_called")
    return {"status": "healthy", "version": "1.0.0"}
