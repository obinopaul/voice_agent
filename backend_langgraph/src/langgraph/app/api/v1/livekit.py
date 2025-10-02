"""LiveKit API endpoints for handling real-time communication.

This module provides endpoints for generating LiveKit tokens.
"""

from fastapi import APIRouter, Depends, HTTPException
from livekit import api
from src.langgraph.app.api.v1.auth import get_current_session
from src.langgraph.app.core.config import settings
from src.langgraph.app.core.logging import logger
from src.langgraph.app.models.session import Session
from src.langgraph.app.schemas.livekit import LiveKitTokenResponse

router = APIRouter()

@router.post("/token", response_model=LiveKitTokenResponse)
async def create_livekit_token(session: Session = Depends(get_current_session)):
    """Create a LiveKit token for the authenticated user.

    Args:
        session: The current session from the auth token.

    Returns:
        LiveKitTokenResponse: The LiveKit server URL and participant token.
    """
    try:
        logger.info("creating_livekit_token", session_id=session.id)

        # Create a LiveKit access token
        token = (
            api.AccessToken(
                settings.LIVEKIT_API_KEY,
                settings.LIVEKIT_API_SECRET,
            )
            .with_identity(session.user_id)
            .with_name(session.name)
            .with_grant(api.VideoGrant(room_join=True, room=session.id))
        )

        logger.info("livekit_token_created", session_id=session.id)

        return LiveKitTokenResponse(
            serverUrl=settings.LIVEKIT_SERVER_URL,
            participantToken=token.to_jwt(),
        )
    except Exception as e:
        logger.error("create_livekit_token_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
