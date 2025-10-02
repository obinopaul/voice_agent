"""Pydantic models for LiveKit API responses."""

from pydantic import BaseModel


class LiveKitTokenResponse(BaseModel):
    """Response model for a LiveKit token request."""

    serverUrl: str
    participantToken: str
