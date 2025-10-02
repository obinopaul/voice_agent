
# Morgana Backend Documentation

This document provides a comprehensive overview of the Morgana backend system. It is intended to serve as a technical guide for developers, especially for those building a frontend client to interact with this backend.

## 1. System Architecture

The backend is a multi-service system orchestrated with Docker Compose. The primary components are:

- **FastAPI (API) Service**: The core of the application. It's a Python-based API built with FastAPI that exposes endpoints for user authentication, session management, and chat interactions with a LangGraph agent.
- **LiveKit Server**: A real-time media server that handles all WebRTC communication (audio/video streaming). It acts as the central hub connecting clients (like a web frontend) and agents.
- **LiveKit Agent**: A specialized Python worker that connects to the LiveKit Server as a participant. It listens for user audio, transcribes it, sends the text to the FastAPI backend for processing, receives the response, synthesizes it into speech, and streams it back.
- **PostgreSQL (DB)**: The primary database for storing user accounts, sessions, and chat history.
- **Redis**: A fast in-memory cache, used for rate limiting and potentially other caching tasks.

## 2. Getting Started

### Environment Setup

1.  Copy the `.env.example` file to a new file named `.env`.
2.  Fill in the required values in the `.env` file. The key variables are:
    - `MORGANA_BACKEND_URL`: The base URL for the backend (e.g., `http://localhost:8000`).
    - `POSTGRES_*`: Credentials for the database.
    - `OPENAI_API_KEY`: Your key for the language model.
    - `LIVEKIT_*`: Credentials for the LiveKit server (`devkey` and `secret` for local development).
    - `API_USER_EMAIL` & `API_USER_PASSWORD`: The master credentials the LiveKit Agent will use to authenticate itself with the API.

### Running the System

The entire backend stack can be started with a single command from the project's root directory:

```bash
docker compose up --build
```

This command will:
1.  Build the Docker images for the `api`, `livekit-agent`, and any other custom services.
2.  Download the pre-built images for PostgreSQL, Redis, and the LiveKit Server.
3.  Start all the services and connect them on a shared Docker network.

## 3. Core API (FastAPI)

The API is the primary interface for any client application.

- **Base URL**: All API endpoints are prefixed with `/api/v1`. When running locally, the full base URL is `http://localhost:8000/api/v1`.
- **Health Check**: You can check if the API is running by making a `GET` request to the `/health` endpoint.
  - `GET /health` -> Returns `{"status": "ok"}`.

### Authentication Flow

The API uses a two-token authentication system to ensure security and proper session management. A frontend client **must** follow this flow.

#### **Step 1: Obtain a User Token**

This token represents an authenticated user. It is long-lived and is used to create new chat sessions.

- **Option A: Register a New User**
  - **Endpoint**: `POST /api/v1/auth/register`
  - **Request Body**:
    ```json
    {
      "email": "user@example.com",
      "password": "A_Strong_Password123!"
    }
    ```
  - **Success Response (200 OK)**:
    ```json
    {
      "id": 1,
      "email": "user@example.com",
      "token": {
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "token_type": "bearer",
        "expires_at": "2025-09-27T15:00:00Z"
      }
    }
    ```

- **Option B: Log In an Existing User**
  - **Endpoint**: `POST /api/v1/auth/login`
  - **Request Body**: `application/x-www-form-urlencoded`
    - `username`: The user's email.
    - `password`: The user's password.
  - **Success Response (200 OK)**: Returns a `TokenResponse` containing the `access_token`.

#### **Step 2: Create a Chat Session & Obtain a Session Token**

This token represents a single, unique conversation. It is required for all chat-related endpoints.

- **Endpoint**: `POST /api/v1/auth/session`
- **Authentication**: Requires the **User Token** from Step 1 in the header:
  - `Authorization: Bearer <user_access_token>`
- **Success Response (200 OK)**:
  ```json
  {
    "session_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "name": "Friendly-Chat-123",
    "token": {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "token_type": "bearer",
      "expires_at": "2025-09-27T16:00:00Z"
    }
  }
  ```
  The `token.access_token` from this response is the **Session Token**.

### Chat & Message History Endpoints

All endpoints in this section require the **Session Token** for authentication (`Authorization: Bearer <session_token>`).

- **Main Chat Endpoint (Streaming)**
  - **Endpoint**: `POST /api/v1/chatbot/chat/stream`
  - **Description**: Sends a list of messages and streams back the agent's response.
  - **Request Body**:
    ```json
    {
      "messages": [
        {"role": "user", "content": "Hello, who are you?"}
      ]
    }
    ```
  - **Response**: A `text/event-stream` response (Server-Sent Events). Each event is a JSON object with `content` (the chunk of text) and `done` (a boolean indicating if the stream is finished).

- **Message History**
  - **Get all messages in a session**: `GET /api/v1/chatbot/messages`
  - **Delete all messages in a session**: `DELETE /api/v1/chatbot/messages`

## 4. LiveKit Integration Guide

This section explains how the real-time voice components work and how a frontend should connect.

### Architecture

- The **LiveKit Server** (on port `7880`) is the central hub.
- The **LiveKit Agent** and any **Frontend Client** are both participants that connect to the server. **They do not connect directly to each other.**

### Agent Behavior

The `livekit-agent` service is a worker that:
1.  Starts up and uses its master credentials (`API_USER_EMAIL`, `API_USER_PASSWORD`) to get a long-lived **User Token** from the API.
2.  Connects to the LiveKit Server and waits in a room.
3.  When a new participant (a frontend user) joins the room, the agent is notified.
4.  **Crucially**, the agent then calls the backend's `/api/v1/auth/session` endpoint to create a **new, dedicated chat session** for that specific user.
5.  It uses the **Session Token** from that new session for all its communication with the `/chat/stream` endpoint on behalf of that user.

### How a Frontend Should Connect (Recommended Flow)

A frontend cannot just connect to the LiveKit server directly; it needs to be authorized. The backend is the sole authority for creating sessions and granting access.

#### **Step 1: Obtain a Session Token**
Perform the two-step authentication flow described in Section 3 to get a **Session Token** and a `session_id`.

#### **Step 2: Request a LiveKit Token**
Make a `POST` request to the `/api/v1/livekit/token` endpoint, authenticating with the **Session Token**.

- **Endpoint**: `POST /api/v1/livekit/token`
- **Authentication**: Requires the **Session Token** from Step 1 in the header:
  - `Authorization: Bearer <session_token>`
- **Success Response (200 OK)**:
  ```json
  {
    "serverUrl": "wss://your-livekit-server-url.com",
    "participantToken": "livekit_participant_jwt_token"
  }
  ```

#### **Step 3: Connect to LiveKit**
Use the LiveKit Client SDK (e.g., for JavaScript/React) to connect to the LiveKit Server using the URL and token received.

Once connected, the agent will detect the new participant and the conversation can begin.
