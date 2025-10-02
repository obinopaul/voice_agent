# Morgana Frontend Documentation

This document provides a comprehensive overview of the Morgana frontend system. It is intended to serve as a technical guide for developers building or integrating with the Morgana backend.

## 1. Project Overview

The frontend is a Next.js application built with TypeScript that provides a user interface for interacting with an AI agent. It supports two modes of interaction:

-   **Voice Mode:** Real-time, full-duplex voice conversations powered by [LiveKit](https://livekit.io/). This includes live transcription of speech.
-   **Chat Mode:** Text-based chat with an AI agent, which can be powered by a LangGraph backend.

The application is designed to be customizable and can be adapted to work with different backends and AI agent services.

## 2. Core Technologies

-   **Framework:** [Next.js](https://nextjs.org/) (using the App Router)
-   **Language:** [TypeScript](https://www.typescriptlang.org/)
-   **Real-time Communication:** [LiveKit](https://livekit.io/)
-   **UI Components:** [shadcn/ui](https://ui.shadcn.com/) and custom components.
-   **Styling:** [Tailwind CSS](https://tailwindcss.com/)
-   **State Management:** React Context and custom hooks.

## 3. Project Structure

The project follows a standard Next.js App Router structure. Here are the key directories:

-   `/app`: Contains the application's pages and layouts.
    -   `/app/login`: The authentication page.
    -   `/app/(app)`: The main application view, protected by authentication.
-   `/components`: Reusable React components.
    -   `/components/livekit`: Components specifically for LiveKit integration (e.g., media controls, video tiles).
-   `/context`: React context providers, such as `AuthContext`.
-   `/hooks`: Custom React hooks for managing logic like authentication and LiveKit connections.
-   `/lib`: Utility functions and type definitions.
-   `app-config.ts`: A central file for UI and feature configuration.

## 4. Authentication Flow

The application uses a two-step token-based authentication process to secure communication with the backend.

1.  **Login:** The user enters their credentials on the `/login` page.
2.  **User Token Request:** The `AuthContext` sends a `POST` request to `/api/v1/auth/login` on your backend with the user's email and password. The backend validates the credentials and returns a short-lived **user token**.
3.  **Session Token Request:** Immediately after receiving the user token, the frontend makes a `POST` request to `/api/v1/auth/session` on your backend, including the user token in the `Authorization` header. The backend then returns a longer-lived **session token**.
4.  **Token Storage:** This session token is stored in the browser's `localStorage` and is used for all subsequent authenticated requests from the frontend.
5.  **Authenticated Access:** The main application page checks for the presence of the session token. If it's missing, the user is redirected back to the `/login` page.

## 5. LiveKit Integration and Entering a Room

Once authenticated, the user can start a voice session. This is how the frontend connects to a LiveKit room:

1.  **Start Voice Call:** The user clicks the "Start Voice Call" button in the UI.
2.  **LiveKit Token Request:** The `useLiveKitConnection` hook (a custom hook in this project) sends a `POST` request to `/api/v1/livekit/token` on your backend. This request is authenticated with the session token.
3.  **Backend Creates a Room and Token:** Your backend receives this request and should:
    -   Decide which LiveKit room the user should join (e.g., create a new one or use an existing one).
    -   Generate a unique **participant token** for that user and room using the LiveKit Server SDK.
    -   Return the LiveKit server URL and the generated participant token to the frontend.
4.  **Connecting to the Room:** The frontend receives the LiveKit server URL and participant token. The `App` component then uses these details to connect to the LiveKit room using the LiveKit Client SDK.
5.  **Session Begins:** Once connected, the frontend renders the `SessionView` component, which handles the in-call experience, including displaying media from other participants (like the AI agent) and managing the user's microphone and camera.

## 6. Key Components and Hooks

-   **`context/AuthContext.tsx`**: Manages the entire authentication lifecycle, including storing the session token and providing `login` and `logout` functions.
-   **`app/(app)/page.tsx`**: The main entry point for the authenticated user experience. It renders the core `App` component.
-   **`components/app.tsx`**: The main state machine for the UI. It manages whether the user is in the "welcome" screen or an active "session" and whether the session is in "chat" or "voice" mode. It also initializes the LiveKit `Room` object.
-   **`components/session-view.tsx`**: The heart of the user interaction. It displays the conversation (from chat or voice transcription) and includes the `AgentControlBar` for media controls. It dynamically selects the message source based on whether the session is in "voice" (using LiveKit) or "chat" (using LangGraph) mode.
-   **`hooks/useLiveKitConnection.ts`**: A crucial custom hook responsible for communicating with your backend to get the necessary credentials to connect to a LiveKit room.
-   **`hooks/useChatAndTranscription.ts`**: This hook merges the real-time voice transcriptions from LiveKit with the text-based chat messages into a single, chronological list. This provides a unified view of the entire conversation.
-   **`components/livekit/agent-control-bar/agent-control-bar.tsx`**: This component provides the user with all the necessary controls during a session, such as toggling the microphone and camera, selecting media devices, and leaving the call.

## 7. How to Integrate Your Own Backend

To make this frontend work with your own backend, you need to implement three API endpoints and configure the frontend to point to them.

**Step 1: Set Environment Variables**

In your `.env.local` file, set the following variable to the base URL of your backend:

```
NEXT_PUBLIC_MORGANA_BACKEND_URL=http://your-backend-url.com
```

**Step 2: Implement the Required Backend Endpoints**

Your backend must expose the following three endpoints:

1.  **`POST /api/v1/auth/login`**
    -   **Request Body:** `x-www-form-urlencoded` with `username` and `password`.
    -   **Functionality:** Authenticate the user.
    -   **Response Body (JSON):**
        ```json
        {
          "access_token": "your_user_jwt_token",
          "token_type": "bearer"
        }
        ```

2.  **`POST /api/v1/auth/session`**
    -   **Request Headers:** `Authorization: Bearer <user_jwt_token>`
    -   **Functionality:** Create a longer-lived session for the user.
    -   **Response Body (JSON):**
        ```json
        {
          "token": {
            "access_token": "your_session_jwt_token"
          }
        }
        ```

3.  **`POST /api/v1/livekit/token`**
    -   **Request Headers:** `Authorization: Bearer <session_jwt_token>`
    -   **Functionality:** Generate a LiveKit access token for the authenticated user. You will need to use the LiveKit Server SDK for this.
    -   **Response Body (JSON):**
        ```json
        {
          "serverUrl": "wss://your-livekit-server-url.com",
          "participantToken": "livekit_participant_jwt_token"
        }
        ```

**Step 3: Customize Configuration (Optional)**

You can customize the look and feel of the application by editing the `app-config.ts` file. This allows you to change the app name, logos, colors, and enable or disable features like video and screen sharing.
