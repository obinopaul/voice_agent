import asyncio
import logging
import os
import requests
import uuid
from livekit import rtc, api
from dotenv import load_dotenv
from jose import jwt
from src.langgraph.app.core.config import settings

# Load environment variables
load_dotenv()

# --- Configuration ---
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "http://livekit:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# --- Test Details ---
ROOM_NAME = "morgana-test-room"
TEST_USER_EMAIL = f"test-user-{uuid.uuid4()}@example.com"
TEST_USER_PASSWORD = "A_Secure_Password_123_$"


async def get_session_token(email, password):
    """Authenticate and get a session token."""
    # Attempt to register the user
    requests.post(f"{API_BASE_URL}/auth/register", json={"email": email, "password": password})

    # Log in to get a user token
    login_payload = {"username": email, "password": password}
    login_response = requests.post(f"{API_BASE_URL}/auth/login", data=login_payload)
    login_response.raise_for_status()
    user_token = login_response.json()["access_token"]

    # Create a new chat session
    user_headers = {"Authorization": f"Bearer {user_token}"}
    session_response = requests.post(f"{API_BASE_URL}/auth/session", headers=user_headers)
    session_response.raise_for_status()
    return session_response.json()["token"]["access_token"]


async def main():
    """
    Connects to a LiveKit room, sends a message to the agent,
    and verifies the interaction via the API.
    """
    logging.info("Starting LiveKit agent integration test...")

    try:
        # 1. Get Session Token and Thread ID
        logging.info(f"Authenticating as {TEST_USER_EMAIL}...")
        session_token = await get_session_token(TEST_USER_EMAIL, TEST_USER_PASSWORD)
        session_headers = {"Authorization": f"Bearer {session_token}"}
        
        # Decode the token to get the thread_id
        payload = jwt.decode(session_token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        thread_id = payload.get("sub")
        if not thread_id:
            raise Exception("Test Failed: Could not extract thread_id from session token.")
            
        logging.info(f"Authentication successful. Using thread_id: {thread_id}")

        # 2. Connect to LiveKit Room using the thread_id as our identity
        CLIENT_IDENTITY = thread_id
        
        lk_token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(CLIENT_IDENTITY)
            .with_name("Integration Test Client")
            .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME))
            .to_jwt()
        )

        room = rtc.Room()

        @room.on("data_received")
        def on_data_received(data: rtc.DataPacket):
            response = data.data.decode("utf-8")
            logging.info(f"Received data from agent (or other participant): \"{response}\"")

        await room.connect(f"ws://{LIVEKIT_URL.split('//')[1]}", lk_token)
        logging.info(f"Connected to LiveKit room '{ROOM_NAME}'.")

        # 3. Send a message via LiveKit
        test_message = "Hello Morgana, what is the capital of France?"
        logging.info(f"Sending message to agent: \"{test_message}\"")
        await room.local_participant.publish_data(test_message)

        # 4. Wait and Verify via API
        logging.info("Waiting for 15 seconds for agent to process and respond...")
        await asyncio.sleep(15)

        logging.info("Verifying chat history via API...")
        history_response = requests.get(f"{API_BASE_URL}/chatbot/messages", headers=session_headers)
        history_response.raise_for_status()
        history = history_response.json()

        logging.info(f"API Chat History: {history}")

        messages = history.get("messages", [])
        if len(messages) < 2:
            raise Exception(f"Test Failed: Expected at least 2 messages in history, found {len(messages)}")

        last_message = messages[-1]
        if last_message["role"] != "assistant" or "paris" not in last_message["content"].lower():
            raise Exception(f"Test Failed: Agent response was not correct. Last message: {last_message}")

        logging.info("\033[92mSUCCESS: Agent responded correctly and history was updated.\033[0m")

    except Exception as e:
        logging.error(f"\033[91mTEST FAILED: {e}\033[0m", exc_info=True)
    finally:
        if room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            logging.info("Disconnecting from room...")
            await room.disconnect()
        logging.info("Test finished.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # To run this test, ensure the main API and the run_livekit.py agent are running.
    asyncio.run(main())
