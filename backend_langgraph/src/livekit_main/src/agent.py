import asyncio
import json
import logging
import os
import uuid
from typing import AsyncIterable

import requests
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.agents import llm
from livekit.agents.llm import ChatContext, ChatMessage, ChatRole
from livekit.plugins import deepgram, silero
from livekit.plugins.turn_detector.english import EnglishModel

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi-livekit-agent")

# Suppress noisy logs from underlying libraries
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class FastAPIAdapterStream(llm.LLMStream):
    """
    This class handles the streaming of responses from the FastAPI backend.
    It's initialized with a specific, per-session adapter.
    """
    def __init__(self, adapter: 'FastAPIAdapter', chat_ctx: ChatContext):
        super().__init__(adapter)
        self._adapter = adapter
        self._chat_ctx = chat_ctx

    async def _run(self) -> None:
        messages = [{"role": msg.role.value, "content": msg.content}
                    for msg in self._chat_ctx.messages if msg.role != ChatRole.SYSTEM]

        payload = {"messages": messages}
        stream_url = f"{self._adapter.base_url}/chatbot/chat/stream"
        
        try:
            with requests.post(
                stream_url,
                json=payload,
                headers=self._adapter.session_headers,
                stream=True,
                timeout=120,
            ) as r:
                if r.status_code != 200:
                    logger.error(f"API Error: Status {r.status_code} - {r.text}")
                    self._event_ch.send_nowait(llm.ChatChunk(
                        delta=llm.ChoiceDelta(role=ChatRole.ASSISTANT, content="Sorry, an error occurred.")
                    ))
                    return

                for line in r.iter_lines():
                    if line.startswith(b'data:'):
                        data_str = line.decode('utf-8')[5:].strip()
                        if data_str:
                            try:
                                data_json = json.loads(data_str)
                                content = data_json.get("content", "")
                                if content:
                                    self._event_ch.send_nowait(llm.ChatChunk(
                                        delta=llm.ChoiceDelta(role=ChatRole.ASSISTANT, content=content)
                                    ))
                                if data_json.get("done"):
                                    break
                            except json.JSONDecodeError:
                                logger.warning(f"Could not decode JSON: {data_str}")

        except requests.RequestException as e:
            logger.error(f"Failed to connect to FastAPI stream: {e}")
            self._event_ch.send_nowait(llm.ChatChunk(
                delta=llm.ChoiceDelta(role=ChatRole.ASSISTANT, content="I'm having trouble connecting right now.")
            ))
        finally:
            self._event_ch.close()


class FastAPIAdapter(llm.LLM):
    """
    A lightweight LLM adapter that is initialized with a UNIQUE session token for a single conversation.
    """
    def __init__(self, *, base_url: str, session_token: str):
        super().__init__(capabilities={"streaming": True})
        self.base_url = base_url
        self.session_headers = {"Authorization": f"Bearer {session_token}"}

    def chat(self, *, chat_ctx: ChatContext, **kwargs) -> FastAPIAdapterStream:
        return FastAPIAdapterStream(self, chat_ctx)

async def agent_login(base_url: str, email: str, password: str) -> str | None:
    """
    Logs the agent in with its master credentials to get a long-lived user token.
    This token is then used to create individual sessions for each user.
    """
    logger.info("Attempting to log in agent...")
    try:
        # Register user (this is idempotent, fails safely if user exists)
        requests.post(f"{base_url}/auth/register", json={"email": email, "password": password}, timeout=10)
        
        # Login to get the user token
        login_payload = {"username": email, "password": password}
        login_response = requests.post(f"{base_url}/auth/login", data=login_payload, timeout=10)
        
        if login_response.status_code == 200:
            user_token = login_response.json().get("access_token")
            logger.info("✅ Agent successfully logged in.")
            return user_token
        else:
            logger.error(f"Agent login failed: {login_response.status_code} - {login_response.text}")
            return None
    except requests.RequestException as e:
        logger.error(f"Agent login request failed: {e}")
        return None

async def create_new_session(base_url: str, user_token: str) -> str | None:
    """
    Uses the agent's user token to create a new, unique session token for a participant.
    """
    logger.info("Creating new session for participant...")
    try:
        user_headers = {"Authorization": f"Bearer {user_token}"}
        session_response = requests.post(f"{base_url}/auth/session", headers=user_headers, timeout=10)
        
        if session_response.status_code == 200:
            session_token = session_response.json().get("token", {}).get("access_token")
            logger.info("✅ New session created successfully.")
            return session_token
        else:
            logger.error(f"Session creation failed: {session_response.status_code} - {session_response.text}")
            return None
    except requests.RequestException as e:
        logger.error(f"Session creation request failed: {e}")
        return None

async def handle_participant(participant: rtc.RemoteParticipant, base_url: str, user_token: str, vad_plugin):
    """
    This function is called for each new participant.
    It creates a unique session and handles the entire conversation lifecycle.
    """
    logger.info(f"New participant connected: {participant.identity}")
    
    # 1. Create a new isolated session for this participant
    session_token = await create_new_session(base_url, user_token)
    if not session_token:
        logger.error(f"Could not create session for participant {participant.identity}. Aborting.")
        # Optionally publish a data message to the user that something went wrong
        return

    # 2. Initialize the LLM adapter with the unique session token
    fastapi_adapter = FastAPIAdapter(base_url=base_url, session_token=session_token)

    # 3. Configure the agent session
    session = AgentSession(
        vad=vad_plugin,
        stt=deepgram.STT(model="nova-2", language="en-US"),
        llm=fastapi_adapter,
        tts=deepgram.TTS(model="aura-asteria-en"),
    )

    agent = Agent(instructions="You are a helpful voice AI assistant.")
    
    # Start the session and run the conversation loop
    task = asyncio.create_task(session.start(agent=agent, participant=participant))
    
    await asyncio.sleep(1)
    await session.say("Hello! I'm connected and ready to chat. How can I help you?", allow_interruptions=True)
    
    # Wait until the participant disconnects
    await participant.wait_for_disconnect()
    logger.info(f"Participant {participant.identity} disconnected. Ending session.")
    task.cancel() # Clean up the running session task

async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit agent worker.
    It logs the agent in and then listens for participants.
    """
    morgana_url = os.getenv("MORGANA_BACKEND_URL", "http://localhost:8000")
    base_url = f"{morgana_url}/api/v1"
    email = os.getenv("API_USER_EMAIL")
    password = os.getenv("API_USER_PASSWORD")
    
    # 1. Agent logs in once at startup to get its master user token
    user_token = await agent_login(base_url, email, password)
    if not user_token:
        logger.critical("FATAL: Could not get agent user token. Agent cannot start.")
        return

    # 2. Connect to the LiveKit room
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    logger.info(f"Connected to LiveKit room: {ctx.room.name}")

    # 3. Set up a handler for each participant who joins the room
    def on_participant_connected(participant: rtc.RemoteParticipant):
        asyncio.create_task(handle_participant(participant, base_url, user_token, ctx.proc.userdata["vad"]))

    # 4. Register the handler and process any participants already in the room
    ctx.room.on("participant_connected", on_participant_connected)
    for p in ctx.room.participants.values():
        on_participant_connected(p)
        
    # Keep the agent alive
    await asyncio.Event().wait()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )