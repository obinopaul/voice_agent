# livekit.py
import logging
import os
import asyncio
import base64
from dotenv import load_dotenv
import httpx

# LangGraph and LiveKit imports
from langgraph.pregel.remote import RemoteGraph
from langgraph.pregel import Pregel
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from livekit.agents import Agent, AgentSession, llm
from livekit.agents.llm import ChatContext, ImageContent, ChatMessage
from livekit.plugins import deepgram, silero, hume
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit import rtc
from langchain_core.messages import BaseMessageChunk, AIMessage, HumanMessage, SystemMessage

# Import the new authentication utility
from src.langgraph.app.utils.user_auth import get_session_token

# Load environment variables
load_dotenv()
logger = logging.getLogger("voice-agent")


# ===================================================================================
# Authenticated Remote Graph
# ===================================================================================
class AuthenticatedRemoteGraph(RemoteGraph):
    """A RemoteGraph that includes an Authorization header in its requests."""
    def __init__(self, url: str, session_token: str):
        super().__init__(url=url)
        self.session_token = session_token

    def _create_client(self, **kwargs) -> httpx.AsyncClient:
        """Overrides the client creation to add the auth header."""
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.session_token}"},
            **kwargs
        )

# ===================================================================================
# LangGraph Adapter to Bridge LiveKit and LangGraph/LangServe
# This class handles the communication between LiveKit's chat interface and your agent.
# No changes are needed here.
# ===================================================================================

class LangGraphAdapter(llm.LLM):
    def __init__(self, graph: Pregel, config: dict):
        super().__init__()
        self._graph = graph
        self._config = config

    def chat(self, *, chat_ctx: llm.ChatContext, **kwargs) -> llm.LLMStream:
        return LangGraphStream(self, chat_ctx=chat_ctx, graph=self._graph)

class LangGraphStream(llm.LLMStream):
    def __init__(self, llm_adapter: LangGraphAdapter, *, chat_ctx: llm.ChatContext, graph: Pregel):
        super().__init__(llm_adapter, chat_ctx=chat_ctx)
        self._graph = graph
        self._llm_adapter = llm_adapter

    async def _run(self):
        # Convert LiveKit message history to LangChain format
        langchain_messages = self._chat_ctx_to_langchain_messages()
        
        # The input to the graph is a dictionary with a 'messages' key
        graph_input = {"messages": langchain_messages}

        try:
            # Stream the response from the remote graph, passing the full config
            async for chunk in self._graph.astream(
                graph_input,
                config=self._llm_adapter._config,
                stream_mode="messages"
            ):
                if chunk and isinstance(chunk, list) and len(chunk) > 0:
                    message = chunk[0]
                    if livekit_chunk := self._to_livekit_chunk(message):
                        self._event_ch.send_nowait(livekit_chunk)
        except Exception as e:
            logger.error(f"Error streaming from LangGraph: {e}", exc_info=True)

    def _chat_ctx_to_langchain_messages(self) -> list:
        messages = []
        for msg in self._chat_ctx.messages:
            if msg.role == llm.ChatRole.ASSISTANT:
                messages.append(AIMessage(content=msg.text))
            elif msg.role == llm.ChatRole.USER:
                # This part can be expanded to handle vision/image content
                messages.append(HumanMessage(content=msg.text))
        return messages

    @staticmethod
    def _to_livekit_chunk(msg) -> llm.ChatChunk | None:
        if not msg or not isinstance(msg.content, str):
            return None
        return llm.ChatChunk(delta=llm.ChoiceDelta(role="assistant", content=msg.content))


# ===================================================================================
# VisionAssistant for handling video streams
# This class is well-structured and requires no changes.
# ===================================================================================
class VisionAssistant(Agent):
    """Enhanced agent with vision capabilities for processing camera and screen sharing."""
    def __init__(self):
        super().__init__(instructions="")
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        # ... (rest of the class is unchanged)

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Add visual context to the conversation when available."""
        if self._latest_frame:
            try:
                # Pass the actual frame as ImageContent so LLM can see it (per docs)
                new_message.content.append(ImageContent(image=self._latest_frame))
                logger.info("Added latest video frame to conversation context")
            except Exception as e:
                logger.warning(f"Failed to process video frame: {e}")
            finally:
                self._latest_frame = None

    def _create_video_stream(self, track: rtc.Track, source: rtc.TrackSource | int | str | None):
        if self._video_stream is not None:
            asyncio.create_task(self._video_stream.aclose())

        self._video_stream = rtc.VideoStream(track)
        logger.info(f"Created VideoStream for track {getattr(track, 'sid', None)}")
        
        async def read_stream():
            try:
                async for event in self._video_stream:
                    self._latest_frame = event.frame
            except Exception as e:
                logger.error(f"Error reading video stream: {e}")

        task = asyncio.create_task(read_stream())
        self._tasks.append(task)


# ===================================================================================
# LiveKit Entrypoint - THIS SECTION IS UPDATED
# ===================================================================================
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    participant = await ctx.wait_for_participant()

    # --- Authenticate and get session token ---
    try:
        logger.info("Authenticating and creating a new session...")
        session_token = get_session_token()
        logger.info("Successfully obtained session token.")
    except Exception as e:
        logger.error(f"Failed to authenticate and create session: {e}", exc_info=True)
        # Optionally, you could say something to the user here before exiting
        await ctx.disconnect()
        return

    # --- ROBUST SESSION & THREAD MANAGEMENT ---
    session_id = participant.sid
    user_id = participant.identity
    thread_id = user_id

    logger.info(f"Initialized agent for user '{user_id}' in session '{session_id}'")
    logger.info(f"Conversation memory will be managed under thread_id: '{thread_id}'")

    # Connect to your running LangServe agent endpoint
    langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:8000/chatbot")
    logger.info(f"Connecting to LangGraph agent at: {langgraph_url}")
    
    # Use the authenticated graph
    graph = AuthenticatedRemoteGraph(url=langgraph_url, session_token=session_token)

    # --- CORRECTLY CONFIGURED FOR YOUR BACKEND ---
    agent_config = {
        "configurable": {
            "thread_id": thread_id
        },
        "metadata": {
            "user_id": user_id,
            "session_id": session_id
        }
    }

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=LangGraphAdapter(graph, config=agent_config),
        tts=hume.TTS(),
        turn_detection=EnglishModel(),
    )

    vision_agent = VisionAssistant()
    await session.start(agent=vision_agent, room=ctx.room)
    await session.say("Hey, I'm online. How can I help?", allow_interruptions=True)

# ===================================================================================

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
