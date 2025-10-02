# Production-Ready Voice Agent

This project is a voice-based conversational AI agent built using LiveKit, DeepGram, and OpenAI.

## Architecture

The system consists of three main parts:

1.  **LiveKit Server**: Acts as the central media server, routing audio between the user and the agent.
2.  **Backend Agent (Python)**: A Python application that listens to the user, transcribes their speech, generates a response using an LLM, and speaks the response back.
3.  **Frontend Client (Next.js)**: A web application that provides the user interface, handles the connection to the LiveKit room, and displays the conversation.

## Getting Started

### Prerequisites

- Node.js (v18 or later)
- Python 3.11+ and `uv` (https://github.com/astral-sh/uv)
- Docker (for containerized deployment)
- Access keys for LiveKit, OpenAI, and DeepGram.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd voice-agent
    ```

2.  **Configure Environment Variables:**

    Copy the example environment file and fill in your API keys:

    ```bash
    cp .env.example .env
    ```

    You will need to do the same for the frontend:

    ```bash
    cp frontend/.env.local.example frontend/.env.local
    ```

### Backend Usage

- Navigate to the backend directory: `cd backend`
- Install dependencies: `uv sync`

- **Download VAD models:**
  ```bash
  uv run python download.py
  ```

- **Run the agent:**
  ```bash
  uv run python -m src.main dev
  ```

## Frontend Usage

- Navigate to the frontend directory: `cd frontend`
- Install dependencies: `npm install`
- Run the development server: `npm run dev`
- Open your browser to `http://localhost:3000`