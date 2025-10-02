# Morgana - Voice Multi-Agent System

Welcome to the backend of Morgana, a sophisticated, voice-enabled, multi-agentic system. This backend is designed to be a robust, production-ready platform for creating and orchestrating conversational AI agents.

## Features

- **Multi-Agentic Core**: Built on LangGraph, allowing for complex and stateful agentic workflows.
- **Voice-Enabled**: Integrated with LiveKit for real-time, low-latency voice communication, including text-to-speech (TTS) and speech-to-text (STT).
- **Extensible Toolset**: Includes several MCP (Multi-Agent Collaboration Protocol) tools, such as a browser agent and a Microsoft tools agent.
- **Production-Ready**: The entire system is containerized with Docker, allowing for a one-click deployment of all components.
- **Monitoring**: Includes configurations for Prometheus and Grafana for monitoring the system (not enabled by default in the main compose file).

## Architecture

The Morgana backend consists of several services, all orchestrated by Docker Compose:

- **LangGraph API (`api`)**: The core of the system, this is a FastAPI application that serves the LangGraph agents. It handles the main logic and orchestration of the different agents and tools.
- **PostgreSQL Database (`db`)**: A persistent database for the LangGraph API, used for storing conversation history, user data, and other application state.
- **Redis (`redis`)**: A Redis instance used for caching and as a message broker for the LangGraph API.
- **Microsoft MCP Agent (`mcp_tools_agent`)**: A separate service that provides tools for interacting with Microsoft services (e.g., Office 365, Azure). It communicates with the main API.
- **Browser MCP Agent (`mcp_browser_agent`)**: A service that gives the agents the ability to control a web browser, enabling them to perform tasks like web scraping, form filling, and more.
- **LiveKit (`livekit`)**: A real-time communication server that handles the voice part of the application. It connects to the LangGraph API to provide a seamless voice experience.

All services are connected through a Docker network, allowing them to communicate with each other.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker
- Docker Compose

### Installation

1.  **Clone the repository** (if you haven't already):

    ```bash
    git clone <your-repository-url>
    cd morgana/backend
    ```

2.  **Set up the environment variables**:

    Copy the example environment file:

    ```bash
    cp .env.example .env
    ```

    Now, open the `.env` file and fill in the required API keys and other configuration values. At a minimum, you will need to provide your `OPENAI_API_KEY`.

3.  **Build and run the services**:

    The following command will build the Docker images for all the services and start them in detached mode:

    ```bash
    docker compose up --build -d
    ```

    <!-- ensure to download all the livekit files
    ```
    uv run -m backend.src.run_livekit download-files
    ``` -->

    This command will start all the services defined in the `compose.yml` file. The `--build` flag ensures that the images are rebuilt if there are any changes in the code. The `-d` flag runs the containers in the background.

4.  **Verify the services are running**:

    You can check the status of the running containers with:

    ```bash
    docker compose ps
    ```

    You should see all the services in the `Up` state.

## Services and Ports

Once the system is running, the following services will be accessible on your local machine:

| Service               | Port(s)               | Description                                      |
| --------------------- | --------------------- | ------------------------------------------------ |
| LangGraph API         | `8000`                | The main FastAPI application.                    |
| Microsoft MCP Agent   | `8002`                | The Microsoft tools agent.                       |
| Browser MCP Agent     | `8003`                | The browser agent.                               |
| LiveKit (Signaling)   | `7880`                | LiveKit WebSocket signaling.                     |
| LiveKit (TURN/TLS)    | `7881`                | LiveKit TURN/TLS server.                         |
| LiveKit (Media)       | `7882/udp`            | LiveKit UDP for media traffic.                   |

## Usage

Once the system is running, you can start interacting with it:

- **API**: The LangGraph API is available at `http://localhost:8000`. You can access the auto-generated documentation at `http://localhost:8000/docs`.
- **LiveKit**: You can connect a LiveKit client to `ws://localhost:7880` to interact with the voice agent. You will need to use the `devkey` and `secret` for authentication.

## Development

### Stopping the services

To stop all the running services, use the following command:

```bash
docker compose down
```

### Viewing logs

You can view the logs for all services with:

```bash
docker compose logs -f
```

To view the logs for a specific service, add the service name at the end:

```bash
docker compose logs -f api
```

### Connecting to the database

If you need to inspect the database, you can connect to it using any PostgreSQL client with the following credentials (from the `.env` file):

- **Host**: `localhost`
- **Port**: `5432` (you would need to expose it in the `compose.yml` first)
- **Database**: `morgana`
- **User**: `morgana`
- **Password**: `morgana`

Alternatively, you can connect to the running container:

```bash
docker compose exec -it db psql -U morgana
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
