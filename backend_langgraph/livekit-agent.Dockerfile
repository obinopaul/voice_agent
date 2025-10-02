# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv, the package manager used by the project
RUN pip install uv

# Copy the entire project context
COPY . .

# Install dependencies using the correct 'uv sync' command
RUN uv sync --no-cache

# Download voice models using the command from the README
RUN uv run python -m src.livekit_main.src.agent download-files

# --- Run the Agent ---
# Set the command to run your LiveKit agent when the container starts
CMD ["uv", "run", "python", "-m", "src.livekit_main.src.agent", "dev"]
