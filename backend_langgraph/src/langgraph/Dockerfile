FROM python:3.13.2-slim

# Set the primary working directory and Python path
WORKDIR /app
ENV PYTHONPATH /app

# Set non-sensitive environment variables
ARG APP_ENV=production
ARG POSTGRES_URL

ENV APP_ENV=${APP_ENV} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POSTGRES_URL=${POSTGRES_URL}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && pip install --upgrade pip \
    && pip install uv \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project context
COPY . .

# --- DIAGNOSTIC STEP ---
# List the contents of /app to ensure `src` is present
RUN echo "--- Listing contents of /app ---" && ls -R /app

# Move to the app directory to install its specific dependencies
WORKDIR /app/src/langgraph

# Create venv and install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt && \
    uv pip install -e .

# Go back to the main project workdir
WORKDIR /app

# Make entrypoint script executable
RUN chmod +x /app/src/langgraph/scripts/docker-entrypoint.sh

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Create log directory
RUN mkdir -p /app/logs

# Default port
EXPOSE 8000

# Log the environment we're using
RUN echo "Using ${APP_ENV} environment"

# Command to run the application
ENTRYPOINT ["/app/src/langgraph/scripts/docker-entrypoint.sh"]
CMD ["/app/src/langgraph/.venv/bin/uvicorn", "src.langgraph.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
