.PHONY: help download-files dev

# Default target
help:
	@echo "Available targets:"
	@echo "  make download-files  - Download required voice models (VAD, etc.)"
	@echo "  make dev             - Run the LiveKit agent"
	@echo "  make help            - Show this help message"

# Download required files
download-files:
	@echo "Downloading necessary models for voice processing..."
	uv run python -m src.agent download-files

# Run the LiveKit agent in development mode
dev:
	@echo "Starting the LiveKit agent..."
	uv run python -m src.agent dev