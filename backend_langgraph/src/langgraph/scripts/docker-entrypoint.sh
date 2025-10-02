#!/bin/bash
set -e

# The environment variables are now expected to be passed by Docker Compose.
# This script will simply validate them and execute the command.

# Print initial environment values
echo "Starting with these environment variables provided by Docker Compose:"
echo "APP_ENV: ${APP_ENV:-development}"

# Check required sensitive environment variables
required_vars=("JWT_SECRET_KEY" "LLM_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "ERROR: The following required environment variables are missing:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo "Please provide these variables in the .env file in your project root."
    exit 1
fi

# Print final environment info
echo -e "\nFinal environment configuration:"
echo "Environment: ${APP_ENV:-development}"

# Show only the part after @ for database URL (for security)
if [[ -n "$POSTGRES_URL" && "$POSTGRES_URL" == *"@"* ]]; then
    DB_DISPLAY=$(echo "$POSTGRES_URL" | sed 's/.*@/@/')
    echo "Database URL: *********$DB_DISPLAY"
else
    echo "Database URL: ${POSTGRES_URL:-Not set}"
fi

echo "LLM_Model: ${LLM_MODEL:-Not set}"
echo "Debug Mode: ${DEBUG:-false}"

# Execute the CMD
exec "$@"
