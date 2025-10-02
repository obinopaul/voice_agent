
import requests
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

def get_session_token(email: str = None, password: str = None) -> str:
    """
    Authenticates a user and creates a new chat session.

    Args:
        email: The user's email.
        password: The user's password.

    Returns:
        The session token.
    """
    if not email:
        email = os.getenv("TEST_USER_EMAIL", f"testuser_{uuid.uuid4()}@example.com")
    if not password:
        password = os.getenv("TEST_USER_PASSWORD", "A_Secure_Password_123_$")

    # Register a new user
    register_payload = {"email": email, "password": password}
    register_response = requests.post(f"{BASE_URL}/auth/register", json=register_payload)
    if register_response.status_code != 200:
        # If registration fails, it might be because the user already exists.
        # We can proceed to login.
        pass

    # Log in to get a user token
    login_payload = {"username": email, "password": password}
    login_response = requests.post(f"{BASE_URL}/auth/login", data=login_payload)
    login_response.raise_for_status()
    user_token = login_response.json()["access_token"]

    # Create a new chat session
    user_headers = {"Authorization": f"Bearer {user_token}"}
    session_response = requests.post(f"{BASE_URL}/auth/session", headers=user_headers)
    session_response.raise_for_status()
    session_token = session_response.json()["token"]["access_token"]

    return session_token
