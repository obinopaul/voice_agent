

import requests
import uuid
import json

# The base URL for our running API service
BASE_URL = "http://localhost:8000/api/v1"

def run_test(description: str, response: requests.Response, expected_status: int):
    """Helper function to print test results."""
    print(f"--- Test: {description} ---")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == expected_status:
        print(f"\033[92mPASS: Received expected status code {expected_status}\033[0m")
        try:
            response_json = response.json()
            print(f"Response JSON: {response_json}")
            return response_json
        except requests.exceptions.JSONDecodeError:
            print(f"Response Text: {response.text}")
            return response.text
    else:
        print(f"\033[91mFAIL: Expected status code {expected_status}, but got {response.status_code}\033[0m")
        print(f"Response: {response.text}")
        exit(1) # Exit on failure

def get_session_token(user_headers):
    """Gets a new session token."""
    print("\n--- Creating a new chat session... ---")
    session_response = requests.post(f"{BASE_URL}/auth/session", headers=user_headers)
    session_json = run_test("Create Session", session_response, 200)
    
    session_token = session_json.get("token", {}).get("access_token")
    if not session_token:
        print("\033[91mFAIL: Could not retrieve session token from session response.\033[0m")
        exit(1)
        
    print(f"\033[92mSUCCESS: Obtained session token.\033[0m")
    return {"Authorization": f"Bearer {session_token}"}

def main():
    """Runs a sequence of tests against the API."""
    print("""
    ============================================
    Full API Test Suite (Corrected)
    ============================================
    This script will test:
    1. API Health Check
    2. User Registration and Login
    3. Multi-Session Chat Conversations
    """)

    # Test 1: Health Check
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        run_test("Health Check", health_response, 200)
    except requests.ConnectionError as e:
        print(f"\033[91mFAIL: Could not connect to the API at {BASE_URL}. Is it running?\033[0m")
        print(e)
        exit(1)

    # Test 2: User Registration and Login
    email = f"testuser_{uuid.uuid4()}@example.com"
    password = "A_Secure_Password_123_$"
    
    print("\nStep 2a: Registering a new user...")
    register_payload = {"email": email, "password": password}
    register_response = requests.post(f"{BASE_URL}/auth/register", json=register_payload)
    run_test("User Registration", register_response, 200)

    print("\nStep 2b: Logging in to get a user token...")
    login_payload = {"username": email, "password": password}
    login_response = requests.post(f"{BASE_URL}/auth/login", data=login_payload)
    login_json = run_test("User Login", login_response, 200)
    
    user_token = login_json.get("access_token")
    if not user_token:
        print("\033[91mFAIL: Could not retrieve user access token from login response.\033[0m")
        exit(1)
    
    print(f"\033[92mSUCCESS: Obtained user access token.\033[0m")
    user_headers = {"Authorization": f"Bearer {user_token}"}

    # Test 3: Multi-Session Chat
    print("\nStep 3: Testing multi-session chat conversations...")

    # --- SESSION 1 ---
    session_1_headers = get_session_token(user_headers)
    
    print("\n--- Sending message to Session 1 ---")
    chat_payload_1 = {"messages": [{"role": "user", "content": "My favorite color is blue."}]}
    chat_response_1 = requests.post(f"{BASE_URL}/chatbot/chat", json=chat_payload_1, headers=session_1_headers)
    run_test("Chat in Session 1", chat_response_1, 200)

    # --- SESSION 2 ---
    session_2_headers = get_session_token(user_headers)

    print("\n--- Sending message to Session 2 ---")
    chat_payload_2 = {"messages": [{"role": "user", "content": "What is the capital of Germany?"}]}
    chat_response_2 = requests.post(f"{BASE_URL}/chatbot/chat", json=chat_payload_2, headers=session_2_headers)
    run_test("Chat in Session 2", chat_response_2, 200)
    
    # --- VERIFICATION ---
    print("\nStep 4: Verifying conversation isolation...")

    # Verify history of Session 1
    print("\n--- Retrieving history for Session 1 ---")
    history_1_response = requests.get(f"{BASE_URL}/chatbot/messages", headers=session_1_headers)
    history_1_json = run_test("Get History for Session 1", history_1_response, 200)
    
    history_1_text = json.dumps(history_1_json)
    if "blue" in history_1_text and "Germany" not in history_1_text:
        print("\033[92mVERIFICATION SUCCESS: Session 1 history is correct and isolated.\033[0m")
    else:
        print("\033[91mVERIFICATION FAILED: Session 1 history is incorrect or not isolated.\033[0m")

    # Verify history of Session 2
    print("\n--- Retrieving history for Session 2 ---")
    history_2_response = requests.get(f"{BASE_URL}/chatbot/messages", headers=session_2_headers)
    history_2_json = run_test("Get History for Session 2", history_2_response, 200)

    history_2_text = json.dumps(history_2_json)
    if "Germany" in history_2_text and "blue" not in history_2_text:
        print("\033[92mVERIFICATION SUCCESS: Session 2 history is correct and isolated.\033[0m")
    else:
        print("\033[91mVERIFICATION FAILED: Session 2 history is incorrect or not isolated.\033[0m")
        
    # Verify history of Session 1 again to ensure it hasn't changed
    print("\n--- Retrieving history for Session 1 again ---")
    history_1_again_response = requests.get(f"{BASE_URL}/chatbot/messages", headers=session_1_headers)
    run_test("Get History for Session 1 Again", history_1_again_response, 200)


if __name__ == "__main__":
    main()
