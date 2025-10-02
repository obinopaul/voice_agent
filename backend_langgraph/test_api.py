
import requests
import uuid
import time

# The base URL for our running API service
BASE_URL = "http://localhost:8000"

# The main agent endpoint
AGENT_URL = f"{BASE_URL}/agent/invoke"

# Generate a unique conversation ID for this test run
THREAD_ID = str(uuid.uuid4())


def run_test(description: str, response: requests.Response, expected_status: int):
    """Helper function to print test results."""
    print(f"--- Test: {description} ---")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == expected_status:
        print(f"\033[92mPASS: Received expected status code {expected_status}\033[0m")
        try:
            print(f"Response JSON: {response.json()}")
        except requests.exceptions.JSONDecodeError:
            print(f"Response Text: {response.text}")
    else:
        print(f"\033[91mFAIL: Expected status code {expected_status}, but got {response.status_code}\033[0m")
        print(f"Response: {response.text}")
        exit(1) # Exit on failure
    print("-" * 30)

def main():
    """Runs a sequence of tests against the API."""
    print("""
    ============================================
    API and Conversation Memory Test
    ============================================
    This script will test:
    1. The basic health of the API.
    2. The agent's ability to respond to a first-time query.
    3. The agent's memory by asking a follow-up question.
    
    Conversation Thread ID for this run: {THREAD_ID}
    """)

    # Test 1: Health Check
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        run_test("Health Check", health_response, 200)
    except requests.ConnectionError as e:
        print(f"\033[91mFAIL: Could not connect to the API at {BASE_URL}. Is it running?\033[0m")
        print(e)
        exit(1)

    # Test 2: First Interaction (Introduce information)
    print("\nStep 2: Asking the agent to remember my name...")
    first_payload = {
        "input": {
            "messages": [{"type": "human", "content": "Hello! My name is Paul."}],
            "active_agent": "Smol_Agent"
        },
        "config": {"configurable": {"thread_id": THREAD_ID, "assistant_id": "default"}}
    }
    first_response = requests.post(AGENT_URL, json=first_payload)
    run_test("First Interaction", first_response, 200)

    # Give the agent a moment
    time.sleep(2)

    # Test 3: Second Interaction (Test memory)
    print("\nStep 3: Asking the agent to recall my name...")
    second_payload = {
        "input": {
            "messages": [{"type": "human", "content": "What is my name?"}],
            "active_agent": "Smol_Agent"
        },
        "config": {"configurable": {"thread_id": THREAD_ID, "assistant_id": "default"}}
    }
    second_response = requests.post(AGENT_URL, json=second_payload)
    run_test("Memory Test", second_response, 200)

    # Verification
    try:
        response_json = second_response.json()
        final_content = response_json.get("output", {}).get("messages", [{}])[-1].get("content", "")
        if "paul" in final_content.lower():
            print("\033[92m\nVERIFICATION SUCCESS: Agent correctly remembered the name 'Paul'.\033[0m")
        else:
            print("\033[91m\nVERIFICATION FAILED: Agent did not recall the name 'Paul' in its response.\033[0m")
    except Exception as e:
        print(f"\033[91m\nVERIFICATION FAILED: Could not parse final response. Error: {e}\033[0m")

if __name__ == "__main__":
    main()
