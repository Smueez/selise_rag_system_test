"""
Test script for the agent
"""
import requests
import json

BASE_URL = "http://localhost:8000/api"


def test_query_non_streaming():
    """Test non-streaming query"""
    print("Testing non-streaming query...")

    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "query": "What is this document about?",
            "conversation_history": []
        }
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_query_streaming():
    """Test streaming query"""
    print("\nTesting streaming query...")

    response = requests.post(
        f"{BASE_URL}/query/stream",
        json={
            "query": "What are the main topics covered in this document?",
            "conversation_history": []
        },
        stream=True
    )

    print(f"Status: {response.status_code}")
    print("Streaming response:")

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                data = json.loads(decoded_line[6:])
                print(f"  {data}")


if __name__ == "__main__":
    # Make sure server is running
    print("Make sure the server is running: uvicorn app.main:app --reload\n")

    # Test non-streaming
    test_query_non_streaming()

    # Test streaming
    test_query_streaming()