import os
import requests
import time

API_URL = "http://localhost:8000"
TEST_PDF = "rag_chatbot/data/Chol-hwan Kang, Pierre Rigoulot - The Aquariums of_230415_123826.pdf"

def test_full_flow():
    print("--- Starting Full Flow Test ---")
    
    # 1. Check if server is up
    try:
        res = requests.get(f"{API_URL}/books")
        print(f"Server Check: Success (Status {res.status_code})")
    except Exception as e:
        print(f"Server Check: FAILED. Is the server running? {e}")
        return

    # 2. Test Upload
    print("\nTesting Upload...")
    if not os.path.exists(TEST_PDF):
        print(f"Error: {TEST_PDF} not found for testing.")
        return

    with open(TEST_PDF, "rb") as f:
        files = {"file": (os.path.basename(TEST_PDF), f, "application/pdf")}
        res = requests.post(f"{API_URL}/upload", files=files)
    
    if res.status_code == 200:
        book_data = res.json()
        col_id = book_data["collection_id"]
        print(f"Upload: Success! Collection ID: {col_id}")
    else:
        print(f"Upload: FAILED (Status {res.status_code}): {res.text}")
        return

    # 3. Test Chat
    print(f"\nTesting Chat for {col_id}...")
    chat_payload = {"message": "What is the main theme of this book?"}
    res = requests.post(f"{API_URL}/chat/{col_id}", json=chat_payload)
    
    if res.status_code == 200:
        print(f"Chat: Success! AI response: {res.json()['answer']}")
    else:
        print(f"Chat: FAILED (Status {res.status_code}): {res.text}")

if __name__ == "__main__":
    test_full_flow()
