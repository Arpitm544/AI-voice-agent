from agent import process_interaction
import json
import os
from dotenv import load_dotenv

load_dotenv()

def run_test():
    if not os.getenv("GROQ_API_KEY"):
        print("Skipping test, GROQ_API_KEY is not set.")
        return
        
    messages = []
    
    # Test 1: Add To-Do
    print("User: Add 'buy groceries' to my to-do list.")
    messages.append({"role": "user", "content": "Add 'buy groceries' to my to-do list."})
    messages, response = process_interaction(messages)
    print(f"Agent: {response}\n")

    # Test 2: Remember Event
    print("User: Remember that my favorite color is blue.")
    messages.append({"role": "user", "content": "Remember that my favorite color is blue."})
    messages, response = process_interaction(messages)
    print(f"Agent: {response}\n")

    # Test 3: List To-Dos
    print("User: What is on my to-do list?")
    messages.append({"role": "user", "content": "What is on my to-do list?"})
    messages, response = process_interaction(messages)
    print(f"Agent: {response}\n")

    # Check data.json
    with open("data.json", "r") as f:
        data = json.load(f)
        print("DATA STORAGE:", json.dumps(data, indent=2))

if __name__ == "__main__":
    run_test()
