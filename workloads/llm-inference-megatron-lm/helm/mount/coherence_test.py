# AMD CORPORATION 2025. All rights reserved.
import json
import sys

import requests

# Define test questions and expected answers
test_questions = [
    {"question": "Who lives in houses?", "expected": ["humans", "people", "families"], "score": 0},
    {"question": "What is an apple?", "expected": ["fruit", "a fruit", "type of fruit"], "score": 0},
    {
        "question": "What color is the sky on a clear day?",
        "expected": ["blue", "light blue", "the sky is blue"],
        "score": 0,
    },
    {
        "question": "What do plants need to grow?",
        "expected": ["water", "sunlight", "sun", "light", "soil", "nutrients"],
        "score": 0,
    },
    {"question": "What is the capital of France?", "expected": ["paris"], "score": 0},
]


def check_coherence(answer, expected):
    """Check if the answer contains any of the expected keywords (case-insensitive)"""
    answer_lower = answer.lower()
    for exp in expected:
        if exp.lower() in answer_lower:
            return True
    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the server URL as an argument")
        sys.exit(1)

    url = sys.argv[1]
    url = "http://" + url + "/api"
    headers = {"Content-Type": "application/json"}

    # Ask for number of tokens to generate
    # while True:
    #     try:
    #         tokens_to_generate = int(input("Enter number of tokens to generate for each question: "))
    #         if tokens_to_generate > 0:
    #             break
    #         print("Please enter a positive integer.")
    #     except ValueError:
    #         print("Please enter a valid integer.")

    tokens_to_generate = 52
    total_score = 0

    print(f"\nStarting coherence test with {len(test_questions)} questions...\n")

    for i, test in enumerate(test_questions, 1):
        print(f"Question {i}: {test['question']}")

        data = {"prompts": [test["question"]], "tokens_to_generate": tokens_to_generate}
        response = requests.put(url, data=json.dumps(data), headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.json()['message']}")
            answer = "[ERROR]"
        else:
            answer = response.json()["text"][0]
            print(f"Model's answer: {answer}")

            # Check coherence
            if check_coherence(answer, test["expected"]):
                test["score"] = 1
                total_score += 1
                print("✓ Coherent answer (1 point)")
            else:
                print("✗ Incoherent answer (0 points)")

        print(f"Expected concepts: {', '.join(test['expected'])}")
        print()  # Add empty line between questions

    print(f"\nTest completed. Total score: {total_score}/5")
    print("Detailed results:")
    for i, test in enumerate(test_questions, 1):
        print(f"Q{i}: {test['score']} point - {test['question']}")
