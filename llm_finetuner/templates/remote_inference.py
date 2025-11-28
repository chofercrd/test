import argparse
import time
import random

# This is a Mock Inference Script

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)

    args = parser.parse_args()

    # Simulate loading model time
    time.sleep(2)

    responses = [
        "That is an interesting question. Based on my fine-tuning, I think...",
        "Here is the code you requested: def hello_world(): print('Hello')",
        "I am a large language model trained by you.",
        "The capital of France is Paris."
    ]

    response = f"Simulated Response from {args.model_path}:\n\n"
    response += random.choice(responses)
    response += f"\n\n(Original Prompt: {args.prompt})"

    print(response)

if __name__ == "__main__":
    main()
