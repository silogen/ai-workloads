import argparse

import requests  # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument(
    "--url",
    type=str,
    required=True,
    help="Torchserve inference endpoint for xLSTM",
)
parser.add_argument("--prompt", type=str, required=True, help="Prompt for xLSTM")
args = parser.parse_args()

payload = {
    "prompt": args.prompt,
}

response = requests.post(args.url, json=payload).json()["output"]

print(f"Response: {response}")
