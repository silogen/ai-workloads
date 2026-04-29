import argparse
import json
from datetime import datetime

import requests  # type: ignore

parser = argparse.ArgumentParser()
parser.add_argument(
    "--url",
    type=str,
    required=True,
    help="TorchServe VACE inference endpoint, e.g. http://localhost:8080/predictions/vace",
)
parser.add_argument("--prompt", type=str, required=True, help="Text prompt for VACE generation")
parser.add_argument(
    "--output", type=str, default=None, help="Path to save the generated video (defaults to vace_<timestamp>.mp4)"
)
parser.add_argument("--task", type=str, required=False, help="VACE task to run (e.g. outpainting, inpainting and more)")
parser.add_argument("--video", type=str, required=False, help="Path to the input video file")
parser.add_argument("--mode", type=str, required=False, help="Mode of certain tasks")
args = parser.parse_args()

if args.task:
    # Build payload
    payload = {"prompt": args.prompt, "task": args.task, "mode": args.mode}

    # Read the video file into memory
    video_bytes = open(args.video, "rb").read()

    # Default output filename if not provided
    if args.output:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"vace_{ts}.mp4"

    # Send as multipart/form-data so we can include raw bytes
    start_time = datetime.now()
    resp = requests.post(
        args.url,
        data={"body": json.dumps(payload)},
        files={"video": ("input.mp4", video_bytes, "video/mp4")},
        stream=True,
    )
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"Request took {elapsed_time:.2f} seconds")

else:
    # Build payload
    payload = {"prompt": args.prompt}

    # Default output filename if not provided
    if args.output:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"vace_{ts}.mp4"

    # Send request and stream response
    # Add timing
    start_time = datetime.now()
    resp = requests.post(args.url, json=payload, stream=True)
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"Request took {elapsed_time:.2f} seconds")

resp.raise_for_status()

with open(out_path, "wb") as f:
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print(f"Saved generated video to {out_path}")
