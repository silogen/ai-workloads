#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change directory to the location of this script
cd "$(dirname "$0")"

pip install --no-cache-dir -r requirements.txt

python inference_service.py
