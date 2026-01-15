#!/bin/bash
cd "$(dirname "$0")"

# Activate your Python environment (adjust path as needed)
# source /path/to/your/venv/bin/activate

# Path to upscale models - change this to your models folder
export UPSCALE_MODELS_PATH="${UPSCALE_MODELS_PATH:-./models}"

python server.py
