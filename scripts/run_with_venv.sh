#!/bin/bash
# Wrapper script to run Python script with virtual environment

# Get the directory of this script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SARA_DIR="$(dirname "$SCRIPT_DIR")"

# Activate the virtual environment and run the Python script
source "$SARA_DIR/env/bin/activate"
python3 "$SARA_DIR/scripts/daily_message_transfer.py"
