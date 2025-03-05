#!/bin/bash
# setup_cron.sh
# Setup cron job to run the message transfer script daily at 2 AM

# Base directory for the Sara project
SARA_DIR="/home/david/Sara"

# Path to the transfer script
SCRIPT_PATH="$SARA_DIR/scripts/daily_message_transfer.py"

# Path to the virtual environment
VENV_PATH="$SARA_DIR/env"

# Make sure script directory exists
mkdir -p "$SARA_DIR/scripts/"
mkdir -p "$SARA_DIR/logs/"

# Copy the script to the right location
echo "Copying script to $SCRIPT_PATH"
cp daily_message_transfer.py $SCRIPT_PATH
chmod +x $SCRIPT_PATH

# Create a wrapper script that activates the virtual environment
WRAPPER_SCRIPT="$SARA_DIR/scripts/run_with_venv.sh"
cat > $WRAPPER_SCRIPT << 'EOF'
#!/bin/bash
# Wrapper script to run Python script with virtual environment

# Get the directory of this script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SARA_DIR="$(dirname "$SCRIPT_DIR")"

# Activate the virtual environment and run the Python script
source "$SARA_DIR/env/bin/activate"
python3 "$SARA_DIR/scripts/daily_message_transfer.py"
EOF

# Make the wrapper script executable
chmod +x $WRAPPER_SCRIPT

# Create or update the crontab entry to use the wrapper script
CRON_JOB="0 2 * * * $WRAPPER_SCRIPT >> $SARA_DIR/logs/daily_transfer_cron.log 2>&1"

# Check if the cron job already exists
EXISTING_CRON=$(crontab -l 2>/dev/null | grep "$SCRIPT_PATH\|$WRAPPER_SCRIPT" || echo "")

if [ -n "$EXISTING_CRON" ]; then
    echo "Cron job already exists. Updating..."
    # Remove existing job and add new one
    (crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH\|$WRAPPER_SCRIPT") | crontab -
fi

# Add the new job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "Cron job set up to run daily at 2 AM"
echo "Job: $CRON_JOB"
echo ""
echo "To verify, run: crontab -l"