#!/bin/bash
# setup_cron.sh
# Setup cron job to run the message transfer script daily at 2 AM

# Path to the transfer script
SCRIPT_PATH="/home/david/Sara/scripts/daily_message_transfer.py"

# Make sure script directory exists
mkdir -p /home/david/Sara/scripts/
mkdir -p /home/david/Sara/logs/

# Copy the script to the right location
echo "Copying script to $SCRIPT_PATH"
cp daily_message_transfer.py $SCRIPT_PATH
chmod +x $SCRIPT_PATH

# Create or update the crontab entry
CRON_JOB="0 2 * * * /usr/bin/python3 $SCRIPT_PATH >> /home/david/Sara/logs/daily_transfer_cron.log 2>&1"

# Check if the cron job already exists
EXISTING_CRON=$(crontab -l 2>/dev/null | grep "$SCRIPT_PATH" || echo "")

if [ -n "$EXISTING_CRON" ]; then
    echo "Cron job already exists. Updating..."
    # Remove existing job and add new one
    (crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH") | crontab -
fi

# Add the new job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "Cron job set up to run daily at 2 AM"
echo "Job: $CRON_JOB"
echo ""
echo "To verify, run: crontab -l"