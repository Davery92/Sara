#!/usr/bin/env python3
"""
Simple test script for sending notifications to an ntfy server.
"""

import requests
import argparse
import sys

def send_notification(server, topic, message, title=None, priority=None, tags=None):
    """
    Send a notification to an ntfy server.
    
    Args:
        server (str): The ntfy server URL (e.g., http://10.185.1.8:8888)
        topic (str): The notification topic
        message (str): The notification message
        title (str, optional): The notification title
        priority (int, optional): Priority level (1-5, with 5 being highest)
        tags (list, optional): List of tags for the notification
    
    Returns:
        dict: Server response
    """
    # Prepare URL
    url = f"{server}/{topic}"
    
    # Prepare headers
    headers = {}
    if title:
        headers["Title"] = title
    if priority and 1 <= priority <= 5:
        headers["Priority"] = str(priority)
    if tags:
        headers["Tags"] = ",".join(tags)
    
    # Send notification
    try:
        response = requests.post(url, data=message, headers=headers)
        
        # Print response details
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Notification sent successfully!")
            return {"status": "success", "message": "Notification sent"}
        else:
            print(f"❌ Failed to send notification: {response.text}")
            return {"status": "error", "message": response.text}
    
    except Exception as e:
        print(f"❌ Error sending notification: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Send a test notification to an ntfy server")
    parser.add_argument("--server", default="http://10.185.1.8:8888", help="NTFY server URL")
    parser.add_argument("--topic", default="Sara", help="Notification topic")
    parser.add_argument("--message", default="This is a test notification", help="Message content")
    parser.add_argument("--title", default="Test Notification", help="Notification title")
    parser.add_argument("--priority", type=int, choices=range(1, 6), default=3, help="Priority (1-5)")
    parser.add_argument("--tags", default="test,notification", help="Comma-separated tags")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process tags
    tags = args.tags.split(",") if args.tags else []
    
    print(f"Sending notification to {args.server}/{args.topic}...")
    print(f"Title: {args.title}")
    print(f"Message: {args.message}")
    print(f"Priority: {args.priority}")
    print(f"Tags: {tags}")
    print("-" * 40)
    
    # Send notification
    result = send_notification(
        server=args.server,
        topic=args.topic,
        message=args.message,
        title=args.title,
        priority=args.priority,
        tags=tags
    )
    
    # Exit with appropriate status code
    sys.exit(0 if result["status"] == "success" else 1)