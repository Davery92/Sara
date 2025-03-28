import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("briefing-notification")

class NotificationService:
    """Service for sending notifications when briefings are complete"""
    
    def __init__(self, ntfy_server="http://10.185.1.8:8888", ntfy_topic="Sara"):
        """
        Initialize the notification service
        
        Args:
            ntfy_server (str): URL of the ntfy server
            ntfy_topic (str): The notification topic
        """
        self.ntfy_server = ntfy_server
        self.ntfy_topic = ntfy_topic
        logger.info(f"Notification service initialized with server {ntfy_server} and topic {ntfy_topic}")
    
    def send_ntfy_notification(self, 
                              message: str, 
                              title: Optional[str] = None, 
                              priority: Optional[int] = 3,
                              tags: Optional[list] = None) -> Dict[str, Any]:
        """
        Send a notification via ntfy server
        
        Args:
            message (str): The notification message
            title (str, optional): The notification title
            priority (int, optional): Priority level (1-5, with 5 being highest)
            tags (list, optional): List of tags for the notification
            
        Returns:
            dict: Result with status and message
        """
        # Prepare URL
        url = f"{self.ntfy_server}/{self.ntfy_topic}"
        
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
            logger.info(f"Sending ntfy notification: {title}")
            response = requests.post(url, data=message, headers=headers)
            
            if response.status_code == 200:
                logger.info("Notification sent successfully")
                return {"status": "success", "message": "Notification sent"}
            else:
                logger.error(f"Failed to send notification: {response.text}")
                return {"status": "error", "message": response.text}
        
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return {"status": "error", "message": str(e)}

    def send_briefing_completion_notification(self, query: str, filename: str) -> Dict[str, Any]:
        """
        Send a notification that a briefing has been completed
        
        Args:
            query (str): The search query that was used
            filename (str): The filename of the briefing
            
        Returns:
            dict: Result with status and message
        """
        title = "Briefing Complete"
        message = f"Your briefing on '{query}' is now available. Check the dashboard or ask to show the briefing."
        tags = ["briefing", "complete"]
        
        return self.send_ntfy_notification(
            message=message,
            title=title,
            priority=3,
            tags=tags
        )

# Create a singleton instance
notification_service = NotificationService()