import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional
import asyncio

logger = logging.getLogger("timer-reminder-tools")

# This file contains the actual functions that will be called by the AI
# through the tool system. These functions interact with the timer and reminder
# APIs to create and manage timers and reminders.

# API Base URL - adjust to your server's address
API_BASE_URL = "http://localhost:7009"  # Update with your server address

async def set_timer(duration: int, message: str, title: str = "Timer Notification", 
                   priority: int = 3):
    """Create a timer that will send a notification after the specified duration."""
    # Use default tags since they're not in the tool definition anymore
    tags = ["clock", "timer"]
    
    try:
        url = f"{API_BASE_URL}/timers/"
        payload = {
            "duration": duration,
            "message": message,
            "title": title,
            "priority": priority,
            "tags": tags
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to create timer: {error_text}")
                    return f"Failed to create timer: {error_text}"
                
                result = await response.json()
                logger.info(f"Timer created successfully: {result}")
                
                # Format a nice response for the user
                timer_time = result["scheduled_time"]
                return f"Timer set successfully! You'll be notified with '{message}' at {timer_time}."
    except Exception as e:
        logger.error(f"Error creating timer: {str(e)}")
        return f"Error creating timer: {str(e)}"

async def set_reminder(datetime: str, message: str, title: str = "Reminder Notification", 
                      priority: int = 3):
    """Create a reminder that will send a notification at the specified date and time."""
    # Use default values
    tags = ["calendar", "reminder"]
    recurrence = None
    
    try:
        url = f"{API_BASE_URL}/reminders/"
        payload = {
            "datetime": datetime,
            "message": message,
            "title": title,
            "priority": priority,
            "tags": tags
        }
        
        if recurrence:
            payload["recurrence"] = recurrence
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to create reminder: {error_text}")
                    return f"Failed to create reminder: {error_text}"
                
                result = await response.json()
                logger.info(f"Reminder created successfully: {result}")
                
                # Format a nice response for the user
                reminder_time = result["scheduled_time"]
                return f"Reminder set successfully! You'll be reminded about '{message}' at {reminder_time}."
    except Exception as e:
        logger.error(f"Error creating reminder: {str(e)}")
        return f"Error creating reminder: {str(e)}"

async def list_timers() -> str:
    """List all active timers."""
    try:
        url = f"{API_BASE_URL}/timers/"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to list timers: {error_text}")
                    return f"Failed to list timers: {error_text}"
                
                timers = await response.json()
                
                if not timers:
                    return "You don't have any active timers."
                
                # Format a nice response with the timers
                response_text = "Here are your active timers:\n\n"
                for timer in timers:
                    status = timer.get("status", "unknown")
                    if status in ["pending", "running"]:
                        response_text += f"• ID: {timer['id']}\n"
                        response_text += f"  Message: {timer['message']}\n"
                        response_text += f"  Scheduled for: {timer['scheduled_time']}\n"
                        response_text += f"  Status: {status}\n\n"
                
                # If there are no active timers after filtering
                if response_text == "Here are your active timers:\n\n":
                    return "You don't have any active timers."
                
                return response_text
    except Exception as e:
        logger.error(f"Error listing timers: {str(e)}")
        return f"Error listing timers: {str(e)}"

async def list_reminders() -> str:
    """List all active reminders."""
    try:
        url = f"{API_BASE_URL}/reminders/"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to list reminders: {error_text}")
                    return f"Failed to list reminders: {error_text}"
                
                reminders = await response.json()
                
                if not reminders:
                    return "You don't have any active reminders."
                
                # Format a nice response with the reminders
                response_text = "Here are your active reminders:\n\n"
                for reminder in reminders:
                    status = reminder.get("status", "unknown")
                    if status in ["pending", "scheduled"]:
                        response_text += f"• ID: {reminder['id']}\n"
                        response_text += f"  Message: {reminder['message']}\n"
                        response_text += f"  Scheduled for: {reminder['scheduled_time']}\n"
                        response_text += f"  Status: {status}\n\n"
                
                # If there are no active reminders after filtering
                if response_text == "Here are your active reminders:\n\n":
                    return "You don't have any active reminders."
                
                return response_text
    except Exception as e:
        logger.error(f"Error listing reminders: {str(e)}")
        return f"Error listing reminders: {str(e)}"

async def cancel_timer(timer_id: str) -> str:
    """Cancel an active timer."""
    try:
        url = f"{API_BASE_URL}/timers/{timer_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to cancel timer: {error_text}")
                    return f"Failed to cancel timer: {error_text}"
                
                result = await response.json()
                logger.info(f"Timer cancelled: {result}")
                
                return f"Timer {timer_id} has been cancelled successfully."
    except Exception as e:
        logger.error(f"Error cancelling timer: {str(e)}")
        return f"Error cancelling timer: {str(e)}"

async def cancel_reminder(reminder_id: str) -> str:
    """Cancel an active reminder."""
    try:
        url = f"{API_BASE_URL}/reminders/{reminder_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to cancel reminder: {error_text}")
                    return f"Failed to cancel reminder: {error_text}"
                
                result = await response.json()
                logger.info(f"Reminder cancelled: {result}")
                
                return f"Reminder {reminder_id} has been cancelled successfully."
    except Exception as e:
        logger.error(f"Error cancelling reminder: {str(e)}")
        return f"Error cancelling reminder: {str(e)}"