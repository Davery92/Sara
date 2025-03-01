import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
import time
import threading
from fastapi import APIRouter, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import uuid
import os
import pytz
from dateutil import parser as date_parser

# Configure logging
logger = logging.getLogger("timer-reminder-module")

# NTFY server configuration
NTFY_SERVER = "http://10.185.1.8:8888"

# Directory for storing persistence data
DATA_DIR = "/home/david/Sara/data"
TIMERS_FILE = os.path.join(DATA_DIR, "timers.json")
REMINDERS_FILE = os.path.join(DATA_DIR, "reminders.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

class TimerRequest(BaseModel):
    """Request model for creating a timer"""
    duration: int  # Duration in seconds
    message: str
    title: Optional[str] = "Timer Notification"
    priority: Optional[int] = 3  # Default medium priority
    tags: Optional[List[str]] = ["clock", "timer"]

class ReminderRequest(BaseModel):
    """Request model for creating a reminder"""
    datetime: str  # ISO format or human-readable datetime string
    message: str
    title: Optional[str] = "Reminder Notification"
    priority: Optional[int] = 3
    tags: Optional[List[str]] = ["calendar", "reminder"]
    recurrence: Optional[str] = None  # daily, weekly, monthly, etc.

class TimerReminderResponse(BaseModel):
    """Response model for timer/reminder creation"""
    id: str
    type: str  # "timer" or "reminder"
    message: str
    scheduled_time: str
    status: str

# Global storage for active timers and reminders
active_timers = {}
active_reminders = {}

# Create router for the timer/reminder endpoints
timer_router = APIRouter(prefix="/timers", tags=["timers"])
reminder_router = APIRouter(prefix="/reminders", tags=["reminders"])

def save_timers():
    """Save all active timers to a file"""
    try:
        with open(TIMERS_FILE, 'w') as f:
            json.dump(active_timers, f, indent=2)
        logger.info(f"Saved {len(active_timers)} timers to {TIMERS_FILE}")
    except Exception as e:
        logger.error(f"Error saving timers: {str(e)}")

def save_reminders():
    """Save all active reminders to a file"""
    try:
        with open(REMINDERS_FILE, 'w') as f:
            json.dump(active_reminders, f, indent=2)
        logger.info(f"Saved {len(active_reminders)} reminders to {REMINDERS_FILE}")
    except Exception as e:
        logger.error(f"Error saving reminders: {str(e)}")

def load_timers():
    """Load timers from file"""
    global active_timers
    try:
        if os.path.exists(TIMERS_FILE):
            with open(TIMERS_FILE, 'r') as f:
                loaded_timers = json.load(f)
                # Only load timers that haven't expired or been completed
                for timer_id, timer in loaded_timers.items():
                    if timer["status"] in ["pending", "running"]:
                        active_timers[timer_id] = timer
            logger.info(f"Loaded {len(active_timers)} active timers from {TIMERS_FILE}")
    except Exception as e:
        logger.error(f"Error loading timers: {str(e)}")

def load_reminders():
    """Load reminders from file"""
    global active_reminders
    try:
        if os.path.exists(REMINDERS_FILE):
            with open(REMINDERS_FILE, 'r') as f:
                loaded_reminders = json.load(f)
                # Only load reminders that haven't been triggered or cancelled
                for reminder_id, reminder in loaded_reminders.items():
                    if reminder["status"] in ["pending", "scheduled"]:
                        active_reminders[reminder_id] = reminder
            logger.info(f"Loaded {len(active_reminders)} active reminders from {REMINDERS_FILE}")
    except Exception as e:
        logger.error(f"Error loading reminders: {str(e)}")

async def send_ntfy_notification(topic: str = "notify", title: str = None, message: str = None, 
                                priority: int = 3, tags: List[str] = None):
    """Send a notification to the NTFY server"""
    if not message:
        return {"error": "Message is required"}
    
    # Prepare headers
    headers = {}
    if title:
        headers["Title"] = title
    if priority and 1 <= priority <= 5:
        headers["Priority"] = str(priority)
    if tags:
        headers["Tags"] = ",".join(tags)
    
    # Send the notification
    try:
        url = f"{NTFY_SERVER}/{topic}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=message, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to send notification: {error_text}")
                    return {"error": f"Notification failed with status {response.status}: {error_text}"}
                else:
                    result = await response.text()
                    logger.info(f"Notification sent successfully: {result}")
                    return {"status": "success", "message": "Notification sent"}
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        return {"error": f"Failed to send notification: {str(e)}"}

async def handle_timer(timer_id: str, duration: int, message: str, title: str, 
                      priority: int, tags: List[str], topic: str = "Sara"):
    """Handle a timer by waiting for the specified duration and then sending a notification"""
    try:
        logger.info(f"Timer {timer_id} started for {duration} seconds")
        # Update timer status
        active_timers[timer_id]["status"] = "running"
        save_timers()  # Save status change
        
        # Wait for the specified duration
        await asyncio.sleep(duration)
        
        # Send the notification
        notification_result = await send_ntfy_notification(
            topic=topic,
            title=title,
            message=message,
            priority=priority,
            tags=tags
        )
        
        # Update timer status
        if "error" in notification_result:
            active_timers[timer_id]["status"] = "failed"
            active_timers[timer_id]["error"] = notification_result["error"]
        else:
            active_timers[timer_id]["status"] = "completed"
            active_timers[timer_id]["completed_at"] = datetime.now().isoformat()
        
        save_timers()  # Save status change
        
        logger.info(f"Timer {timer_id} completed")
        return notification_result
    except Exception as e:
        logger.error(f"Error in timer {timer_id}: {str(e)}")
        active_timers[timer_id]["status"] = "failed"
        active_timers[timer_id]["error"] = str(e)
        save_timers()  # Save status change
        return {"error": str(e)}

async def handle_reminder(reminder_id: str, target_time: datetime, message: str, title: str, 
                         priority: int, tags: List[str], topic: str = "Sara"):
    """Handle a reminder by waiting until the specified time and then sending a notification"""
    try:
        now = datetime.now()
        
        # Calculate seconds to wait
        if target_time > now:
            seconds_to_wait = (target_time - now).total_seconds()
            logger.info(f"Reminder {reminder_id} scheduled for {target_time.isoformat()} (in {seconds_to_wait:.1f} seconds)")
            
            # Update reminder status
            active_reminders[reminder_id]["status"] = "scheduled"
            save_reminders()  # Save status change
            
            # Wait until the target time
            await asyncio.sleep(seconds_to_wait)
            
            # Send the notification
            notification_result = await send_ntfy_notification(
                topic=topic,
                title=title,
                message=message,
                priority=priority,
                tags=tags
            )
            
            # Update reminder status
            if "error" in notification_result:
                active_reminders[reminder_id]["status"] = "failed"
                active_reminders[reminder_id]["error"] = notification_result["error"]
            else:
                active_reminders[reminder_id]["status"] = "completed"
                active_reminders[reminder_id]["completed_at"] = datetime.now().isoformat()
            
            save_reminders()  # Save status change
            
            logger.info(f"Reminder {reminder_id} completed")
            return notification_result
        else:
            logger.warning(f"Reminder {reminder_id} scheduled for past time {target_time.isoformat()}")
            active_reminders[reminder_id]["status"] = "failed"
            active_reminders[reminder_id]["error"] = "Reminder time is in the past"
            save_reminders()  # Save status change
            return {"error": "Reminder time is in the past"}
    except Exception as e:
        logger.error(f"Error in reminder {reminder_id}: {str(e)}")
        active_reminders[reminder_id]["status"] = "failed"
        active_reminders[reminder_id]["error"] = str(e)
        save_reminders()  # Save status change
        return {"error": str(e)}

def parse_datetime(datetime_str: str) -> datetime:
    """Parse a datetime string into a datetime object
    Supports ISO format and various human-readable formats
    """
    try:
        # For exact ISO format
        return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
    except ValueError:
        try:
            # Try dateutil parser for more flexible parsing
            return date_parser.parse(datetime_str)
        except Exception as e:
            raise ValueError(f"Unable to parse datetime string: {datetime_str}. Error: {str(e)}")

async def restart_timers():
    """Restart all active timers after server restart"""
    for timer_id, timer in list(active_timers.items()):
        if timer["status"] in ["pending", "running"]:
            try:
                # Calculate remaining time
                created_time = datetime.fromisoformat(timer["created_at"])
                scheduled_time = datetime.fromisoformat(timer["scheduled_time"])
                now = datetime.now()
                
                # If scheduled time is in the future, start a new timer
                if scheduled_time > now:
                    remaining_seconds = (scheduled_time - now).total_seconds()
                    logger.info(f"Restarting timer {timer_id} with {remaining_seconds:.1f} seconds remaining")
                    
                    # Start a new background task for this timer
                    asyncio.create_task(
                        handle_timer(
                            timer_id,
                            int(remaining_seconds),
                            timer["message"],
                            timer["title"],
                            timer.get("priority", 3),
                            timer.get("tags", ["timer"])
                        )
                    )
                else:
                    # Timer has already expired, mark as missed
                    logger.info(f"Timer {timer_id} has expired during server downtime")
                    timer["status"] = "missed"
                    timer["error"] = "Timer expired during server downtime"
                    
                    # Optionally, send a notification that the timer was missed
                    asyncio.create_task(
                        send_ntfy_notification(
                            topic="Sara",
                            title=f"Missed Timer: {timer['title']}",
                            message=f"{timer['message']} (This timer was missed during server downtime)",
                            priority=timer.get("priority", 3),
                            tags=timer.get("tags", ["timer", "missed"])
                        )
                    )
            except Exception as e:
                logger.error(f"Error restarting timer {timer_id}: {str(e)}")
                timer["status"] = "error"
                timer["error"] = str(e)
    
    # Save updated timer statuses
    save_timers()

async def restart_reminders():
    """Restart all active reminders after server restart"""
    for reminder_id, reminder in list(active_reminders.items()):
        if reminder["status"] in ["pending", "scheduled"]:
            try:
                # Parse target time
                target_time = datetime.fromisoformat(reminder["scheduled_time"])
                now = datetime.now()
                
                # If scheduled time is in the future, start a new reminder
                if target_time > now:
                    logger.info(f"Restarting reminder {reminder_id} scheduled for {target_time.isoformat()}")
                    
                    # Start a new background task for this reminder
                    asyncio.create_task(
                        handle_reminder(
                            reminder_id,
                            target_time,
                            reminder["message"],
                            reminder["title"],
                            reminder.get("priority", 3),
                            reminder.get("tags", ["reminder"])
                        )
                    )
                else:
                    # Reminder has already passed, mark as missed
                    logger.info(f"Reminder {reminder_id} has passed during server downtime")
                    reminder["status"] = "missed"
                    reminder["error"] = "Reminder time passed during server downtime"
                    
                    # Optionally, send a notification that the reminder was missed
                    asyncio.create_task(
                        send_ntfy_notification(
                            topic="Sara",
                            title=f"Missed Reminder: {reminder['title']}",
                            message=f"{reminder['message']} (This reminder was missed during server downtime)",
                            priority=reminder.get("priority", 3),
                            tags=reminder.get("tags", ["reminder", "missed"])
                        )
                    )
                    
                    # If this is a recurring reminder, schedule the next occurrence
                    if reminder.get("recurrence"):
                        # TODO: Implement recurring reminder logic
                        pass
            except Exception as e:
                logger.error(f"Error restarting reminder {reminder_id}: {str(e)}")
                reminder["status"] = "error"
                reminder["error"] = str(e)
    
    # Save updated reminder statuses
    save_reminders()

# Timer endpoints
@timer_router.post("/", response_model=TimerReminderResponse)
async def create_timer(timer_req: TimerRequest, background_tasks: BackgroundTasks):
    """Create a new timer"""
    timer_id = str(uuid.uuid4())
    scheduled_time = datetime.now() + timedelta(seconds=timer_req.duration)
    
    # Store timer metadata
    active_timers[timer_id] = {
        "id": timer_id,
        "type": "timer",
        "duration": timer_req.duration,
        "message": timer_req.message,
        "title": timer_req.title,
        "priority": timer_req.priority,
        "tags": timer_req.tags,
        "created_at": datetime.now().isoformat(),
        "scheduled_time": scheduled_time.isoformat(),
        "status": "pending"
    }
    
    # Save new timer to file
    save_timers()
    
    # Start the timer in the background
    background_tasks.add_task(
        handle_timer,
        timer_id,
        timer_req.duration,
        timer_req.message,
        timer_req.title,
        timer_req.priority,
        timer_req.tags
    )
    
    return {
        "id": timer_id,
        "type": "timer",
        "message": timer_req.message,
        "scheduled_time": scheduled_time.isoformat(),
        "status": "pending"
    }

@timer_router.get("/", response_model=List[Dict[str, Any]])
async def list_timers():
    """List all active timers"""
    return list(active_timers.values())

@timer_router.get("/{timer_id}", response_model=Dict[str, Any])
async def get_timer(timer_id: str):
    """Get details of a specific timer"""
    if timer_id not in active_timers:
        raise HTTPException(status_code=404, detail="Timer not found")
    return active_timers[timer_id]

@timer_router.delete("/{timer_id}")
async def cancel_timer(timer_id: str):
    """Cancel a timer (if possible)"""
    if timer_id not in active_timers:
        raise HTTPException(status_code=404, detail="Timer not found")
    
    # Mark as cancelled in the dictionary, but note that the background task
    # may still be running - we can't easily cancel it once it's started
    timer = active_timers[timer_id]
    if timer["status"] in ["pending", "running"]:
        timer["status"] = "cancelled"
        # Save the updated status
        save_timers()
        return {"status": "success", "message": "Timer cancelled"}
    else:
        return {"status": "error", "message": f"Cannot cancel timer with status {timer['status']}"}

# Reminder endpoints
@reminder_router.post("/", response_model=TimerReminderResponse)
async def create_reminder(reminder_req: ReminderRequest, background_tasks: BackgroundTasks):
    """Create a new reminder"""
    reminder_id = str(uuid.uuid4())
    
    # Parse the target datetime
    try:
        target_time = parse_datetime(reminder_req.datetime)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Store reminder metadata
    active_reminders[reminder_id] = {
        "id": reminder_id,
        "type": "reminder",
        "datetime": reminder_req.datetime,
        "message": reminder_req.message,
        "title": reminder_req.title,
        "priority": reminder_req.priority,
        "tags": reminder_req.tags,
        "recurrence": reminder_req.recurrence,
        "created_at": datetime.now().isoformat(),
        "scheduled_time": target_time.isoformat(),
        "status": "pending"
    }
    
    # Save new reminder to file
    save_reminders()
    
    # Start the reminder in the background
    background_tasks.add_task(
        handle_reminder,
        reminder_id,
        target_time,
        reminder_req.message,
        reminder_req.title,
        reminder_req.priority,
        reminder_req.tags
    )
    
    return {
        "id": reminder_id,
        "type": "reminder",
        "message": reminder_req.message,
        "scheduled_time": target_time.isoformat(),
        "status": "pending"
    }

@reminder_router.get("/", response_model=List[Dict[str, Any]])
async def list_reminders():
    """List all active reminders"""
    return list(active_reminders.values())

@reminder_router.get("/{reminder_id}", response_model=Dict[str, Any])
async def get_reminder(reminder_id: str):
    """Get details of a specific reminder"""
    if reminder_id not in active_reminders:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return active_reminders[reminder_id]

@reminder_router.delete("/{reminder_id}")
async def cancel_reminder(reminder_id: str):
    """Cancel a reminder (if possible)"""
    if reminder_id not in active_reminders:
        raise HTTPException(status_code=404, detail="Reminder not found")
    
    # Mark as cancelled in the dictionary
    reminder = active_reminders[reminder_id]
    if reminder["status"] in ["pending", "scheduled"]:
        reminder["status"] = "cancelled"
        # Save the updated status
        save_reminders()
        return {"status": "success", "message": "Reminder cancelled"}
    else:
        return {"status": "error", "message": f"Cannot cancel reminder with status {reminder['status']}"}

# Basic test function to send a notification directly
@timer_router.post("/test-notification")
async def test_notification(
    message: str = Body(..., embed=True),
    title: str = Body("Test Notification", embed=True),
    priority: int = Body(3, embed=True),
    tags: List[str] = Body(["test"], embed=True)
):
    """Test the notification functionality directly"""
    result = await send_ntfy_notification(
        topic="Sara", 
        title=title,
        message=message,
        priority=priority,
        tags=tags
    )
    return result

def integrate_timer_reminder_with_server(app):
    """Integrate the timer and reminder routers with the main FastAPI app"""
    app.include_router(timer_router)
    app.include_router(reminder_router)
    
    # Run the startup function
    @app.on_event("startup")
    async def timer_reminder_startup():
        # Load saved data
        load_timers()
        load_reminders()
        
        # Restart active timers and reminders
        await asyncio.gather(restart_timers(), restart_reminders())
        
        logger.info("Timer and reminder module started with persistence")
    
    logger.info("Timer and reminder module integrated with server")