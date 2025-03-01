import logging
from fastapi import FastAPI
import asyncio

# Import timer/reminder module components
from modules.timer_reminder_module import integrate_timer_reminder_with_server
from modules.timer_reminder_tools import TIMER_REMINDER_TOOL_DEFINITIONS

logger = logging.getLogger("timer-reminder-integration")

# Import the tool functions - these are now async functions
from modules.timer_reminder_tool_functions import (
    set_timer, set_reminder, list_timers, list_reminders, cancel_timer, cancel_reminder
)

def integrate_timer_reminder_tools(app: FastAPI, available_tools: dict, tool_definitions: list):
    """
    Integrate the timer and reminder tools into the server
    
    Parameters:
    - app: The FastAPI app
    - available_tools: The dictionary of available tools
    - tool_definitions: The list of tool definitions
    """
    logger.info("Integrating timer and reminder tools")
    
    # Add timer and reminder functions to available tools
    available_tools.update({
        'set_timer': set_timer,
        'set_reminder': set_reminder,
        'list_timers': list_timers,
        'list_reminders': list_reminders,
        'cancel_timer': cancel_timer,
        'cancel_reminder': cancel_reminder
    })
    
    # Add timer and reminder tool definitions to the list of tool definitions
    tool_definitions.extend(TIMER_REMINDER_TOOL_DEFINITIONS)
    
    # Integrate the timer and reminder API endpoints with the server
    integrate_timer_reminder_with_server(app)
    
    logger.info("Timer and reminder tools integrated successfully")
    
    return {
        'available_tools': available_tools,
        'tool_definitions': tool_definitions
    }