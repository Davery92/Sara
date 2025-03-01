# Replace the contents of timer_reminder_tools.py with this simpler version:

TIMER_REMINDER_TOOL_DEFINITIONS = [
    {
        'type': 'function',
        'function': {
            'name': 'set_timer',
            'description': 'Create a timer that will send a notification after a specified duration.',
            'parameters': {
                'type': 'object',
                'required': ['duration', 'message'],
                'properties': {
                    'duration': {
                        'type': 'integer',
                        'description': 'Duration in seconds for the timer'
                    },
                    'message': {
                        'type': 'string',
                        'description': 'Message to show in the notification'
                    },
                    'title': {
                        'type': 'string',
                        'description': 'Title for the notification (optional)'
                    }
                }
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'set_reminder',
            'description': 'Create a reminder that will send a notification at a specified date and time.',
            'parameters': {
                'type': 'object',
                'required': ['datetime', 'message'],
                'properties': {
                    'datetime': {
                        'type': 'string',
                        'description': 'Target date and time for the reminder (ISO format or human-readable)'
                    },
                    'message': {
                        'type': 'string',
                        'description': 'Message to show in the notification'
                    },
                    'title': {
                        'type': 'string',
                        'description': 'Title for the notification (optional)'
                    }
                }
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'list_timers',
            'description': 'List all active timers.',
            'parameters': {
                'type': 'object',
                'properties': {}
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'list_reminders',
            'description': 'List all active reminders.',
            'parameters': {
                'type': 'object',
                'properties': {}
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'cancel_timer',
            'description': 'Cancel an active timer.',
            'parameters': {
                'type': 'object',
                'required': ['timer_id'],
                'properties': {
                    'timer_id': {
                        'type': 'string',
                        'description': 'ID of the timer to cancel'
                    }
                }
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'cancel_reminder',
            'description': 'Cancel an active reminder.',
            'parameters': {
                'type': 'object',
                'required': ['reminder_id'],
                'properties': {
                    'reminder_id': {
                        'type': 'string',
                        'description': 'ID of the reminder to cancel'
                    }
                }
            }
        }
    }
]