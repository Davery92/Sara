import logging

# Configure logger
logger = logging.getLogger("intent-tool-mapping")

# Define mapping from intents to tools
INTENT_TO_TOOL_MAP = {
    "search": ["search_perplexica"],
    "search_perplexica": ["search_perplexica"],
    "remember": ["append_core_memory"],
    "append_core_memory": ["append_core_memory"],  # Add this line
    "note_create": ["create_note"],
    "read_note": ["read_note"],
    "append_note": ["append_note"],
    "delete_note": ["delete_note"],
    "list_notes": ["list_notes"],
    "thinking": ["send_message"],
    "set_timer": ["set_timer"],
    "set_reminder": ["set_reminder"],
    "create_note": ["create_note"],
    "none": []  # No tools for general conversation
}

# Define confidence threshold for each intent
INTENT_CONFIDENCE_THRESHOLDS = {
    "search": 0.3,
    "remember": 0.5,
    "note_create": 0.5,
    "note_read": 0.5,
    "note_update": 0.5,
    "note_delete": 0.6,  # Higher threshold for destructive actions
    "note_list": 0.5,
    "thinking": 0.5,
    "timer_set": 0.5,
    "reminder_set": 0.5,
    "none": 0.5,  # Default threshold
}

# Add this function to intent_tool_mapping.py:
def should_skip_tools_for_intent(prediction_result):
    """
    Determine if tools should be skipped based on the intent prediction.
    
    Args:
        prediction_result (dict): The prediction result from intent classifier
        
    Returns:
        bool: True if tools should be skipped, False otherwise
    """
    if not prediction_result or "predictions" not in prediction_result or not prediction_result["predictions"]:
        return True
        
    # Get the top prediction
    top_prediction = prediction_result["predictions"][0]
    top_intent = top_prediction["intent"]
    confidence = top_prediction["probability"]
    
    # Always skip tools for certain intent types, regardless of confidence
    if top_intent in ["none", "thinking", "send_message", "no_tool_required"]:
        logger.info(f"Skipping tools for intent type '{top_intent}'")
        return True
        
    # Check against the confidence threshold
    threshold = INTENT_CONFIDENCE_THRESHOLDS.get(top_intent, INTENT_CONFIDENCE_THRESHOLDS["none"])
    if confidence < threshold:
        logger.info(f"Skipping tools for intent '{top_intent}' with confidence {confidence:.4f} (below threshold {threshold})")
        return True
        
    return False

def get_tools_for_intent(prediction_result, all_tools):
    """
    Get the appropriate tools based on the predicted intent.
    
    Args:
        prediction_result (dict): The prediction result from the intent classifier
        all_tools (list): All available tools
        
    Returns:
        list: List of tools to use for this intent, or None if no specific tools
    """
    if not prediction_result or "predictions" not in prediction_result or not prediction_result["predictions"]:
        logger.warning("No intent predictions available")
        return None
    
    # Get the top prediction
    top_prediction = prediction_result["predictions"][0]
    top_intent = top_prediction["intent"]
    confidence = top_prediction["probability"]
    
    # Check if we meet the confidence threshold
    threshold = INTENT_CONFIDENCE_THRESHOLDS.get(top_intent, INTENT_CONFIDENCE_THRESHOLDS["none"])
    
    if confidence < threshold:
        logger.info(f"Intent '{top_intent}' predicted with confidence {confidence:.4f}, below threshold {threshold}")
        return None
    
    # Get the tools for this intent
    tool_names = INTENT_TO_TOOL_MAP.get(top_intent, [])
    
    if not tool_names:
        logger.info(f"No specific tools for intent '{top_intent}'")
        return None
    
    # Find the matching tools from the available tools
    selected_tools = []
    for tool in all_tools:
        if isinstance(tool, dict) and 'function' in tool and 'name' in tool['function']:
            if tool['function']['name'] in tool_names:
                selected_tools.append(tool)
    
    logger.info(f"Selected {len(selected_tools)} tools for intent '{top_intent}' with confidence {confidence:.4f}")
    
    return selected_tools if selected_tools else None
