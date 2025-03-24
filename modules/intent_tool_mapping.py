import logging

# Configure logger
logger = logging.getLogger("intent-tool-mapping")

# Define mapping from intents to tools
INTENT_TO_TOOL_MAP = {
    "search": ["search_perplexica"],
    "search_perplexica": ["search_perplexica"],
    "remember": ["append_core_memory"],
    "note_create": ["create_note"],
    "note_read": ["read_note"],
    "note_update": ["append_note"],
    "note_delete": ["delete_note"],
    "note_list": ["list_notes"],
    "thinking": ["send_message"],
    "timer_set": ["set_timer"],
    "reminder_set": ["set_reminder"],
    "none": []  # No tools for general conversation
}

# Define confidence threshold for each intent
INTENT_CONFIDENCE_THRESHOLDS = {
    "search": 0.3,
    "remember": 0.5,
    "note_create": 0.3,
    "note_read": 0.3,
    "note_update": 0.3,
    "note_delete": 0.6,  # Higher threshold for destructive actions
    "note_list": 0.3,
    "thinking": 0.3,
    "timer_set": 0.3,
    "reminder_set": 0.3,
    "none": 0.3,  # Default threshold
}

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
