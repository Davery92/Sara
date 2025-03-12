import json
import re
import os
import ollama
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ----------------- EXTRACTION FUNCTIONS -----------------

def extract_messages(log_content: str) -> List[Dict[str, str]]:
    """Extract user and assistant messages from chat logs."""
    messages = []
    lines = log_content.strip().split('\n')
    current_role = None
    current_content = []
    
    for line in lines:
        if line.startswith("User:"):
            # If we were building a message, save it before starting a new one
            if current_role:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })
            
            # Start a new user message
            current_role = "user"
            current_content = [line[5:].strip()]  # Remove "User:" prefix
        
        elif line.startswith("Assistant:"):
            # If we were building a message, save it before starting a new one
            if current_role:
                messages.append({
                    "role": current_role, 
                    "content": "\n".join(current_content).strip()
                })
            
            # Start a new assistant message
            current_role = "assistant"
            current_content = [line[10:].strip()]  # Remove "Assistant:" prefix
        
        else:
            # Continue the current message
            if current_role:
                current_content.append(line.strip())
    
    # Add the last message if there is one
    if current_role:
        messages.append({
            "role": current_role,
            "content": "\n".join(current_content).strip()
        })
    
    return messages

def analyze_conversation(messages: List[Dict[str, str]], use_ollama: bool = True) -> List[Dict]:
    """Analyze all conversations to identify contexts."""
    results = []
    
    # Process each user message
    for i, message in enumerate(messages):
        if message["role"] != "user":
            continue
            
        user_msg = message["content"]
        
        # Skip very short messages as they likely don't have much context
        if len(user_msg.strip()) < 5:
            continue
            
        # Get the assistant's response if available
        assistant_msg = ""
        if i+1 < len(messages) and messages[i+1]["role"] == "assistant":
            assistant_msg = messages[i+1]["content"]
        
        try:
            # Analyze the conversation
            if use_ollama:
                result = analyze_with_ollama(user_msg, assistant_msg)
            else:
                result = basic_context_analysis(user_msg)
                
            if result:
                results.append(result)
                print(f"Processed message {i+1}: {result['context_type']}")
        except Exception as e:
            print(f"Error processing message {i+1}: {e}")
    
    return results

def analyze_with_ollama(user_msg: str, assistant_msg: str, model: str = "qwen2.5:32b") -> Dict:
    """Use Ollama to determine the context type and generate structured data."""
    prompt = f"""
    Analyze this conversation and identify the main context type.
    
    USER MESSAGE:
    {user_msg}
    
    ASSISTANT RESPONSE:
    {assistant_msg}
    
    Identify a single word or short phrase as the context_type (e.g., diet, fitness, greeting, commute, worklife).
    Then write a brief summary of what the user is discussing.
    
    Create a JSON object with exactly these fields:
    - context_type: A single word or short phrase for the category
    - data.content: A brief summary of what the user is asking about or discussing
    - data.confidence_score: A number between 0.5-0.99 representing confidence
    - data.last_updated: "2025-03-06"
    
    Return ONLY valid JSON with NOTHING else.
    """
    
    try:
        # Call Ollama API
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract JSON from response
        json_text = response['message']['content'].strip()
        
        # Clean up any markdown formatting
        json_text = re.sub(r'^```json', '', json_text)
        json_text = re.sub(r'^```', '', json_text)
        json_text = re.sub(r'```$', '', json_text)
        json_text = json_text.strip()
        
        # Parse the JSON
        try:
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response: {e}")
            print(f"Response text: {json_text}")
            # Fall back to basic analysis
            return basic_context_analysis(user_msg)
            
    except Exception as e:
        print(f"Error with Ollama: {e}")
        return basic_context_analysis(user_msg)

def basic_context_analysis(user_msg: str) -> Dict:
    """Basic rule-based context analysis when Ollama is unavailable."""
    user_msg_lower = user_msg.lower()
    
    # Simple keyword matching
    keywords = {
        "diet": ["diet", "nutrition", "food", "eat", "meal", "carnivore", "protein", "carb", "fat"],
        "fitness": ["workout", "exercise", "training", "strength", "gym", "muscle", "weight"],
        "greeting": ["hello", "hi ", "hey", "morning", "afternoon", "evening"],
        "commute": ["drive", "commute", "traffic", "road", "car", "bus", "train"],
        "routine": ["routine", "schedule", "agenda", "plan", "time", "clock", "day"],
        "work": ["work", "job", "office", "career", "meeting", "email", "boss", "colleague"],
        "pets": ["dog", "cat", "pet", "animal", "walk"],
    }
    
    # Find matching context
    max_matches = 0
    best_context = "general"
    
    for context, terms in keywords.items():
        matches = sum(1 for term in terms if term in user_msg_lower)
        if matches > max_matches:
            max_matches = matches
            best_context = context
    
    # Generate a simple summary
    summary = user_msg[:60] + "..."
    
    # Calculate a confidence score (higher for more keyword matches)
    confidence = min(0.5 + (max_matches * 0.1), 0.95)
    
    return {
        "context_type": best_context,
        "data": {
            "content": f"User message about {best_context}: {summary}",
            "confidence_score": confidence,
            "last_updated": "2025-03-06"
        }
    }

# ----------------- CONSOLIDATION FUNCTIONS -----------------

def consolidate_contexts(contexts: List[Dict]) -> Dict[str, Dict]:
    """Group contexts by type and consolidate similar ones."""
    # First, clean up context types
    for context in contexts:
        context_type = context.get('context_type', 'unknown').lower()
        # Normalize context types
        if context_type in ['food', 'eating', 'meals', 'cooking']:
            context['context_type'] = 'nutrition'
        elif context_type in ['workout', 'exercise', 'gym']:
            context['context_type'] = 'fitness'
        elif context_type in ['hi', 'hello', 'hey', 'welcome']:
            context['context_type'] = 'greeting'
        elif context_type in ['driving', 'traffic', 'transportation']:
            context['context_type'] = 'commute'
    
    # Group by context type
    grouped = defaultdict(list)
    for context in contexts:
        context_type = context.get('context_type', 'unknown')
        grouped[context_type].append(context)
    
    # Merge contexts within each group
    consolidated = {}
    for context_type, items in grouped.items():
        merged = merge_contexts(items)
        consolidated[context_type] = merged
    
    return consolidated

def merge_contexts(contexts: List[Dict]) -> Dict:
    """Merge multiple contexts of the same type."""
    if not contexts:
        return {}
    
    # Start with the first context as a base
    merged = contexts[0].copy()
    
    # For single contexts, just return as is
    if len(contexts) == 1:
        return merged
    
    # Initialize the data section
    if 'data' not in merged:
        merged['data'] = {}
    
    # Collect all content entries
    contents = []
    if 'content' in merged['data']:
        contents.append(merged['data']['content'])
    
    # Find maximum confidence
    max_confidence = merged['data'].get('confidence_score', 0)
    
    # Process all other contexts
    for context in contexts[1:]:
        if 'data' in context:
            # Add unique content
            if 'content' in context['data']:
                content = context['data']['content']
                if content not in contents:
                    contents.append(content)
            
            # Update max confidence
            if 'confidence_score' in context['data']:
                max_confidence = max(max_confidence, context['data']['confidence_score'])
    
    # Update merged context
    merged['data']['entries'] = contents
    merged['data']['confidence_score'] = max_confidence
    
    # Generate a combined content field
    if len(contents) > 1:
        merged['data']['content'] = f"Multiple {merged['context_type']} topics: {contents[0]} (+ {len(contents)-1} more)"
    elif contents:
        merged['data']['content'] = contents[0]
    
    return merged

# ----------------- MAIN WORKFLOW -----------------

def process_chat_logs(log_file: str, use_ollama: bool = True) -> None:
    """Complete workflow to process chat logs and create context files."""
    print(f"Reading chat logs from {log_file}...")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"File not found: {log_file}")
        return
    
    # Extract messages
    messages = extract_messages(log_content)
    print(f"Extracted {len(messages)} messages.")
    
    # Analyze conversations
    print(f"Analyzing conversations{'with Ollama' if use_ollama else 'without Ollama'}...")
    contexts = analyze_conversation(messages, use_ollama)
    print(f"Identified {len(contexts)} context items.")
    
    # Create raw output directory
    os.makedirs("json_raw", exist_ok=True)
    
    # Save raw context files for reference
    for i, context in enumerate(contexts):
        context_type = context.get('context_type', 'unknown')
        with open(f"json_raw/{context_type}_{i+1}.json", 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2)
    
    # Consolidate contexts
    print("Consolidating contexts...")
    consolidated = consolidate_contexts(contexts)
    print(f"Consolidated into {len(consolidated)} context types.")
    
    # Create output directory
    os.makedirs("json_contexts", exist_ok=True)
    
    # Save consolidated contexts
    for context_type, context in consolidated.items():
        # Clean filename
        clean_type = re.sub(r'[^\w\-]', '_', context_type.lower())
        filename = f"{clean_type}.json"
        
        with open(f"json_contexts/{filename}", 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2)
        
        print(f"Saved: {filename}")
    
    print("Processing complete!")
    print("\nContext Summary:")
    for context_type, context in consolidated.items():
        entry_count = len(context.get('data', {}).get('entries', []))
        confidence = context.get('data', {}).get('confidence_score', 0)
        print(f"  - {context_type}: {entry_count} entries, {confidence:.2f} confidence")

def main():
    # You can change these parameters
    log_file = "/home/david/Sara/logs/03-04-2025.txt"  # Path to your chat log file
    use_ollama = True       # Set to False to use basic analysis instead of Ollama
    
    # Run the workflow
    process_chat_logs(log_file, use_ollama)

if __name__ == "__main__":
    main()