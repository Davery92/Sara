"""
Note management tools for the OpenAI-compatible server.
This module provides functions for creating, reading, updating, and deleting notes.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("openai-compatible-server")

# Define the notes directory
NOTES_DIRECTORY = "/home/david/Sara/notes"

# Ensure notes directory exists
os.makedirs(NOTES_DIRECTORY, exist_ok=True)

def list_notes() -> Dict[str, Any]:
    """List all available notes with metadata"""
    try:
        notes = []
        for filename in os.listdir(NOTES_DIRECTORY):
            if filename.endswith('.json'):
                file_path = os.path.join(NOTES_DIRECTORY, filename)
                with open(file_path, 'r') as f:
                    note_data = json.load(f)
                    note_data['filename'] = filename
                    notes.append(note_data)
        
        # Sort notes by last modified time, newest first
        notes.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
        return {"notes": notes, "count": len(notes)}
    except Exception as e:
        logger.error(f"Error listing notes: {e}")
        return {"notes": [], "count": 0, "error": str(e)}

def get_note_names_for_prompt() -> str:
    """Get a formatted list of note names for the system prompt"""
    try:
        notes_result = list_notes()
        notes = notes_result.get("notes", [])
        
        if not notes:
            return "No notes available."
        
        note_list = ["Available notes:"]
        for note in notes:
            title = note.get('title', 'Untitled note')
            created = note.get('created', 'Unknown date')
            # Format the created date to be more readable if it's in ISO format
            if 'T' in created:
                try:
                    dt = datetime.fromisoformat(created)
                    created = dt.strftime("%Y-%m-%d")
                except:
                    pass
            
            # Add tags if available
            tags_str = ""
            if note.get('tags'):
                tags_str = f" [Tags: {', '.join(note['tags'])}]"
            
            note_list.append(f"- {title} (created: {created}){tags_str}")
        
        return "\n".join(note_list)
    except Exception as e:
        logger.error(f"Error getting note names for prompt: {e}")
        return "Error retrieving notes."

def create_note(title: str, content: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a new note"""
    try:
        # Generate a filename based on the title
        safe_title = "".join(c if c.isalnum() else "_" for c in title).lower()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{safe_title}_{timestamp}.json"
        file_path = os.path.join(NOTES_DIRECTORY, filename)
        
        # Create the note data
        now = datetime.now().isoformat()
        note_data = {
            "title": title,
            "content": content,
            "tags": tags or [],
            "created": now,
            "last_modified": now
        }
        
        # Write the note to file
        with open(file_path, 'w') as f:
            json.dump(note_data, f, indent=2)
        
        logger.info(f"Created new note: {title} ({filename})")
        
        # Return the note data with the filename
        note_data['filename'] = filename
        return note_data
    except Exception as e:
        logger.error(f"Error creating note: {e}")
        return {"error": str(e)}

def read_note(identifier: str) -> Dict[str, Any]:
    """Read a note by title or filename"""
    try:
        notes_result = list_notes()
        notes = notes_result.get("notes", [])
        
        # Try to find the note by exact title match first
        for note in notes:
            if note.get('title') == identifier:
                logger.info(f"Read note by title: {identifier}")
                return note
        
        # Then try to find by filename
        for note in notes:
            if note.get('filename') == identifier:
                logger.info(f"Read note by filename: {identifier}")
                return note
        
        # Lastly, try fuzzy matching on title
        for note in notes:
            if identifier.lower() in note.get('title', '').lower():
                logger.info(f"Read note by fuzzy match: {identifier} -> {note.get('title')}")
                return note
        
        logger.warning(f"Note not found: {identifier}")
        return {"error": f"Note not found: {identifier}"}
    except Exception as e:
        logger.error(f"Error reading note: {e}")
        return {"error": str(e)}

def append_note(identifier: str, content: str) -> Dict[str, Any]:
    """Append content to an existing note"""
    try:
        # First, find the note
        note = read_note(identifier)
        if "error" in note:
            return note
        
        # Get the file path
        filename = note.get('filename')
        file_path = os.path.join(NOTES_DIRECTORY, filename)
        
        # Append content and update timestamp
        note['content'] = note.get('content', '') + '\n\n' + content
        note['last_modified'] = datetime.now().isoformat()
        
        # Write the updated note
        with open(file_path, 'w') as f:
            json.dump(note, f, indent=2)
        
        logger.info(f"Appended to note: {note.get('title')} ({filename})")
        return note
    except Exception as e:
        logger.error(f"Error appending to note: {e}")
        return {"error": str(e)}

def delete_note(identifier: str) -> Dict[str, Any]:
    """Delete a note by title or filename"""
    try:
        # First, find the note
        note = read_note(identifier)
        if "error" in note:
            return note
        
        # Get the file path
        filename = note.get('filename')
        file_path = os.path.join(NOTES_DIRECTORY, filename)
        
        # Delete the file
        os.remove(file_path)
        
        logger.info(f"Deleted note: {note.get('title')} ({filename})")
        return {"success": True, "message": f"Note '{note.get('title')}' deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        return {"error": str(e)}

def update_system_prompt_with_notes(original_prompt: str) -> str:
    """Update the system prompt to include available notes"""
    notes_info = get_note_names_for_prompt()
    
    # Check if the prompt already has notes information
    if "Available notes:" in original_prompt:
        # Replace the existing notes section
        lines = original_prompt.split('\n')
        new_lines = []
        skip_mode = False
        
        for line in lines:
            if line.strip() == "Available notes:":
                new_lines.append(notes_info)
                skip_mode = True
            elif skip_mode and line.strip().startswith('-'):
                continue  # Skip existing note entries
            else:
                skip_mode = False
                new_lines.append(line)
        
        return '\n'.join(new_lines)
    else:
        # Append notes information at the end
        return f"{original_prompt}\n\n{notes_info}"

# Refresh notes information in the system prompt periodically
# This function can be called by a background task or scheduler
def refresh_notes_in_system_prompt(system_prompt_var):
    """Update the global system prompt with fresh notes information"""
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = update_system_prompt_with_notes(system_prompt_var)
    logger.info("Refreshed notes in system prompt")
    return SYSTEM_PROMPT