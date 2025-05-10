# This file contains utility functions used throughout the application
# It provides helper functions for various tasks

import re
from typing import Dict, List, Any, Optional

def extract_email(text: str) -> Optional[str]:
    """Extract email address from text using regex"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None

def extract_phone(text: str) -> Optional[str]:
    """Extract phone number from text using regex"""
    # Pattern matches various phone formats
    phone_pattern = r'(\+\d{1,3}[-.\s]?)?(\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}'
    match = re.search(phone_pattern, text)
    return match.group(0) if match else None

def extract_years_experience(text: str) -> Optional[int]:
    """Extract years of experience from text"""
    # Look for patterns like "5 years", "5 yrs", etc.
    pattern = r'(\d+)\s*(years?|yrs?)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # If no match with "years", try to find any number
    numbers = re.findall(r'\d+', text)
    if numbers:
        # Assume first number could be years of experience if reasonable
        year = int(numbers[0])
        if 0 <= year <= 50:  # Reasonable range for years of experience
            return year
    
    return None

def format_tech_stack(tech_stack: str) -> List[str]:
    """Format tech stack string into a list of technologies"""
    # Split by common separators and clean up
    technologies = re.split(r'[,;/]|\sand\s|\s+', tech_stack)
    return [tech.strip() for tech in technologies if tech.strip()]

def is_valid_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    return bool(re.fullmatch(pattern, email))

def is_valid_phone(phone: str) -> bool:
    """Validate phone number format"""
    # Simplified validation - would need to be enhanced for international numbers
    pattern = r'(\+\d{1,3}[-.\s]?)?(\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}'
    return bool(re.fullmatch(pattern, phone))

def analyze_conversation_state(conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze the conversation to determine current state
    
    Returns a dict with:
    - current_stage: greeting, info_collection, tech_stack, technical_questions, or closing
    - collected_info: dict of information collected so far
    - missing_info: list of fields still needed
    """
    # Default values
    state = {
        "current_stage": "greeting",
        "collected_info": {},
        "missing_info": [
            "full_name", "email", "phone", "years_experience", 
            "desired_positions", "current_location", "tech_stack"
        ]
    }
    
    # Basic logic to identify stage
    assistant_messages = [msg["content"] for msg in conversation_history if msg["role"] == "assistant"]
    user_messages = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
    
    if not assistant_messages:
        return state
        
    # Simple stage determination logic
    # This would be much more sophisticated in a real implementation
    last_assistant_msg = assistant_messages[-1].lower()
    
    if "technical question" in last_assistant_msg and "?" in last_assistant_msg:
        state["current_stage"] = "technical_questions"
    elif "tech stack" in last_assistant_msg or "programming languages" in last_assistant_msg:
        state["current_stage"] = "tech_stack"
    elif len(assistant_messages) <= 2:
        state["current_stage"] = "greeting"
    elif any(keyword in last_assistant_msg for keyword in ["thank you", "goodbye", "good luck"]):
        state["current_stage"] = "closing"
    else:
        state["current_stage"] = "info_collection"
    
    # This is a very basic implementation
    # In a real system, you would use NLP to extract information
    return state