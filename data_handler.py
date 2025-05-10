# This file handles data processing and storage for candidate information
# It manages the collection and organization of candidate details

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

class CandidateInfo:
    """Class to represent and manage candidate information"""
    
    def __init__(self):
        """Initialize with empty candidate information"""
        self.data = {
            "full_name": None,
            "email": None,
            "phone": None,
            "years_experience": None,
            "desired_positions": None,
            "current_location": None,
            "tech_stack": None,
            "technical_responses": [],
            "conversation_timestamp": datetime.now().isoformat(),
            "conversation_complete": False
        }
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
    
    def update_field(self, field: str, value: Any) -> bool:
        """
        Update a specific field in the candidate data
        
        Args:
            field: The field name to update
            value: The new value
            
        Returns:
            bool: True if field was updated, False if field doesn't exist
        """
        if field in self.data:
            self.data[field] = value
            return True
        return False
    
    def get_field(self, field: str) -> Any:
        """
        Get the value of a specific field
        
        Args:
            field: The field name to retrieve
            
        Returns:
            Any: The field value, or None if field doesn't exist
        """
        return self.data.get(field)
    
    def add_technical_response(self, question: str, answer: str) -> None:
        """
        Add a technical question and its answer
        
        Args:
            question: The technical question asked
            answer: The candidate's response
        """
        self.data["technical_responses"].append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
    
    def is_complete(self) -> bool:
        """
        Check if all required fields are filled
        
        Returns:
            bool: True if all required fields have values, False otherwise
        """
        required_fields = [
            "full_name", "email", "phone", "years_experience", 
            "desired_positions", "current_location", "tech_stack"
        ]
        return all(self.data.get(field) is not None for field in required_fields)
    
    def mark_complete(self) -> None:
        """Mark the conversation as complete"""
        self.data["conversation_complete"] = True
    
    def save_to_file(self) -> str:
        """
        Save candidate data to a JSON file
        
        Returns:
            str: Success message or error message
        """
        if not self.data["full_name"]:
            return "Cannot save data: Missing candidate name"
            
        # Create a sanitized filename using the candidate's name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^\w\s]', '', self.data['full_name'])  # Remove special chars
        filename = f"data/candidate_{safe_name.replace(' ', '_').lower()}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            return f"Data saved to {filename}"
        except Exception as e:
            return f"Error saving data: {str(e)}"
    
    def extract_info_from_conversation(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Extract candidate information from conversation history
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Dict[str, Any]: Extracted candidate information
        """
        extracted_info = {}
        
        for message in conversation_history:
            if message.get("role") != "user" or not message.get("content"):
                continue
                
            content = message["content"].lower()
            
            # Name extraction
            name_match = re.search(r'my name is\s+([a-zA-Z\s]+)', content)
            if name_match:
                # Properly capitalize names
                name_parts = name_match.group(1).strip().split()
                extracted_info["full_name"] = ' '.join(part.capitalize() for part in name_parts)
            
            # Email extraction
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', content)
            if email_match:
                extracted_info["email"] = email_match.group(0)
            
            # Phone extraction
            phone_match = re.search(r'(\+\d{1,3}|\d{1,3})[\s.-]?\d{3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}', content)
            if phone_match:
                extracted_info["phone"] = phone_match.group(0)
            
            # Years of experience extraction
            exp_match = re.search(r'(\d+)\s+years?\s+(of\s+)?experience', content)
            if exp_match:
                extracted_info["years_experience"] = int(exp_match.group(1))
            
            # Location extraction
            location_match = re.search(r'(live|based|located)\s+in\s+([a-zA-Z\s,]+)', content)
            if location_match:
                extracted_info["current_location"] = location_match.group(2).strip().title()
            
            # Tech stack extraction - look for common patterns
            tech_patterns = [
                r'work with\s+([a-zA-Z0-9\s,\.+#]+)',
                r'experience (?:with|in)\s+([a-zA-Z0-9\s,\.+#]+)',
                r'tech stack\s*(?:includes|is|:)?\s*([a-zA-Z0-9\s,\.+#]+)'
            ]
            
            for pattern in tech_patterns:
                tech_match = re.search(pattern, content)
                if tech_match:
                    tech_stack = tech_match.group(1).strip()
                    # Convert to list if comma-separated
                    if ',' in tech_stack:
                        tech_list = [tech.strip() for tech in tech_stack.split(',')]
                        extracted_info["tech_stack"] = tech_list
                    else:
                        extracted_info["tech_stack"] = tech_stack
                    break
        
        return extracted_info
    
    def update_from_conversation(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Extract and update candidate information from conversation history
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Dict[str, Any]: Dictionary of fields that were updated
        """
        extracted_info = self.extract_info_from_conversation(conversation_history)
        updated_fields = {}
        
        for field, value in extracted_info.items():
            if self.update_field(field, value):
                updated_fields[field] = value
                
        return updated_fields
    
    def get_missing_fields(self) -> List[str]:
        """
        Get a list of required fields that are still missing
        
        Returns:
            List[str]: List of missing field names
        """
        required_fields = [
            "full_name", "email", "phone", "years_experience", 
            "desired_positions", "current_location", "tech_stack"
        ]
        return [field for field in required_fields if self.data.get(field) is None]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert candidate info to a dictionary
        
        Returns:
            Dict[str, Any]: Dictionary representation of candidate data
        """
        return self.data.copy()