import google.generativeai as genai
import time
from config import GEMINI_API_KEY, MODEL_NAME, EXIT_KEYWORDS
import prompts

class LLMService:
    """Service for handling interactions with the Language Model"""

    def __init__(self):
        """Initialize the LLM service with API key and rate limit handling"""
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(MODEL_NAME)
        self.conversation = self.model.start_chat(history=[])
        self.system_prompt_sent = False  # Ensure system prompt is sent only once
        self.last_request_time = 0  # Track last API request time
        self.conversation_ended = False  # Track if the conversation has ended

    def initialize_conversation(self):
        """Reset the conversation"""
        self.conversation = self.model.start_chat(history=[])
        self.system_prompt_sent = False
        self.conversation_ended = False  # Reset end flag

    def get_response(self, user_input: str) -> str:
        """Process user input and get a response from the model while handling rate limits"""
        if self.conversation_ended:
            return "The conversation has already ended. Please start a new session."

        if any(keyword in user_input.lower() for keyword in EXIT_KEYWORDS):
            return self._handle_exit()

        self._enforce_rate_limit()

        try:
            if not self.system_prompt_sent:
                self._send_system_prompt()

            response = self._send_with_retry(user_input)
            self.last_request_time = time.time()
            return response.text if response else "I'm having trouble processing your request. Please try again later."

        except Exception as e:
            print(f"Error in getting LLM response: {e}")
            return "I'm having trouble processing your request. Please try again later."

    def _send_system_prompt(self):
        """Send the system prompt once at the start of the conversation"""
        if self.system_prompt_sent:
            return  # Prevent duplicate system prompts
        try:
            self.conversation.send_message(prompts.SYSTEM_PROMPT)
            self.system_prompt_sent = True
        except Exception as e:
            print(f"Error sending system prompt: {e}")

    def _send_with_retry(self, message: str, max_retries: int = 3, delay: int = 6):
        """Retry sending messages in case of quota errors (429), ensuring only the final response is stored."""
        last_response = None
        for attempt in range(max_retries):
            try:
                last_response = self.conversation.send_message(message)
                return last_response  # Return immediately if successful
            except Exception as e:
                if "429" in str(e):
                    print(f"Rate limit hit. Waiting {delay} seconds before retrying (Attempt {attempt + 1})...")
                    time.sleep(delay)
                else:
                    print(f"Unexpected error: {e}")
                    return None
        return last_response  # Return the last successful response or None

    def _enforce_rate_limit(self, min_delay: int = 6):
        """Ensure a delay between consecutive API requests to prevent rate limit errors"""
        time_since_last = time.time() - self.last_request_time
        if time_since_last < min_delay:
            wait_time = min_delay - time_since_last
            print(f"Waiting {wait_time:.2f} seconds to comply with API rate limits...")
            time.sleep(wait_time)

    def _handle_exit(self) -> str:
        """Handle conversation exit, ensuring the message is not repeated."""
        if self.conversation_ended:
            return "The conversation has already ended. Please start a new session."

        self._enforce_rate_limit()
        try:
            response = self._send_with_retry(prompts.END_CONVERSATION_PROMPT)
            self.conversation_ended = True  # Mark conversation as ended
            return response.text if response else "Thank you for your time. Goodbye!"
        except Exception as e:
            print(f"Error in exit handling: {e}")
            return "Thank you for your time. Goodbye!"

    def start_conversation(self) -> str:
        """Start the conversation with a greeting"""
        if self.conversation_ended:
            return "The conversation has ended. Please start a new session."

        self._enforce_rate_limit()
        try:
            if not self.system_prompt_sent:
                self._send_system_prompt()

            response = self._send_with_retry(prompts.GREETING_PROMPT)
            return response.text if response else "Hello! I'm TalentScout Assistant. What's your name?"
        except Exception as e:
            print(f"Error in starting conversation: {e}")
            return "Hello! I'm TalentScout Assistant. What's your name?"

    def reset_conversation(self):
        """Reset the conversation history"""
        self.initialize_conversation()
