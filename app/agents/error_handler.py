import logging
import traceback
from typing import Dict, Any

from app.agents.base import BaseAgent
from app.utils.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class ErrorHandlerAgent(BaseAgent):
    def __init__(self):
        super().__init__("error_handler")

    async def _execute(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors gracefully and return user-friendly message"""
        return self.handle_error(error, context)

    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors gracefully and return user-friendly message"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log the error for debugging
        logger.error(f"Error in {context}: {error_type} - {error_message}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Return user-friendly error response based on error type
        if "classification" in error_message.lower() or "ClassificationError" in error_type:
            return {
                "error": True,
                "type": "Classification Error",
                "message": "Unable to classify the text. Please try again.",
                "suggestion": "Check if the text contains valid content for analysis."
            }
        elif "retrieval" in error_message.lower() or "RetrievalError" in error_type:
            return {
                "error": True,
                "type": "Policy Retrieval Error", 
                "message": "Unable to retrieve relevant policies.",
                "suggestion": "The policy database may be temporarily unavailable."
            }
        elif "llm" in error_message.lower() or "dial" in error_message.lower():
            return {
                "error": True,
                "type": "AI Service Error",
                "message": "The AI service is temporarily unavailable.",
                "suggestion": "Please try again in a few moments."
            }
        elif "AgentExecutionError" in error_type:
            return {
                "error": True,
                "type": "Agent Error",
                "message": "An agent failed to complete its task.",
                "suggestion": "Please retry the operation."
            }
        else:
            return {
                "error": True,
                "type": "System Error",
                "message": "An unexpected error occurred.",
                "suggestion": "If the problem persists, contact support."
            }

    def validate_input(self, text: str) -> Dict[str, Any]:
        """Validate user input"""
        if not text or not text.strip():
            logger.warning("Empty input received")
            return {
                "valid": False,
                "error": "Empty input",
                "message": "Please enter some text to analyze."
            }
        
        if len(text.strip()) < 3:
            logger.warning("Input too short")
            return {
                "valid": False,
                "error": "Input too short", 
                "message": "Please enter at least 3 characters for analysis."
            }
            
        if len(text) > 5000:
            logger.warning("Input exceeds maximum length")
            return {
                "valid": False,
                "error": "Input too long",
                "message": "Please limit input to 5000 characters or less."
            }
        
        logger.debug("Input validation successful")
        return {"valid": True}