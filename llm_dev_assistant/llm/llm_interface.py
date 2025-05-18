# llm_dev_assistant/llm/llm_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class LLMInterface(ABC):
    """Abstract interface for LLM interactions."""

    @abstractmethod
    def query(self, prompt: str, context: Optional[str] = None,
              system_message: Optional[str] = None) -> str:
        """
        Query the LLM with a prompt.

        Args:
            prompt: The main prompt to send to the LLM
            context: Optional context to provide
            system_message: Optional system message to guide the LLM behavior

        Returns:
            LLM response as a string
        """
        pass

    @abstractmethod
    def get_code_suggestions(self, prompt: str, existing_code: Optional[str] = None,
                             context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get code suggestions from the LLM.

        Args:
            prompt: Description of what code is needed
            existing_code: Optional existing code to modify
            context: Optional context about the project

        Returns:
            Dictionary containing suggested code and explanations
        """
        pass

    @abstractmethod
    def verify_code_changes(self, original_code: str, new_code: str,
                            requirements: str) -> Dict[str, Any]:
        """
        Verify if code changes meet requirements.

        Args:
            original_code: The original code
            new_code: The modified code
            requirements: Requirements the changes should meet

        Returns:
            Dictionary with verification results
        """
        pass

    @abstractmethod
    def plan_next_steps(self, current_state: Dict[str, Any],
                        project_goals: List[str]) -> List[Dict[str, Any]]:
        """
        Plan next development steps.

        Args:
            current_state: Current project state
            project_goals: Project goals and objectives

        Returns:
            List of next steps with details
        """
        pass