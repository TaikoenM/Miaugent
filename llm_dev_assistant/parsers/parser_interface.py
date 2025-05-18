# llm_dev_assistant/parsers/parser_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class ParserInterface(ABC):
    """Abstract interface for code parsers."""

    @abstractmethod
    def parse_directory(self, directory_path: str) -> Dict[str, Any]:
        """Parse a directory and return structured data about its contents."""
        pass

    @abstractmethod
    def generate_context(self, parsed_data: Dict[str, Any]) -> str:
        """Generate a contextual description for LLM prompting based on parsed data."""
        pass

    @abstractmethod
    def get_file_content(self, file_path: str) -> str:
        """Get the content of a specific file."""
        pass