# llm_dev_assistant/llm/openai_adapter.py
import os
import time
from typing import Dict, List, Any, Optional
import logging
import openai
from .llm_interface import LLMInterface
from ..utils.file_utils import read_file
from ..log_system.logger import logger


class OpenAIAdapter(LLMInterface):
    """Adapter for OpenAI's API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the OpenAI adapter.

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: OpenAI model to use
        """
        self.log = logger.get_logger("openai_adapter")
        self.log.info(f"Initializing OpenAI adapter with model: {model}")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.log.error("OpenAI API key not provided or found in environment")
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")

        openai.api_key = self.api_key
        self.model = model
        self.log.debug(f"OpenAI adapter initialized successfully")

    def query(self, prompt: str, context: Optional[str] = None,
              system_message: Optional[str] = None) -> str:
        """
        Query the OpenAI model.

        Args:
            prompt: The main prompt to send
            context: Optional context to provide
            system_message: Optional system message

        Returns:
            Model response as a string
        """
        self.log.info(f"Querying OpenAI model: {self.model}")

        # Log the prompt details
        logger.log_llm_prompt(self.model, prompt, context, system_message)

        messages = []

        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Add context if provided
        if context:
            messages.append({"role": "user", "content": f"Context information:\n{context}"})

        # Add the main prompt
        messages.append({"role": "user", "content": prompt})

        # Handle potential API errors with retries
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.log.debug(f"Sending request to OpenAI API (attempt {attempt + 1}/{max_retries})")
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages
                )
                content = response.choices[0].message.content

                # Log the response
                logger.log_llm_response(self.model, content)

                self.log.info(f"Successfully received response from OpenAI API")
                return content

            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
                self.log.warning(f"API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    self.log.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)  # Exponential backoff
                else:
                    self.log.error(f"Failed to get response after {max_retries} attempts: {str(e)}")
                    raise RuntimeError(f"Failed to get response from OpenAI API after {max_retries} attempts: {str(e)}")
            except Exception as e:
                self.log.error(f"Unexpected error querying OpenAI API: {str(e)}")
                raise RuntimeError(f"Error querying OpenAI API: {str(e)}")

    # Update the remaining methods with log_system...
    # (For brevity, I won't modify every method, but the pattern is similar)