# llm_dev_assistant/llm/vision_llm_adapter.py
import base64
from typing import Dict, List, Any, Optional

from .llm_interface import LLMInterface
from ..log_system.logger import logger


class VisionLLMAdapter(LLMInterface):
    """Adapter for vision-capable LLMs."""

    def __init__(self, model: str = "gpt-4-vision-preview", api_key: Optional[str] = None):
        """
        Initialize the Vision LLM adapter.

        Args:
            model: Model name for the vision LLM
            api_key: API key for the vision LLM service
        """
        self.log = logger.get_logger("vision_llm_adapter")
        self.log.info(f"Initializing Vision LLM adapter with model: {model}")

        self.model = model
        self.api_key = api_key

        # Initialize the API client
        import openai
        if api_key:
            openai.api_key = api_key
        self.client = openai

    def query(self, prompt: str, context: Optional[str] = None,
              system_message: Optional[str] = None) -> str:
        """
        Query the vision LLM with a text prompt.

        Args:
            prompt: The main prompt to send to the LLM
            context: Optional context to provide
            system_message: Optional system message to guide the LLM behavior

        Returns:
            LLM response as a string
        """
        self.log.info(f"Querying vision LLM model: {self.model}")

        # Build the messages
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        if context:
            messages.append({"role": "user", "content": f"Context information:\n{context}"})

        messages.append({"role": "user", "content": prompt})

        try:
            # Call the vision model API
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=messages
            )

            content = response.choices[0].message.content
            self.log.info(f"Successfully received response from vision LLM")
            return content

        except Exception as e:
            self.log.error(f"Error querying vision LLM: {str(e)}")
            return f"Error querying vision LLM: {str(e)}"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze an image using the vision LLM.

        Args:
            image_path: Path to the image file
            prompt: Prompt describing what to analyze in the image

        Returns:
            Analysis result as a string
        """
        self.log.info(f"Analyzing image: {image_path}")

        try:
            # Read the image file
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Prepare the messages for the API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }
            ]

            # Call the vision model API
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=messages
            )

            content = response.choices[0].message.content
            self.log.info(f"Successfully received image analysis from Vision LLM")
            return content

        except Exception as e:
            self.log.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"

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
        system_message = """
        You are an expert programmer assistant. 
        Provide clean, well-commented, and efficient code that follows best practices.
        Your response should be structured as follows:

        ```json
        {
            "code": "The complete suggested code",
            "explanation": "Explanation of key changes and design decisions",
            "testing_suggestions": "Suggestions for testing this code"
        }
        ```

        Provide the complete implementation, not just snippets.
        """

        full_prompt = prompt
        if existing_code:
            full_prompt += f"\n\nExisting code:\n```\n{existing_code}\n```"

        response = self.query(full_prompt, context, system_message)

        # Extract the JSON response
        try:
            import json
            import re

            # Find JSON block in the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group(1))
            else:
                # Try to extract a regular JSON object if not in a code block
                json_match = re.search(r'({.*})', response, re.DOTALL)
                if json_match:
                    response_json = json.loads(json_match.group(1))
                else:
                    # Fall back to a structured response
                    response_json = {
                        "code": self._extract_code(response),
                        "explanation": response,
                        "testing_suggestions": "No specific testing suggestions provided."
                    }

            return response_json
        except Exception as e:
            # Handle parsing errors
            code = self._extract_code(response)
            return {
                "code": code,
                "explanation": response.replace(code, "").strip(),
                "testing_suggestions": "No specific testing suggestions provided."
            }

    def _extract_code(self, text: str) -> str:
        """Extract code blocks from text."""
        import re

        # Look for code blocks with language specifier
        code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_blocks:
            return '\n\n'.join(code_blocks)

        # Look for code blocks without language specifier
        code_blocks = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_blocks:
            return '\n\n'.join(code_blocks)

        # If no code blocks found, return the original text
        return text

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
        system_message = """
        You are a code reviewer. Analyze the original code and the modified code, 
        and determine if the changes meet the specified requirements.

        Provide your response in the following JSON format:

        ```json
        {
            "meets_requirements": true/false,
            "analysis": "Detailed analysis of the changes",
            "issues": ["List of issues found, if any"],
            "suggestions": ["Improvement suggestions, if any"]
        }
        ```
        """

        prompt = f"""
        # Requirements
        {requirements}

        # Original Code
        ```
        {original_code}
        ```

        # Modified Code
        ```
        {new_code}
        ```

        Please review the changes and determine if they meet the requirements.
        """

        response = self.query(prompt, system_message=system_message)

        # Extract the JSON response
        try:
            import json
            import re

            # Find JSON block in the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # Try to find a JSON object
                json_match = re.search(r'({.*})', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    # Fall back to a structured response
                    return {
                        "meets_requirements": "Unknown",
                        "analysis": response,
                        "issues": [],
                        "suggestions": []
                    }
        except Exception as e:
            return {
                "meets_requirements": "Unknown",
                "analysis": response,
                "issues": [f"Failed to parse response: {str(e)}"],
                "suggestions": []
            }

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
        system_message = """
        You are a development planning assistant. Based on the current project state
        and goals, suggest the next development steps in priority order.

        Provide your response in the following JSON format:

        ```json
        [
            {
                "task": "Task description",
                "priority": "high/medium/low",
                "estimated_effort": "small/medium/large",
                "dependencies": ["Any dependent tasks"],
                "implementation_suggestions": "Implementation suggestions"
            },
            ...
        ]
        ```
        """

        # Format current state and goals
        current_state_str = "\n".join([f"- {k}: {v}" for k, v in current_state.items()])
        goals_str = "\n".join([f"- {goal}" for goal in project_goals])

        prompt = f"""
        # Current Project State
        {current_state_str}

        # Project Goals
        {goals_str}

        Please suggest the next development steps based on the current state and goals.
        """

        response = self.query(prompt, system_message=system_message)

        # Extract the JSON response
        try:
            import json
            import re

            # Find JSON block in the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # Try to find a JSON array
                json_match = re.search(r'(\[.*\])', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    # Fall back to a simple list
                    lines = [line.strip() for line in response.split('\n') if line.strip()]
                    tasks = []
                    for line in lines:
                        if line.startswith('-') or line.startswith('*'):
                            task = line[1:].strip()
                            tasks.append({
                                "task": task,
                                "priority": "medium",
                                "estimated_effort": "medium",
                                "dependencies": [],
                                "implementation_suggestions": ""
                            })

                    return tasks
        except Exception as e:
            # Return error as a task
            return [{
                "task": "Error parsing planning response",
                "priority": "high",
                "estimated_effort": "small",
                "dependencies": [],
                "implementation_suggestions": f"Fix error: {str(e)}\nOriginal response: {response}"
            }]