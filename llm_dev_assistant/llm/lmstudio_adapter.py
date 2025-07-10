# llm_dev_assistant/llm/lmstudio_adapter.py
import json
import subprocess
from typing import Dict, List, Any, Optional
import re

from .llm_interface import LLMInterface
from ..log_system.logger import logger


class LMStudioAdapter(LLMInterface):
    """Adapter for LM Studio LLMs."""

    def __init__(self, model_name: str, host: str = "localhost", port: int = 1234,
                 context_length: int = 4096, mount_if_needed: bool = True):
        """
        Initialize the LM Studio adapter.

        Args:
            model_name: Name or path of the model
            host: LM Studio server host
            port: LM Studio server port
            context_length: Maximum context length for the model
            mount_if_needed: Whether to mount the model if it's not already mounted
        """
        self.log = logger.get_logger("lmstudio_adapter")
        self.log.info(f"Initializing LM Studio adapter for model: {model_name}")

        self.model_name = model_name
        self.host = host
        self.port = port
        self.context_length = context_length
        self.base_url = f"http://{host}:{port}"

        if mount_if_needed:
            self.mount_model()

    def mount_model(self) -> bool:
        """
        Mount the model in LM Studio.

        Returns:
            True if successful, False otherwise
        """
        self.log.info(f"Mounting model: {self.model_name}")

        try:
            # Check if LM Studio is running
            # This is a simplified example - in a real implementation,
            # you would communicate with LM Studio API to mount the model
            process = subprocess.run(
                ["curl", "-s", f"{self.base_url}/v1/models"],
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                self.log.error(f"LM Studio server not found at {self.base_url}")
                return False

            # Check if model is already mounted
            response = json.loads(process.stdout)
            for model in response.get("data", []):
                if model.get("id") == self.model_name:
                    self.log.info(f"Model {self.model_name} is already mounted")
                    return True

            # Mount the model (example placeholder)
            # In a real implementation, you would call the appropriate LM Studio API
            self.log.info(f"Mounting model {self.model_name} with context length {self.context_length}")
            process = subprocess.run(
                ["curl", "-s", "-X", "POST",
                 f"{self.base_url}/v1/models/mount",
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps({
                    "model_path": self.model_name,
                    "context_length": self.context_length
                })],
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                self.log.error(f"Failed to mount model: {process.stderr}")
                return False

            self.log.info(f"Successfully mounted model: {self.model_name}")
            return True

        except Exception as e:
            self.log.error(f"Error mounting model: {str(e)}")
            return False

    def unmount_model(self) -> bool:
        """
        Unmount the model from LM Studio.

        Returns:
            True if successful, False otherwise
        """
        self.log.info(f"Unmounting model: {self.model_name}")

        try:
            # In a real implementation, you would call the appropriate LM Studio API
            process = subprocess.run(
                ["curl", "-s", "-X", "POST",
                 f"{self.base_url}/v1/models/unmount",
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps({
                    "model_id": self.model_name
                })],
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                self.log.error(f"Failed to unmount model: {process.stderr}")
                return False

            self.log.info(f"Successfully unmounted model: {self.model_name}")
            return True

        except Exception as e:
            self.log.error(f"Error unmounting model: {str(e)}")
            return False

    def query(self, prompt: str, context: Optional[str] = None,
              system_message: Optional[str] = None) -> str:
        """
        Query the LM Studio model with a prompt.

        Args:
            prompt: The main prompt to send to the LLM
            context: Optional context to provide
            system_message: Optional system message to guide the LLM behavior

        Returns:
            LLM response as a string
        """
        self.log.info(f"Querying LM Studio model: {self.model_name}")

        # Build the messages
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        if context:
            messages.append({"role": "user", "content": f"Context information:\n{context}"})

        messages.append({"role": "user", "content": prompt})

        try:
            # Call the LM Studio API
            process = subprocess.run(
                ["curl", "-s", "-X", "POST",
                 f"{self.base_url}/v1/chat/completions",
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps({
                    "model": self.model_name,
                    "messages": messages
                })],
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                self.log.error(f"Failed to query model: {process.stderr}")
                return f"Error querying model: {process.stderr}"

            response = json.loads(process.stdout)
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            self.log.info("Successfully received response from LM Studio")
            return content

        except Exception as e:
            self.log.error(f"Error querying model: {str(e)}")
            return f"Error querying model: {str(e)}"

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