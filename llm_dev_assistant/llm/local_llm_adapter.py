# llm_dev_assistant/llm/local_llm_adapter.py
import os
import subprocess
import tempfile
import json
from typing import Dict, List, Any, Optional
from .llm_interface import LLMInterface


class LocalLLMAdapter(LLMInterface):
    """Adapter for local LLM models."""

    def __init__(self, model_path: str, max_tokens: int = 2048):
        """
        Initialize the local LLM adapter.

        Args:
            model_path: Path to the model
            max_tokens: Maximum number of tokens for generation
        """
        self.model_path = model_path
        self.max_tokens = max_tokens

        # Verify model exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

    def query(self, prompt: str, context: Optional[str] = None,
              system_message: Optional[str] = None) -> str:
        """
        Query the local LLM.

        Args:
            prompt: The main prompt to send
            context: Optional context to provide
            system_message: Optional system message

        Returns:
            Model response as a string
        """
        # Combine context and prompt
        full_prompt = ""
        if context:
            full_prompt += f"Context:\n{context}\n\n"
        full_prompt += prompt

        # Prepare input for model
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            input_file = f.name
            f.write(full_prompt)

        # Prepare system message if provided
        sys_msg_file = None
        if system_message:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                sys_msg_file = f.name
                f.write(system_message)

        try:
            # Build command
            cmd = [
                self.model_path,
                "--input", input_file,
                "--max-tokens", str(self.max_tokens)
            ]

            if sys_msg_file:
                cmd.extend(["--system", sys_msg_file])

            # Execute command
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            # Check for errors
            if process.returncode != 0:
                raise RuntimeError(f"Error running local LLM: {process.stderr}")

            # Return output
            return process.stdout.strip()

        finally:
            # Clean up temporary files
            os.unlink(input_file)
            if sys_msg_file:
                os.unlink(sys_msg_file)

    def get_code_suggestions(self, prompt: str, existing_code: Optional[str] = None,
                             context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get code suggestions from the local LLM.

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