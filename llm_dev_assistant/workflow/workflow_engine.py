# llm_dev_assistant/workflow/workflow_engine.py
import os
import json
from typing import Dict, List, Any, Optional
from ..llm.llm_interface import LLMInterface
from ..parsers.parser_interface import ParserInterface
from ..code_manager.code_analyzer import CodeAnalyzer
from ..code_manager.change_implementer import ChangeImplementer
from ..code_manager.test_manager import TestManager


class WorkflowEngine:
    """Orchestrates the development workflow with LLM assistance."""

    def __init__(self,
                 llm: LLMInterface,
                 parser: ParserInterface,
                 code_analyzer: CodeAnalyzer,
                 change_implementer: ChangeImplementer,
                 test_manager: TestManager):
        """
        Initialize the workflow engine.

        Args:
            llm: LLM interface for queries
            parser: Parser for code analysis
            code_analyzer: Code analyzer
            change_implementer: Implements code changes
            test_manager: Manages tests
        """
        self.llm = llm
        self.parser = parser
        self.code_analyzer = code_analyzer
        self.change_implementer = change_implementer
        self.test_manager = test_manager

        # Workflow state
        self.workflow_state = {
            "project_path": None,
            "current_task": None,
            "completed_tasks": [],
            "pending_tasks": [],
            "context": None
        }

    def initialize_project(self, project_path: str) -> Dict[str, Any]:
        """
        Initialize a project for development.

        Args:
            project_path: Path to the project directory

        Returns:
            Project context information
        """
        self.workflow_state["project_path"] = project_path

        # Parse the project
        parsed_data = self.parser.parse_directory(project_path)

        # Generate context for LLM
        context = self.parser.generate_context(parsed_data)
        self.workflow_state["context"] = context

        # Initialize code analysis
        code_structure = self.code_analyzer.analyze_project(project_path)

        # Return project information
        return {
            "project_path": project_path,
            "context": context,
            "code_structure": code_structure,
            "status": "initialized"
        }

    def request_code_implementation(self, task_description: str,
                                    file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Request code implementation from LLM.

        Args:
            task_description: Description of the task
            file_path: Optional specific file to modify

        Returns:
            LLM response with code suggestions
        """
        # Set current task
        self.workflow_state["current_task"] = task_description

        # Get context information
        context = self.workflow_state["context"]

        # Get existing code if file_path is provided
        existing_code = None
        if file_path:
            existing_code = self.parser.get_file_content(file_path)

        # Request code from LLM
        suggestions = self.llm.get_code_suggestions(
            prompt=task_description,
            existing_code=existing_code,
            context=context
        )

        # Return suggestions
        return {
            "task": task_description,
            "file_path": file_path,
            "suggestions": suggestions,
            "status": "code_suggested"
        }

    def verify_implementation(self, original_code: str, new_code: str,
                              requirements: str) -> Dict[str, Any]:
        """
        Verify code implementation against requirements.

        Args:
            original_code: Original code
            new_code: New code implementation
            requirements: Requirements for verification

        Returns:
            Verification results
        """
        # Request verification from LLM
        verification = self.llm.verify_code_changes(
            original_code=original_code,
            new_code=new_code,
            requirements=requirements
        )

        return {
            "original_code": original_code,
            "new_code": new_code,
            "requirements": requirements,
            "verification": verification,
            "status": "verified"
        }

    def implement_changes(self, file_path: str, new_code: str) -> Dict[str, Any]:
        """
        Implement code changes.

        Args:
            file_path: Path to the file to modify
            new_code: New code to implement

        Returns:
            Implementation results
        """
        # Get original code
        original_code = self.parser.get_file_content(file_path)

        # Implement changes
        result = self.change_implementer.implement_changes(
            file_path=file_path,
            new_code=new_code
        )

        # Update workflow state
        if self.workflow_state["current_task"]:
            self.workflow_state["completed_tasks"].append(self.workflow_state["current_task"])
            self.workflow_state["current_task"] = None

        return {
            "file_path": file_path,
            "original_code": original_code,
            "new_code": new_code,
            "result": result,
            "status": "implemented"
        }

    def verify_tests(self, file_path: str) -> Dict[str, Any]:
        """
        Verify if existing tests cover the modified file.

        Args:
            file_path: Path to the modified file

        Returns:
            Test verification results
        """
        # Identify relevant tests
        relevant_tests = self.test_manager.find_relevant_tests(file_path)

        # Analyze test coverage
        coverage_analysis = self.test_manager.analyze_test_coverage(file_path, relevant_tests)

        return {
            "file_path": file_path,
            "relevant_tests": relevant_tests,
            "coverage_analysis": coverage_analysis,
            "status": "tests_verified"
        }

    def run_integration_tests(self, test_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run integration tests.

        Args:
            test_paths: Optional specific test paths to run

        Returns:
            Test results
        """
        # Run tests
        test_results = self.test_manager.run_tests(test_paths)

        return {
            "test_paths": test_paths,
            "results": test_results,
            "status": "tests_run"
        }

    def plan_next_steps(self, project_goals: List[str]) -> Dict[str, Any]:
        """
        Plan next development steps.

        Args:
            project_goals: Project goals

        Returns:
            Next steps plan
        """
        # Get current state
        current_state = {
            "project_path": self.workflow_state["project_path"],
            "completed_tasks": self.workflow_state["completed_tasks"],
            "pending_tasks": self.workflow_state["pending_tasks"]
        }

        # Request planning from LLM
        next_steps = self.llm.plan_next_steps(
            current_state=current_state,
            project_goals=project_goals
        )

        # Update workflow state
        self.workflow_state["pending_tasks"] = next_steps

        return {
            "current_state": current_state,
            "goals": project_goals,
            "next_steps": next_steps,
            "status": "planned"
        }

    def save_workflow_state(self, output_path: str) -> str:
        """
        Save the current workflow state to a file.

        Args:
            output_path: Path to save the state

        Returns:
            Path to the saved file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.workflow_state, f, indent=2)
            return output_path
        except Exception as e:
            return f"Error saving workflow state: {str(e)}"

    def load_workflow_state(self, input_path: str) -> Dict[str, Any]:
        """
        Load workflow state from a file.

        Args:
            input_path: Path to the state file

        Returns:
            Loaded workflow state
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                self.workflow_state = json.load(f)
            return self.workflow_state
        except Exception as e:
            return {"error": f"Error loading workflow state: {str(e)}"}