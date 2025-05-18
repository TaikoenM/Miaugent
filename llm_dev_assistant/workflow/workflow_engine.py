# llm_dev_assistant/workflow/workflow_engine.py
import os
import json
from typing import Dict, List, Any, Optional
from ..llm.llm_interface import LLMInterface
from ..parsers.parser_interface import ParserInterface
from ..code_manager.code_analyzer import CodeAnalyzer
from ..code_manager.change_implementer import ChangeImplementer
from ..code_manager.test_manager import TestManager
from ..github.github_client import GitHubClient
from ..github.repo_manager import RepoManager
from ..github.pr_manager import PRManager

class WorkflowEngine:
    """Orchestrates the development workflow with LLM assistance."""

    def __init__(self,
                 llm: LLMInterface,
                 parser: ParserInterface,
                 code_analyzer: CodeAnalyzer,
                 change_implementer: ChangeImplementer,
                 test_manager: TestManager,
                 github_client: Optional[GitHubClient] = None,
                 repo_manager: Optional[RepoManager] = None):
        # ... existing code ...
        self.github_client = github_client
        self.repo_manager = repo_manager

        if github_client and repo_manager:
            self.pr_manager = PRManager(github_client, repo_manager)
        else:
            self.pr_manager = None

    # Add these new methods:

    def clone_repository(self, repo_url: str, target_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Clone a Git repository.

        Args:
            repo_url: Repository URL
            target_dir: Target directory

        Returns:
            Dictionary with clone results
        """
        if not self.repo_manager:
            return {
                "status": "error",
                "message": "Repository manager not initialized"
            }

        result = self.repo_manager.clone_repo(repo_url, target_dir)

        if result["status"] == "success":
            # Initialize project with cloned repository
            self.initialize_project(result["target_dir"])

        return result

    def create_feature_branch(self, branch_name: str) -> Dict[str, Any]:
        """
        Create a feature branch.

        Args:
            branch_name: Branch name

        Returns:
            Dictionary with branch creation results
        """
        if not self.repo_manager:
            return {
                "status": "error",
                "message": "Repository manager not initialized"
            }

        if not self.workflow_state["project_path"]:
            return {
                "status": "error",
                "message": "Project not initialized"
            }

        return self.repo_manager.create_branch(
            self.workflow_state["project_path"],
            branch_name
        )

    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Commit changes.

        Args:
            message: Commit message
            files: List of files to commit

        Returns:
            Dictionary with commit results
        """
        if not self.repo_manager:
            return {
                "status": "error",
                "message": "Repository manager not initialized"
            }

        if not self.workflow_state["project_path"]:
            return {
                "status": "error",
                "message": "Project not initialized"
            }

        return self.repo_manager.commit_changes(
            self.workflow_state["project_path"],
            message,
            files
        )

    def create_pull_request(self, repo_name: str, title: str, description: str,
                            branch_name: str, base_branch: str = "main") -> Dict[str, Any]:
        """
        Create a pull request.

        Args:
            repo_name: GitHub repository name (format: 'owner/repo')
            title: PR title
            description: PR description
            branch_name: Branch with changes
            base_branch: Target branch

        Returns:
            Dictionary with PR creation results
        """
        if not self.pr_manager:
            return {
                "status": "error",
                "message": "PR manager not initialized"
            }

        if not self.workflow_state["project_path"]:
            return {
                "status": "error",
                "message": "Project not initialized"
            }

        return self.pr_manager.create_pr_for_changes(
            self.workflow_state["project_path"],
            repo_name,
            title,
            description,
            branch_name,
            base_branch
        )

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