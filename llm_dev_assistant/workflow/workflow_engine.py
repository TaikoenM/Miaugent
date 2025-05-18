# llm_dev_assistant/workflow/workflow_engine.py
import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from ..llm.llm_interface import LLMInterface
from ..parsers.parser_interface import ParserInterface
from ..code_manager.code_analyzer import CodeAnalyzer
from ..code_manager.change_implementer import ChangeImplementer
from ..code_manager.test_manager import TestManager
from ..logging.logger import logger


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
        self.log = logger.get_logger("workflow_engine")
        self.log.info("Initializing workflow engine")

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

        self.log.debug("Workflow engine initialized successfully")

    def initialize_project(self, project_path: str) -> Dict[str, Any]:
        """
        Initialize a project for development.

        Args:
            project_path: Path to the project directory

        Returns:
            Project context information
        """
        self.log.info(f"Initializing project: {project_path}")
        start_time = time.time()

        # Validate project path
        if not os.path.exists(project_path):
            self.log.error(f"Project path does not exist: {project_path}")
            raise ValueError(f"Project path does not exist: {project_path}")

        self.workflow_state["project_path"] = project_path
        self.log.debug(f"Set project_path in workflow state: {project_path}")

        try:
            # Parse the project
            self.log.debug(f"Parsing project directory: {project_path}")
            parsed_data = self.parser.parse_directory(project_path)
            self.log.info(
                f"Project parsed successfully. Found {len(parsed_data.get('scripts', []))} scripts and {len(parsed_data.get('other_assets', []))} other assets")

            # Log sample of discovered assets
            if parsed_data.get('scripts'):
                script_names = [script.get('name', 'unknown') for script in parsed_data.get('scripts', [])[:5]]
                self.log.debug(
                    f"Sample scripts: {', '.join(script_names)}{'...' if len(parsed_data.get('scripts', [])) > 5 else ''}")

            # Generate context for LLM
            self.log.debug(f"Generating context for LLM")
            context = self.parser.generate_context(parsed_data)
            context_size = len(context)
            self.log.debug(f"Generated context of {context_size} characters")

            # Store context in workflow state
            self.workflow_state["context"] = context
            self.log.debug(f"Context stored in workflow state")

            # Initialize code analysis
            self.log.debug(f"Analyzing project code structure")
            code_structure = self.code_analyzer.analyze_project(project_path)

            # Log code structure statistics
            file_count = len(code_structure.get('files', []))
            function_count = len(code_structure.get('functions', {}))
            class_count = len(code_structure.get('classes', {}))
            module_count = len(code_structure.get('modules', {}))

            self.log.info(
                f"Code analysis complete: {file_count} files, {function_count} functions, {class_count} classes, {module_count} modules")

            # Create result
            result = {
                "project_path": project_path,
                "context": context,
                "code_structure": code_structure,
                "status": "initialized"
            }

            elapsed_time = time.time() - start_time
            self.log.info(f"Project initialized successfully in {elapsed_time:.2f}s: {project_path}")

            return result

        except Exception as e:
            self.log.error(f"Error initializing project: {str(e)}", exc_info=True)
            raise

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
        self.log.info(f"Requesting code implementation for task: {task_description[:100]}...")
        start_time = time.time()

        # Validate project initialized
        if not self.workflow_state["project_path"]:
            self.log.error("Cannot request code implementation: project not initialized")
            raise RuntimeError("Project not initialized. Call initialize_project first.")

        # Set current task
        self.workflow_state["current_task"] = task_description
        self.log.debug(f"Set current task in workflow state")

        try:
            # Get context information
            context = self.workflow_state["context"]
            context_size = len(context) if context else 0
            self.log.debug(f"Using context of {context_size} characters")

            # Get existing code if file_path is provided
            existing_code = None
            if file_path:
                self.log.debug(f"Getting existing code from file: {file_path}")
                if not os.path.exists(file_path):
                    self.log.warning(f"File path does not exist: {file_path}")
                else:
                    existing_code = self.parser.get_file_content(file_path)
                    self.log.debug(f"Retrieved {len(existing_code)} characters of existing code")

            # Request code from LLM
            self.log.debug("Sending request to LLM for code suggestions")
            logger.log_function_call("llm.get_code_suggestions", {
                "prompt": f"{task_description[:50]}...",
                "existing_code": f"{existing_code[:50]}..." if existing_code else None
            })

            suggestions = self.llm.get_code_suggestions(
                prompt=task_description,
                existing_code=existing_code,
                context=context
            )

            self.log.debug("Received code suggestions from LLM")

            # Log suggestion details
            code_length = len(suggestions.get("code", "")) if suggestions.get("code") else 0
            explanation_length = len(suggestions.get("explanation", "")) if suggestions.get("explanation") else 0

            self.log.debug(
                f"Code suggestions: {code_length} characters of code, {explanation_length} characters of explanation")

            # Return suggestions
            result = {
                "task": task_description,
                "file_path": file_path,
                "suggestions": suggestions,
                "status": "code_suggested"
            }

            elapsed_time = time.time() - start_time
            self.log.info(f"Code implementation request completed in {elapsed_time:.2f}s")

            return result

        except Exception as e:
            self.log.error(f"Error requesting code implementation: {str(e)}", exc_info=True)
            raise

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
        self.log.info("Verifying code implementation against requirements")
        start_time = time.time()

        original_code_length = len(original_code)
        new_code_length = len(new_code)
        requirements_length = len(requirements)

        self.log.debug(
            f"Original code: {original_code_length} chars, New code: {new_code_length} chars, Requirements: {requirements_length} chars")

        try:
            # Request verification from LLM
            self.log.debug("Sending verification request to LLM")
            logger.log_function_call("llm.verify_code_changes", {
                "original_code": f"{original_code[:50]}...",
                "new_code": f"{new_code[:50]}...",
                "requirements": f"{requirements[:50]}..."
            })

            verification = self.llm.verify_code_changes(
                original_code=original_code,
                new_code=new_code,
                requirements=requirements
            )

            self.log.debug("Received verification results from LLM")

            # Log verification results
            meets_requirements = verification.get("meets_requirements", "Unknown")
            issues_count = len(verification.get("issues", []))
            suggestions_count = len(verification.get("suggestions", []))

            self.log.info(
                f"Verification result: meets_requirements={meets_requirements}, issues={issues_count}, suggestions={suggestions_count}")

            if meets_requirements is True:
                self.log.info("Verification passed: code meets requirements")
            elif meets_requirements is False:
                self.log.warning("Verification failed: code does not meet requirements")
                if issues_count > 0:
                    for i, issue in enumerate(verification.get("issues", [])):
                        self.log.warning(f"Issue {i + 1}: {issue}")
            else:
                self.log.warning(f"Verification returned unclear result: {meets_requirements}")

            result = {
                "original_code": original_code,
                "new_code": new_code,
                "requirements": requirements,
                "verification": verification,
                "status": "verified"
            }

            elapsed_time = time.time() - start_time
            self.log.info(f"Code verification completed in {elapsed_time:.2f}s")

            return result

        except Exception as e:
            self.log.error(f"Error verifying implementation: {str(e)}", exc_info=True)
            raise

    def implement_changes(self, file_path: str, new_code: str) -> Dict[str, Any]:
        """
        Implement code changes.

        Args:
            file_path: Path to the file to modify
            new_code: New code to implement

        Returns:
            Implementation results
        """
        self.log.info(f"Implementing code changes to file: {file_path}")
        start_time = time.time()

        # Validate file path
        if not os.path.exists(file_path) and os.path.dirname(file_path):
            if not os.path.exists(os.path.dirname(file_path)):
                self.log.warning(f"Directory does not exist for file: {file_path}")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                self.log.info(f"Created directory: {os.path.dirname(file_path)}")
            self.log.info(f"File will be created: {file_path}")

        new_code_length = len(new_code)
        self.log.debug(f"New code length: {new_code_length} characters")

        try:
            # Get original code
            original_code = None
            if os.path.exists(file_path):
                self.log.debug(f"Getting original code from file: {file_path}")
                original_code = self.parser.get_file_content(file_path)
                original_code_length = len(original_code)
                self.log.debug(f"Original code length: {original_code_length} characters")

                # Log diff summary
                if original_code:
                    lines_added = len(new_code.splitlines()) - len(original_code.splitlines())
                    self.log.debug(
                        f"Code diff summary: {abs(lines_added)} lines {'added' if lines_added >= 0 else 'removed'}")
            else:
                self.log.debug(f"No original code (file will be created)")

            # Implement changes
            self.log.debug(f"Implementing changes to file: {file_path}")
            logger.log_function_call("change_implementer.implement_changes", {
                "file_path": file_path,
                "new_code": f"{new_code[:50]}..."
            })

            result = self.change_implementer.implement_changes(
                file_path=file_path,
                new_code=new_code
            )

            self.log.debug(f"Changes implementation result: status={result.get('status')}")

            if result.get('status') == 'success':
                self.log.info(f"Successfully implemented changes to file: {file_path}")
            elif result.get('status') == 'warning':
                self.log.warning(f"Implemented changes with warnings: {result.get('message')}")
            else:
                self.log.error(f"Failed to implement changes: {result.get('message')}")

            # Update workflow state
            if self.workflow_state["current_task"]:
                self.workflow_state["completed_tasks"].append(self.workflow_state["current_task"])
                self.log.debug(f"Added current task to completed tasks: {self.workflow_state['current_task']}")
                self.workflow_state["current_task"] = None
                self.log.debug(f"Reset current task to None")

            implementation_result = {
                "file_path": file_path,
                "original_code": original_code,
                "new_code": new_code,
                "result": result,
                "status": "implemented"
            }

            elapsed_time = time.time() - start_time
            self.log.info(f"Code implementation completed in {elapsed_time:.2f}s")

            return implementation_result

        except Exception as e:
            self.log.error(f"Error implementing changes: {str(e)}", exc_info=True)
            raise

    def verify_tests(self, file_path: str) -> Dict[str, Any]:
        """
        Verify if existing tests cover the modified file.

        Args:
            file_path: Path to the modified file

        Returns:
            Test verification results
        """
        self.log.info(f"Verifying tests for file: {file_path}")
        start_time = time.time()

        # Validate file path
        if not os.path.exists(file_path):
            self.log.error(f"File does not exist: {file_path}")
            raise ValueError(f"File does not exist: {file_path}")

        try:
            # Identify relevant tests
            self.log.debug(f"Finding relevant tests for file: {file_path}")
            logger.log_function_call("test_manager.find_relevant_tests", {
                "file_path": file_path
            })

            relevant_tests = self.test_manager.find_relevant_tests(file_path)
            test_count = len(relevant_tests)

            if test_count > 0:
                self.log.info(f"Found {test_count} relevant test files")
                for i, test_path in enumerate(relevant_tests):
                    self.log.debug(f"Test {i + 1}: {test_path}")
            else:
                self.log.warning(f"No relevant test files found for: {file_path}")

            # Analyze test coverage
            self.log.debug(f"Analyzing test coverage")
            logger.log_function_call("test_manager.analyze_test_coverage", {
                "file_path": file_path,
                "test_paths": relevant_tests
            })

            coverage_analysis = self.test_manager.analyze_test_coverage(file_path, relevant_tests)

            coverage_estimate = coverage_analysis.get("coverage_estimate", "unknown")
            coverage_percent = coverage_analysis.get("coverage_percent", None)

            if coverage_percent is not None:
                self.log.info(f"Test coverage: {coverage_percent:.1f}% ({coverage_estimate})")
            else:
                self.log.info(f"Test coverage estimate: {coverage_estimate}")

            if coverage_estimate in ["poor", "missing"]:
                self.log.warning(f"Insufficient test coverage for file: {file_path}")

            # Check for errors in analysis
            if "error" in coverage_analysis:
                self.log.warning(f"Error in coverage analysis: {coverage_analysis['error']}")

            result = {
                "file_path": file_path,
                "relevant_tests": relevant_tests,
                "coverage_analysis": coverage_analysis,
                "status": "tests_verified"
            }

            elapsed_time = time.time() - start_time
            self.log.info(f"Test verification completed in {elapsed_time:.2f}s")

            return result

        except Exception as e:
            self.log.error(f"Error verifying tests: {str(e)}", exc_info=True)
            raise

    def run_integration_tests(self, test_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run integration tests.

        Args:
            test_paths: Optional specific test paths to run

        Returns:
            Test results
        """
        self.log.info("Running integration tests")
        start_time = time.time()

        if test_paths:
            test_count = len(test_paths)
            self.log.debug(f"Running {test_count} specified test paths")
            for i, test_path in enumerate(test_paths):
                self.log.debug(f"Test {i + 1}: {test_path}")
                if not os.path.exists(test_path):
                    self.log.warning(f"Test file does not exist: {test_path}")
        else:
            self.log.debug("No test paths specified, will search for all tests")

        try:
            # Run tests
            self.log.debug("Running tests")
            logger.log_function_call("test_manager.run_tests", {
                "test_paths": test_paths
            })

            test_results = self.test_manager.run_tests(test_paths)

            # Log test results
            success = test_results.get("success", False)
            result_count = len(test_results.get("results", {}))

            if success:
                self.log.info(f"All tests passed ({result_count} test files)")
            else:
                failed_count = sum(1 for r in test_results.get("results", {}).values() if not r.get("success", False))
                self.log.warning(f"Tests failed: {failed_count} of {result_count} test files failed")

                # Log details of failed tests
                for test_path, result in test_results.get("results", {}).items():
                    if not result.get("success", False):
                        error_msg = result.get("error", "")
                        exit_code = result.get("exit_code", "unknown")
                        self.log.warning(f"Failed test: {test_path}, exit code: {exit_code}, error: {error_msg}")

            # Include error if no tests found or run
            if "error" in test_results:
                self.log.warning(f"Error in test execution: {test_results['error']}")

            result = {
                "test_paths": test_paths,
                "results": test_results,
                "status": "tests_run"
            }

            elapsed_time = time.time() - start_time
            self.log.info(f"Integration tests completed in {elapsed_time:.2f}s")

            return result

        except Exception as e:
            self.log.error(f"Error running integration tests: {str(e)}", exc_info=True)
            raise

    def plan_next_steps(self, project_goals: List[str]) -> Dict[str, Any]:
        """
        Plan next development steps.

        Args:
            project_goals: Project goals

        Returns:
            Next steps plan
        """
        self.log.info("Planning next development steps")
        start_time = time.time()

        # Validate project initialized
        if not self.workflow_state["project_path"]:
            self.log.error("Cannot plan next steps: project not initialized")
            raise RuntimeError("Project not initialized. Call initialize_project first.")

        goal_count = len(project_goals)
        self.log.debug(f"Planning based on {goal_count} project goals")
        for i, goal in enumerate(project_goals):
            self.log.debug(f"Goal {i + 1}: {goal}")

        completed_task_count = len(self.workflow_state["completed_tasks"])
        self.log.debug(f"Project has {completed_task_count} completed tasks")

        try:
            # Get current state
            current_state = {
                "project_path": self.workflow_state["project_path"],
                "completed_tasks": self.workflow_state["completed_tasks"],
                "pending_tasks": self.workflow_state["pending_tasks"]
            }

            # Request planning from LLM
            self.log.debug("Sending planning request to LLM")
            logger.log_function_call("llm.plan_next_steps", {
                "current_state": "...current_state...",
                "project_goals": project_goals
            })

            next_steps = self.llm.plan_next_steps(
                current_state=current_state,
                project_goals=project_goals
            )

            # Log planning results
            step_count = len(next_steps)
            self.log.info(f"LLM suggested {step_count} next development steps")

            # Log step priorities
            high_priority = sum(1 for step in next_steps if step.get("priority") == "high")
            medium_priority = sum(1 for step in next_steps if step.get("priority") == "medium")
            low_priority = sum(1 for step in next_steps if step.get("priority") == "low")

            self.log.debug(f"Step priorities: {high_priority} high, {medium_priority} medium, {low_priority} low")

            # Log high priority steps
            self.log.debug("High priority steps:")
            for i, step in enumerate(next_steps):
                if step.get("priority") == "high":
                    self.log.debug(f"  - {step.get('task', 'Unknown task')}")

            # Update workflow state
            self.workflow_state["pending_tasks"] = next_steps
            self.log.debug(f"Updated pending_tasks in workflow state with {step_count} tasks")

            result = {
                "current_state": current_state,
                "goals": project_goals,
                "next_steps": next_steps,
                "status": "planned"
            }

            elapsed_time = time.time() - start_time
            self.log.info(f"Planning completed in {elapsed_time:.2f}s")

            return result

        except Exception as e:
            self.log.error(f"Error planning next steps: {str(e)}", exc_info=True)
            raise

    def save_workflow_state(self, output_path: str) -> str:
        """
        Save the current workflow state to a file.

        Args:
            output_path: Path to save the state

        Returns:
            Path to the saved file
        """
        self.log.info(f"Saving workflow state to: {output_path}")
        start_time = time.time()

        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                self.log.debug(f"Creating directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)

            # Save state to file
            with open(output_path, 'w', encoding='utf-8') as f:
                self.log.debug(f"Writing workflow state to file")
                json.dump(self.workflow_state, f, indent=2)

            state_size = os.path.getsize(output_path)
            self.log.debug(f"Workflow state saved: {state_size} bytes")

            elapsed_time = time.time() - start_time
            self.log.info(f"Workflow state saved in {elapsed_time:.2f}s to: {output_path}")

            return output_path

        except Exception as e:
            self.log.error(f"Error saving workflow state: {str(e)}", exc_info=True)
            return f"Error saving workflow state: {str(e)}"

    def load_workflow_state(self, input_path: str) -> Dict[str, Any]:
        """
        Load workflow state from a file.

        Args:
            input_path: Path to the state file

        Returns:
            Loaded workflow state
        """
        self.log.info(f"Loading workflow state from: {input_path}")
        start_time = time.time()

        # Validate input path
        if not os.path.exists(input_path):
            self.log.error(f"Workflow state file does not exist: {input_path}")
            return {"error": f"Workflow state file does not exist: {input_path}"}

        try:
            # Load state from file
            with open(input_path, 'r', encoding='utf-8') as f:
                self.log.debug(f"Reading workflow state from file")
                loaded_state = json.load(f)

            # Validate loaded state
            required_keys = ["project_path", "completed_tasks", "pending_tasks"]
            missing_keys = [key for key in required_keys if key not in loaded_state]

            if missing_keys:
                self.log.warning(f"Loaded state is missing keys: {', '.join(missing_keys)}")

            # Update workflow state
            self.workflow_state = loaded_state
            self.log.debug(f"Updated workflow state from loaded file")

            # Log state summary
            project_path = self.workflow_state.get("project_path", "Not set")
            completed_tasks = len(self.workflow_state.get("completed_tasks", []))
            pending_tasks = len(self.workflow_state.get("pending_tasks", []))
            has_context = "context" in self.workflow_state and self.workflow_state["context"] is not None

            self.log.info(
                f"Loaded state summary: project={project_path}, completed_tasks={completed_tasks}, pending_tasks={pending_tasks}, has_context={has_context}")

            elapsed_time = time.time() - start_time
            self.log.info(f"Workflow state loaded in {elapsed_time:.2f}s from: {input_path}")

            return self.workflow_state

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in workflow state file: {str(e)}"
            self.log.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error loading workflow state: {str(e)}"
            self.log.error(error_msg, exc_info=True)
            return {"error": error_msg}