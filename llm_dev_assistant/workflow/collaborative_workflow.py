from typing import Dict, List, Any, Optional, Union
import os
import json
import time
import datetime

from llm_dev_assistant.llm.llm_selector import TaskPurpose, TaskComplexity, choose_llm
from llm_dev_assistant.workflow.workflow_engine import WorkflowEngine
from llm_dev_assistant.workflow.workflow_engine_llm_extension import WorkflowEngineLLMExtension
from llm_dev_assistant.log_system.logger import logger

"""
1. Multi-LLM Collaboration

Role-Based LLMs: Different LLMs take on specialized roles like architecture critic, security expert, and performance optimizer
Consensus Building: A mediator LLM synthesizes feedback from multiple perspectives
Specialized Selection: Automatically selects appropriate LLMs for different tasks (coding, planning, reviewing)

2. Anti-Stagnation Mechanisms

Loop Detection: Identifies when the workflow is stuck in a circular pattern
Iteration Limits: Enforces maximum iterations to prevent endless refinement cycles
Forced Progression: Will advance to the next phase if stuck in a loop

3. Comprehensive Workflow Phases

Initialization: Analyzes existing codebase and feature requirements
Planning: Generates task breakdown with dependencies and complexity estimates
Design Review: Multi-LLM critique and synthesis of plans
Development: Code generation with peer reviews by different LLMs
Testing: Test creation and verification
Reflection: Analyzes code quality and technical debt
Continuation: Prioritizes remaining tasks and plans next steps

4. Advanced Features

State Management: Complete workflow state can be saved/loaded between sessions
Contextual Awareness: Uses project context to make informed decisions
Self-Healing: Adapts to errors and unexpected outcomes
Documentation Trail: Keeps history of decisions, critiques, and iterations
"""

class CollaborativeLLMWorkflow:
    """
    Orchestrates a collaborative workflow with multiple LLMs to automatically
    develop software features with continuous feedback and improvement.
    """

    def __init__(self, workflow_engine: WorkflowEngine):
        """
        Initialize the collaborative LLM workflow.

        Args:
            workflow_engine: The base workflow engine
        """
        self.log = logger.get_logger("collaborative_workflow")
        self.log.info("Initializing collaborative LLM workflow")

        self.workflow_engine = workflow_engine
        self.llm_extension = WorkflowEngineLLMExtension(workflow_engine)

        # Track the state of the collaborative workflow
        self.state = {
            "feature_name": None,
            "feature_description": None,
            "current_phase": None,
            "task_history": [],
            "critique_history": [],
            "code_iterations": {},
            "llm_discussions": [],
            "tests_created": [],
            "current_plan": None,
            "plan_iterations": 0,
            "iteration_limit": 5,  # Default iteration limit
            "loop_detection": {}  # Track potentially circular discussions
        }

        # Define workflow phases
        self.phases = [
            "initialization",
            "planning",
            "design_review",
            "development",
            "testing",
            "reflection",
            "continuation"
        ]

        self.log.debug("Collaborative LLM workflow initialized")

    def start_feature_development(self, feature_name: str, feature_description: str) -> Dict[str, Any]:
        """
        Start the development process for a new feature.

        Args:
            feature_name: Name of the feature
            feature_description: Detailed description of the feature

        Returns:
            Initial workflow state
        """
        self.log.info(f"Starting development of feature: {feature_name}")

        # Reset workflow state
        self.state["feature_name"] = feature_name
        self.state["feature_description"] = feature_description
        self.state["current_phase"] = "initialization"
        self.state["task_history"] = []
        self.state["critique_history"] = []
        self.state["code_iterations"] = {}
        self.state["llm_discussions"] = []
        self.state["tests_created"] = []
        self.state["current_plan"] = None
        self.state["plan_iterations"] = 0

        # Start the workflow
        return self._execute_phase("initialization")

    def _execute_phase(self, phase: str) -> Dict[str, Any]:
        """
        Execute a specific phase of the workflow.

        Args:
            phase: The phase to execute

        Returns:
            Result of the phase execution
        """
        self.log.info(f"Executing workflow phase: {phase}")

        if phase not in self.phases:
            self.log.error(f"Invalid workflow phase: {phase}")
            return {"error": f"Invalid workflow phase: {phase}"}

        self.state["current_phase"] = phase

        if phase == "initialization":
            return self._execute_initialization()
        elif phase == "planning":
            return self._execute_planning()
        elif phase == "design_review":
            return self._execute_design_review()
        elif phase == "development":
            return self._execute_development()
        elif phase == "testing":
            return self._execute_testing()
        elif phase == "reflection":
            return self._execute_reflection()
        elif phase == "continuation":
            return self._execute_continuation()

        return {"error": f"Phase implementation missing: {phase}"}

    def _execute_initialization(self) -> Dict[str, Any]:
        """Execute the initialization phase."""
        self.log.info("Executing initialization phase")

        try:
            # Analyze project structure
            project_path = self.workflow_engine.workflow_state.get("project_path")
            if not project_path:
                self.log.error("Project path not set in workflow engine")
                return {"error": "Project not initialized. Call initialize_project first."}

            analysis_result = self.workflow_engine.code_analyzer.analyze_project(project_path)

            # Use the reflection LLM for deep understanding of the codebase
            reflection_llm = self.llm_extension.get_llm_for_task(
                TaskPurpose.REASONING,
                TaskComplexity.HARD
            )

            # Analyze the feature requirements
            feature_analysis = reflection_llm.query(
                prompt=f"""
                Analyze this feature requirement for implementation:
                Feature Name: {self.state['feature_name']}
                Description: {self.state['feature_description']}

                Identify:
                1. Key components needed
                2. Potential dependencies
                3. Potential challenges
                4. Existing code that may need modification
                5. New code that will need to be created
                """,
                context=self.workflow_engine.workflow_state.get("context", "")
            )

            # Record the results
            result = {
                "phase": "initialization",
                "project_analysis": {
                    "files_count": len(analysis_result.get("files", [])),
                    "modules_count": len(analysis_result.get("modules", {})),
                    "classes_count": len(analysis_result.get("classes", {})),
                    "functions_count": len(analysis_result.get("functions", {}))
                },
                "feature_analysis": feature_analysis,
                "next_phase": "planning"
            }

            # Transition to planning phase
            self.log.info("Initialization phase completed successfully, proceeding to planning")
            return result

        except Exception as e:
            self.log.error(f"Error in initialization phase: {str(e)}", exc_info=True)
            return {"error": f"Initialization failed: {str(e)}"}

    def _execute_planning(self) -> Dict[str, Any]:
        """Execute the planning phase."""
        self.log.info("Executing planning phase")

        try:
            # Use a planning-specialized LLM
            planning_llm = self.llm_extension.get_llm_for_task(
                TaskPurpose.PLANNING,
                TaskComplexity.HARD
            )

            # Generate development plan
            plan = planning_llm.query(
                prompt=f"""
                Create a detailed development plan for implementing the following feature:

                Feature Name: {self.state['feature_name']}
                Description: {self.state['feature_description']}

                Your plan should include:
                1. List of tasks in sequential order
                2. Dependencies between tasks
                3. Estimated complexity for each task (low, medium, high)
                4. Files that need to be created or modified
                5. Testing requirements

                Format your plan as a structured list with clear task descriptions.
                """,
                context=self.workflow_engine.workflow_state.get("context", "")
            )

            # Store the plan
            self.state["current_plan"] = plan
            self.state["plan_iterations"] += 1

            # Record the results
            result = {
                "phase": "planning",
                "development_plan": plan,
                "plan_iteration": self.state["plan_iterations"],
                "next_phase": "design_review"
            }

            self.log.info("Planning phase completed successfully, proceeding to design review")
            return result

        except Exception as e:
            self.log.error(f"Error in planning phase: {str(e)}", exc_info=True)
            return {"error": f"Planning failed: {str(e)}"}

    def _execute_design_review(self) -> Dict[str, Any]:
        """Execute the design review phase with multiple LLMs."""
        self.log.info("Executing design review phase")

        try:
            # Check if we have a plan to review
            if not self.state["current_plan"]:
                self.log.error("No plan available for design review")
                return {"error": "No plan available for design review. Run planning phase first."}

            # Use multiple specialized LLMs for different perspectives
            critics = {
                "architecture_critic": self.llm_extension.get_llm_for_task(
                    TaskPurpose.CODE_REVIEW,
                    TaskComplexity.HARD
                ),
                "security_expert": self.llm_extension.get_llm_for_task(
                    TaskPurpose.CODE_REVIEW,
                    TaskComplexity.HARD
                ),
                "performance_optimizer": self.llm_extension.get_llm_for_task(
                    TaskPurpose.CODE_REVIEW,
                    TaskComplexity.HARD
                ),
                "maintainability_guru": self.llm_extension.get_llm_for_task(
                    TaskPurpose.CODE_REVIEW,
                    TaskComplexity.NORMAL
                )
            }

            # Gather critiques from different perspectives
            critiques = {}
            for role, critic_llm in critics.items():
                self.log.debug(f"Getting critique from {role}")

                system_message = {
                    "architecture_critic": "You are an expert software architect focused on clean, modular design.",
                    "security_expert": "You are a security expert focused on secure coding practices and vulnerability prevention.",
                    "performance_optimizer": "You are a performance optimization expert focused on efficient algorithms and resource usage.",
                    "maintainability_guru": "You are a software maintainability expert focused on readability, scalability, and long-term code health."
                }.get(role, "You are an expert software reviewer.")

                critique = critic_llm.query(
                    prompt=f"""
                    Review the following development plan for the feature:

                    Feature Name: {self.state['feature_name']}
                    Description: {self.state['feature_description']}

                    Development Plan:
                    {self.state['current_plan']}

                    Critique this plan from your specialized perspective. 
                    Identify any issues, improvements, or oversights.

                    Provide your feedback in a structured format:
                    1. Overall assessment (1-10 rating)
                    2. Strengths
                    3. Concerns
                    4. Specific recommendations for improvement
                    """,
                    system_message=system_message
                )

                critiques[role] = critique

            # Store the critiques
            self.state["critique_history"].append(critiques)

            # Use a mediator LLM to synthesize the feedback
            mediator_llm = self.llm_extension.get_llm_for_task(
                TaskPurpose.REASONING,
                TaskComplexity.HARD
            )

            critiques_text = "\n\n".join([f"=== {role.upper()} ===\n{critique}"
                                          for role, critique in critiques.items()])

            synthesis = mediator_llm.query(
                prompt=f"""
                Synthesize the following critiques of a development plan:

                {critiques_text}

                Provide:
                1. Summary of key points from all critiques
                2. List of common concerns 
                3. Prioritized list of actionable changes needed
                4. Assessment of whether the plan needs major revision or minor tweaks
                5. Final recommendation (proceed with development, revise plan, or restart planning)
                """,
                system_message="You are a skilled mediator and synthesis expert who can balance different perspectives."
            )

            # Determine next steps based on synthesis
            if "major revision" in synthesis.lower() or "restart planning" in synthesis.lower():
                # Check if we've hit the iteration limit
                if self.state["plan_iterations"] >= self.state["iteration_limit"]:
                    next_phase = "development"  # Proceed anyway to avoid endless loop
                    self.log.warning(
                        f"Hit plan iteration limit ({self.state['iteration_limit']}). Proceeding to development despite critique.")
                else:
                    next_phase = "planning"  # Go back to planning
                    self.log.info("Returning to planning phase for plan revision")
            else:
                next_phase = "development"  # Proceed to development
                self.log.info("Plan approved with minor or no changes, proceeding to development")

            # Record the results
            result = {
                "phase": "design_review",
                "critiques": critiques,
                "synthesis": synthesis,
                "next_phase": next_phase
            }

            return result

        except Exception as e:
            self.log.error(f"Error in design review phase: {str(e)}", exc_info=True)
            return {"error": f"Design review failed: {str(e)}"}

    def _execute_development(self) -> Dict[str, Any]:
        """Execute the development phase."""
        self.log.info("Executing development phase")

        try:
            # Extract tasks from the plan
            # This is a simplified approach - in a real implementation,
            # you would parse the plan more robustly
            tasks = self._extract_tasks_from_plan(self.state["current_plan"])

            if not tasks:
                self.log.warning("No tasks extracted from plan")
                tasks = [{
                    "description": f"Implement {self.state['feature_name']}",
                    "complexity": "medium"
                }]

            # Process the first task in the list
            # In a full implementation, you might iterate through all tasks
            # or select the highest priority task
            current_task = tasks[0]
            self.log.info(f"Processing task: {current_task['description']}")

            # Use code generation LLM
            generation_complexity = {
                "low": TaskComplexity.NORMAL,
                "medium": TaskComplexity.NORMAL,
                "high": TaskComplexity.HARD
            }.get(current_task.get("complexity", "medium").lower(), TaskComplexity.NORMAL)

            code_generation_llm = self.llm_extension.get_llm_for_task(
                TaskPurpose.CODE_GENERATION,
                generation_complexity
            )

            # Determine if we need to create a new file or modify existing
            # This would be determined by analyzing the task and plan
            file_path = current_task.get("file_path")
            existing_code = None

            if file_path and os.path.exists(file_path):
                # Modify existing file
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_code = f.read()

            # Generate code
            code_result = code_generation_llm.get_code_suggestions(
                prompt=f"""
                Implement the following task as part of the {self.state['feature_name']} feature:

                Task Description: {current_task['description']}
                Feature Description: {self.state['feature_description']}

                {f'This code will be written to: {file_path}' if file_path else 'Determine the appropriate file for this code.'}
                """,
                existing_code=existing_code,
                context=self.workflow_engine.workflow_state.get("context", "")
            )

            # Get a second opinion from a different LLM
            review_llm = self.llm_extension.get_llm_for_task(
                TaskPurpose.CODE_REVIEW,
                TaskComplexity.NORMAL
            )

            review_result = review_llm.verify_code_changes(
                original_code=existing_code or "",
                new_code=code_result.get("code", ""),
                requirements=f"""
                The code should:
                1. Implement the task: {current_task['description']}
                2. Be consistent with the feature: {self.state['feature_name']}
                3. Follow good coding practices
                4. Be well-documented
                5. Include appropriate error handling
                """
            )

            # Store the results
            task_result = {
                "task": current_task,
                "code_suggestion": code_result,
                "code_review": review_result,
                "file_path": file_path,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self.state["task_history"].append(task_result)

            # If the code passed review, implement it
            implementation_result = None
            if review_result.get("meets_requirements") is True:
                self.log.info("Code passed review, implementing changes")

                if not file_path:
                    # Determine file path from code or description
                    file_path = self._determine_file_path(code_result.get("code", ""), current_task)

                # Implement the changes
                implementation_result = self.workflow_engine.implement_changes(
                    file_path=file_path,
                    new_code=code_result.get("code", "")
                )

                # Move to testing phase
                next_phase = "testing"
            else:
                # If code needs improvement, loop back for another iteration
                self.log.warning("Code did not pass review, needs improvement")

                # Check if we've hit the iteration limit for this task
                task_id = current_task.get("id", current_task.get("description", "unknown"))
                if task_id not in self.state["loop_detection"]:
                    self.state["loop_detection"][task_id] = 1
                else:
                    self.state["loop_detection"][task_id] += 1

                if self.state["loop_detection"].get(task_id, 0) >= 3:
                    # After 3 attempts, move on to testing anyway
                    self.log.warning(f"Hit iteration limit for task {task_id}, implementing anyway")

                    if not file_path:
                        file_path = self._determine_file_path(code_result.get("code", ""), current_task)

                    implementation_result = self.workflow_engine.implement_changes(
                        file_path=file_path,
                        new_code=code_result.get("code", "")
                    )

                    next_phase = "testing"
                else:
                    # Try development again with the review feedback
                    next_phase = "development"

            # Record the results
            result = {
                "phase": "development",
                "task_processed": current_task,
                "code_generated": code_result.get("code", ""),
                "review_result": review_result,
                "implementation_result": implementation_result,
                "next_phase": next_phase
            }

            return result

        except Exception as e:
            self.log.error(f"Error in development phase: {str(e)}", exc_info=True)
            return {"error": f"Development failed: {str(e)}"}

    def _execute_testing(self) -> Dict[str, Any]:
        """Execute the testing phase."""
        self.log.info("Executing testing phase")

        try:
            # Get the latest implemented task
            if not self.state["task_history"]:
                self.log.error("No tasks in history for testing")
                return {"error": "No tasks have been implemented yet. Run development phase first."}

            latest_task = self.state["task_history"][-1]
            file_path = latest_task.get("file_path")

            if not file_path or not os.path.exists(file_path):
                self.log.error(f"File path not valid for testing: {file_path}")
                return {"error": f"Invalid file path for testing: {file_path}"}

            # Use test generation LLM
            test_generation_llm = self.llm_extension.get_llm_for_task(
                TaskPurpose.CODE_GENERATION,
                TaskComplexity.NORMAL
            )

            # Generate test code
            test_result = test_generation_llm.get_code_suggestions(
                prompt=f"""
                Create unit tests for the following implemented code:

                File: {file_path}
                Feature: {self.state['feature_name']}
                Task: {latest_task.get('task', {}).get('description', 'Unknown task')}

                The tests should:
                1. Cover the main functionality
                2. Test edge cases
                3. Verify error handling
                4. Be comprehensive yet concise
                """,
                context=f"Code to test:\n```\n{latest_task.get('code_suggestion', {}).get('code', '')}\n```"
            )

            # Determine test file path
            test_file_path = self._determine_test_file_path(file_path)

            # Implement the test
            test_implementation_result = self.workflow_engine.implement_changes(
                file_path=test_file_path,
                new_code=test_result.get("code", "")
            )

            # Store the test
            self.state["tests_created"].append({
                "test_file": test_file_path,
                "target_file": file_path,
                "task": latest_task.get('task'),
                "timestamp": datetime.datetime.now().isoformat()
            })

            # Verify existing tests also cover this change
            test_verification_result = self.workflow_engine.verify_tests(file_path)

            # Run the tests
            test_run_result = self.workflow_engine.run_integration_tests([test_file_path])

            # Record the results
            result = {
                "phase": "testing",
                "test_code_generated": test_result.get("code", ""),
                "test_file_path": test_file_path,
                "test_implementation_result": test_implementation_result,
                "test_verification_result": test_verification_result,
                "test_run_result": test_run_result,
                "next_phase": "reflection"
            }

            return result

        except Exception as e:
            self.log.error(f"Error in testing phase: {str(e)}", exc_info=True)
            return {"error": f"Testing failed: {str(e)}"}

    def _execute_reflection(self) -> Dict[str, Any]:
        """Execute the reflection phase."""
        self.log.info("Executing reflection phase")

        try:
            # Get all task history
            if not self.state["task_history"]:
                self.log.error("No tasks in history for reflection")
                return {"error": "No tasks have been implemented yet. Run development phase first."}

            # Use a reflection-specialized LLM
            reflection_llm = self.llm_extension.get_llm_for_task(
                TaskPurpose.REASONING,
                TaskComplexity.HARD
            )

            # Create a summary of all work done so far
            task_summaries = []
            for i, task in enumerate(self.state["task_history"]):
                task_summaries.append(f"""
                Task {i + 1}: {task.get('task', {}).get('description', 'Unknown task')}
                File: {task.get('file_path', 'Unknown file')}
                Review Passed: {task.get('code_review', {}).get('meets_requirements', 'Unknown')}
                """)

            task_summary_text = "\n".join(task_summaries)

            # Generate reflection
            reflection = reflection_llm.query(
                prompt=f"""
                Reflect on the development of the feature "{self.state['feature_name']}" so far:

                Feature Description: {self.state['feature_description']}

                Summary of tasks completed:
                {task_summary_text}

                Tests created: {len(self.state['tests_created'])}

                Please provide:
                1. Overall assessment of implementation quality
                2. Areas that might need refactoring
                3. Potential technical debt introduced
                4. Suggestions for improvement
                5. Assessment of test coverage adequacy
                6. Recommendation for next steps
                """,
                system_message="You are a thoughtful software architect focused on long-term code health and quality."
            )

            # Record the results
            result = {
                "phase": "reflection",
                "reflection": reflection,
                "tasks_completed": len(self.state["task_history"]),
                "tests_created": len(self.state["tests_created"]),
                "next_phase": "continuation"
            }

            return result

        except Exception as e:
            self.log.error(f"Error in reflection phase: {str(e)}", exc_info=True)
            return {"error": f"Reflection failed: {str(e)}"}

    def _execute_continuation(self) -> Dict[str, Any]:
        """Execute the continuation phase."""
        self.log.info("Executing continuation phase")

        try:
            # Extract tasks from the plan
            tasks = self._extract_tasks_from_plan(self.state["current_plan"])

            # Filter out completed tasks
            completed_task_descriptions = [t.get('task', {}).get('description', '')
                                           for t in self.state["task_history"]]

            remaining_tasks = [t for t in tasks
                               if t.get('description', '') not in completed_task_descriptions]

            if not remaining_tasks:
                self.log.info("All tasks completed, feature development finished")
                return {
                    "phase": "continuation",
                    "status": "completed",
                    "message": "All tasks completed, feature development finished",
                    "feature_name": self.state["feature_name"],
                    "tasks_completed": len(self.state["task_history"]),
                    "tests_created": len(self.state["tests_created"])
                }

            # Use planning LLM to decide next steps
            planning_llm = self.llm_extension.get_llm_for_task(
                TaskPurpose.PLANNING,
                TaskComplexity.NORMAL
            )

            # Create a summary of remaining tasks
            remaining_task_summaries = []
            for i, task in enumerate(remaining_tasks):
                remaining_task_summaries.append(f"""
                Task {i + 1}: {task.get('description', 'Unknown task')}
                Complexity: {task.get('complexity', 'Unknown')}
                """)

            remaining_task_text = "\n".join(remaining_task_summaries)

            # Get recommendation for next step
            next_step = planning_llm.query(
                prompt=f"""
                Recommend the next development step for the feature "{self.state['feature_name']}":

                Feature Description: {self.state['feature_description']}

                Tasks completed: {len(self.state["task_history"])}
                Tests created: {len(self.state["tests_created"])}

                Remaining tasks:
                {remaining_task_text}

                Provide:
                1. Which task should be tackled next and why
                2. Any dependencies or prerequisites that should be considered
                3. Recommended approach for the next task
                """,
                system_message="You are a project manager focused on efficient task prioritization and execution."
            )

            # Record the results
            result = {
                "phase": "continuation",
                "status": "in_progress",
                "next_step_recommendation": next_step,
                "remaining_tasks": len(remaining_tasks),
                "next_phase": "development"  # Start the development cycle again
            }

            return result

        except Exception as e:
            self.log.error(f"Error in continuation phase: {str(e)}", exc_info=True)
            return {"error": f"Continuation failed: {str(e)}"}

    def _extract_tasks_from_plan(self, plan: str) -> List[Dict[str, Any]]:
        """
        Extract tasks from a plan text.

        This is a simplified implementation. In a real system,
        you would use a more sophisticated approach to parse the plan.

        Args:
            plan: The plan text

        Returns:
            List of task dictionaries
        """
        # Use a task extraction LLM
        extraction_llm = self.llm_extension.get_llm_for_task(
            TaskPurpose.REASONING,
            TaskComplexity.NORMAL
        )

        # Extract tasks using LLM
        extraction_result = extraction_llm.query(
            prompt=f"""
            Extract a structured list of tasks from the following development plan:

            {plan}

            For each task, provide:
            1. A clear task description
            2. Estimated complexity (low, medium, high)
            3. Dependencies (if any)
            4. Files to modify (if specified)

            Format your response as a JSON array of task objects.
            """,
            system_message="You are a task extraction specialist. Parse plans into structured task lists."
        )

        # Try to parse JSON from the response
        try:
            import re
            import json

            # Find JSON in the response
            json_match = re.search(r'\[[\s\S]*\]', extraction_result)
            if json_match:
                tasks = json.loads(json_match.group(0))
                return tasks
            else:
                # Fallback: simple line-based extraction
                tasks = []
                lines = plan.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('*') or
                                 (len(line) > 2 and line[0].isdigit() and line[1] == '.')):
                        task_text = line[2:].strip() if line[1] == '.' else line[1:].strip()
                        tasks.append({"description": task_text, "complexity": "medium"})

                return tasks

        except Exception as e:
            self.log.error(f"Error parsing tasks from plan: {str(e)}")
            return []

    def _determine_file_path(self, code: str, task: Dict[str, Any]) -> str:
        """
        Determine appropriate file path for code if not specified.

        Args:
            code: The code to analyze
            task: The task description

        Returns:
            Suggested file path
        """
        # Get project path
        project_path = self.workflow_engine.workflow_state.get("project_path", "")

        # Use a code-analysis LLM
        analysis_llm = self.llm_extension.get_llm_for_task(
            TaskPurpose.CODE_REVIEW,
            TaskComplexity.NORMAL
        )

        file_suggestion = analysis_llm.query(
            prompt=f"""
            Based on the following code and task, suggest an appropriate file path:

            Task: {task.get('description', 'Unknown task')}

            Code:
            ```
            {code[:1000]}  # First 1000 chars only to avoid token limits
            ```

            Suggest a specific file path within the project structure where this code should be placed.
            Respond with just the relative file path, nothing else.
            """,
            context=self.workflow_engine.workflow_state.get("context", "")
        )

        # Clean up the response to get just the file path
        file_path = file_suggestion.strip().strip('`').strip()

        # Make sure it's a valid path
        if not file_path.endswith('.py'):
            file_path += '.py'

        # Make it absolute if needed
        if not os.path.isabs(file_path) and project_path:
            file_path = os.path.join(project_path, file_path)

        return file_path

    def _determine_test_file_path(self, source_file_path: str) -> str:
        """
        Determine appropriate test file path for a source file.

        Args:
            source_file_path: The source file path

        Returns:
            Test file path
        """
        # Get project path
        project_path = self.workflow_engine.workflow_state.get("project_path", "")

        # Handle common test directory patterns
        if not project_path:
            # Without a project path, just prepend "test_" to the filename
            filename = os.path.basename(source_file_path)
            dir_path = os.path.dirname(source_file_path)
            return os.path.join(dir_path, f"test_{filename}")

        # Try to find a test directory
        tests_dir = None
        for test_dir_name in ["tests", "test", "unit_tests"]:
            test_dir = os.path.join(project_path, test_dir_name)
            if os.path.exists(test_dir) and os.path.isdir(test_dir):
                tests_dir = test_dir
                break

        if not tests_dir:
            # Create a tests directory if none exists
            tests_dir = os.path.join(project_path, "tests")
            os.makedirs(tests_dir, exist_ok=True)

        # Create test file path
        rel_path = os.path.relpath(source_file_path, project_path)
        rel_dir = os.path.dirname(rel_path)
        filename = os.path.basename(source_file_path)

        if rel_dir:
            # Maintain directory structure in tests directory
            test_dir = os.path.join(tests_dir, rel_dir)
            os.makedirs(test_dir, exist_ok=True)
            return os.path.join(test_dir, f"test_{filename}")
        else:
            # File is in project root
            return os.path.join(tests_dir, f"test_{filename}")

    def continue_workflow(self) -> Dict[str, Any]:
        """
        Continue execution of the workflow from the current phase.

        Returns:
            Result of the next phase execution
        """
        current_phase = self.state["current_phase"]

        if not current_phase or current_phase not in self.phases:
            self.log.error(f"Invalid current phase: {current_phase}")
            return {"error": f"Invalid current phase: {current_phase}"}

        # Get the next phase from the current phase's result
        phase_result = getattr(self, f"_execute_{current_phase}")()
        next_phase = phase_result.get("next_phase")

        if not next_phase:
            self.log.warning(f"No next phase specified in result of {current_phase}")
            # Try to determine next phase
            current_idx = self.phases.index(current_phase)
            if current_idx < len(self.phases) - 1:
                next_phase = self.phases[current_idx + 1]
            else:
                next_phase = "development"  # Default to cycling back to development

        # Execute the next phase
        return self._execute_phase(next_phase)

    def execute_full_workflow(self) -> Dict[str, Any]:
        """
        Execute the full workflow from start to finish.

        Returns:
            Consolidated results from all phases
        """
        self.log.info(f"Executing full workflow for feature: {self.state['feature_name']}")

        results = {}
        current_phase = "initialization"

        while current_phase:
            # Execute current phase
            phase_result = self._execute_phase(current_phase)
            results[current_phase] = phase_result

            # Check for errors
            if "error" in phase_result:
                self.log.error(f"Workflow stopped due to error in {current_phase}: {phase_result['error']}")
                break

            # Get next phase
            next_phase = phase_result.get("next_phase")

            # Check for completion
            if current_phase == "continuation" and phase_result.get("status") == "completed":
                self.log.info("Workflow completed successfully")
                break

            # Prevent infinite loops
            if next_phase == current_phase:
                self.log.warning(f"Potential infinite loop detected (staying in {current_phase})")
                # Force progression
                current_idx = self.phases.index(current_phase)
                if current_idx < len(self.phases) - 1:
                    next_phase = self.phases[current_idx + 1]
                else:
                    break

            current_phase = next_phase

        return {
            "feature_name": self.state["feature_name"],
            "feature_description": self.state["feature_description"],
            "tasks_completed": len(self.state["task_history"]),
            "tests_created": len(self.state["tests_created"]),
            "phase_results": results,
            "status": "completed" if "continuation" in results and results["continuation"].get(
                "status") == "completed" else "partial"
        }

    def save_workflow_state(self, output_path: str) -> str:
        """
        Save the current workflow state to a file.

        Args:
            output_path: Path to save the state

        Returns:
            Path to the saved file
        """
        self.log.info(f"Saving collaborative workflow state to: {output_path}")

        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Save state to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)

            return output_path

        except Exception as e:
            self.log.error(f"Error saving workflow state: {str(e)}")
            return f"Error: {str(e)}"

    def load_workflow_state(self, input_path: str) -> Dict[str, Any]:
        """
        Load workflow state from a file.

        Args:
            input_path: Path to the state file

        Returns:
            Loaded workflow state
        """
        self.log.info(f"Loading collaborative workflow state from: {input_path}")

        try:
            # Load state from file
            with open(input_path, 'r', encoding='utf-8') as f:
                self.state = json.load(f)

            return self.state

        except Exception as e:
            self.log.error(f"Error loading workflow state: {str(e)}")
            return {"error": f"Failed to load workflow state: {str(e)}"}