# llm_dev_assistant/workflow/workflow_engine_llm_extension.py
from typing import Dict, List, Any, Optional, Union

from ..log_system.logger import logger
from ..llm.llm_selector import TaskPurpose, TaskComplexity, choose_llm, LLMInterface
from ..llm.llm_selector import LLMSelector
from ..config.llm_config import llm_config


class WorkflowEngineLLMExtension:
    """Extension for WorkflowEngine that adds advanced LLM selection capabilities."""

    def __init__(self, workflow_engine):
        """
        Initialize the LLM extension.

        Args:
            workflow_engine: The WorkflowEngine instance to extend
        """
        self.log = logger.get_logger("workflow_engine_llm_extension")
        self.log.info("Initializing WorkflowEngine LLM extension")

        self.workflow_engine = workflow_engine
        self.llm_selector = LLMSelector()

        # Store specialized LLM instances
        self.specialized_llms = {}

        self.log.debug("WorkflowEngine LLM extension initialized")

    def get_llm_for_task(self, task_purpose: Union[TaskPurpose, str],
                         task_complexity: Optional[Union[TaskComplexity, str]] = None) -> LLMInterface:
        """
        Get an appropriate LLM for a specific task.

        Args:
            task_purpose: The purpose of the task
            task_complexity: Optional task complexity (if None, uses default for the task)

        Returns:
            LLM instance for the task
        """
        self.log.info(f"Getting LLM for task: {task_purpose}")

        # Convert string to enum if needed
        if isinstance(task_purpose, str):
            try:
                task_purpose = TaskPurpose[task_purpose.upper()]
            except KeyError:
                self.log.warning(f"Unknown task purpose: {task_purpose}, defaulting to CODE_GENERATION")
                task_purpose = TaskPurpose.CODE_GENERATION

        # Get complexity if not provided
        if task_complexity is None:
            task_complexity = llm_config.get_task_complexity(task_purpose)
        elif isinstance(task_complexity, str):
            try:
                task_complexity = TaskComplexity[task_complexity.upper()]
            except KeyError:
                self.log.warning(f"Unknown task complexity: {task_complexity}, using default for task")
                task_complexity = llm_config.get_task_complexity(task_purpose)

        # Get model configuration
        model_config = llm_config.get_preferred_model_for_task(task_purpose, task_complexity)

        # If we already have a specialized LLM for this task, return it
        task_key = f"{task_purpose.name}:{task_complexity.name}"
        if task_key in self.specialized_llms:
            self.log.debug(f"Using existing specialized LLM for {task_key}")
            return self.specialized_llms[task_key]

        # Otherwise, select an appropriate LLM
        llm, model_type = self.llm_selector.select_llm(
            task_purpose=task_purpose,
            task_complexity=task_complexity,
            vision_model_name=llm_config.config["vision_model"],
            online_model_name=llm_config.config["online_model"],
            local_model_path=llm_config.config["local_model"],
            lm_studio_model=llm_config.config["lm_studio_model"]
        )

        # Cache the LLM for future use
        self.specialized_llms[task_key] = llm

        self.log.info(f"Selected {model_type} LLM for {task_key}")
        return llm

    def clear_llm_cache(self) -> None:
        """Clear all cached LLM instances."""
        self.log.info("Clearing LLM cache")
        self.specialized_llms.clear()
        self.llm_selector.clear_all_instances()

    def set_default_llm(self, llm: LLMInterface) -> None:
        """
        Set the default LLM for the workflow engine.

        Args:
            llm: LLM instance to use as default
        """
        self.log.info(f"Setting default LLM: {type(llm).__name__}")
        self.workflow_engine.llm = llm

    def generate_code(self,
                      prompt: str,
                      existing_code: Optional[str] = None,
                      context: Optional[str] = None,
                      complexity: Union[TaskComplexity, str] = TaskComplexity.NORMAL) -> Dict[str, Any]:
        """
        Generate code using the most appropriate LLM for the task.

        Args:
            prompt: Description of the code to generate
            existing_code: Optional existing code to modify
            context: Optional context about the project
            complexity: Complexity of the task

        Returns:
            Dictionary with generated code and explanations
        """
        self.log.info(f"Generating code with complexity: {complexity}")

        # Get appropriate LLM for code generation
        llm = self.get_llm_for_task(TaskPurpose.CODE_GENERATION, complexity)

        # Generate code
        self.log.debug(f"Sending code generation prompt to LLM: {prompt[:100]}...")
        code_suggestions = llm.get_code_suggestions(prompt, existing_code, context)

        return code_suggestions

    def review_code(self,
                    original_code: str,
                    new_code: str,
                    requirements: str) -> Dict[str, Any]:
        """
        Review code changes using the most appropriate LLM for the task.

        Args:
            original_code: Original code
            new_code: New code
            requirements: Requirements the changes should meet

        Returns:
            Dictionary with review results
        """
        self.log.info("Reviewing code changes")

        # Get appropriate LLM for code review
        llm = self.get_llm_for_task(TaskPurpose.CODE_REVIEW)

        # Review code
        self.log.debug("Sending code review request to LLM")
        review_results = llm.verify_code_changes(original_code, new_code, requirements)

        return review_results

    def create_development_plan(self,
                                current_state: Dict[str, Any],
                                project_goals: List[str]) -> List[Dict[str, Any]]:
        """
        Create a development plan using the most appropriate LLM for the task.

        Args:
            current_state: Current project state
            project_goals: Project goals

        Returns:
            List of development steps
        """
        self.log.info("Creating development plan")

        # Get appropriate LLM for planning
        llm = self.get_llm_for_task(TaskPurpose.PLANNING, TaskComplexity.HARD)

        # Create plan
        self.log.debug(f"Sending planning request to LLM with {len(project_goals)} goals")
        plan = llm.plan_next_steps(current_state, project_goals)

        return plan

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze an image using a vision-capable LLM.

        Args:
            image_path: Path to the image
            prompt: Description of what to analyze

        Returns:
            Analysis results
        """
        self.log.info(f"Analyzing image: {image_path}")

        # Get vision LLM
        llm = self.get_llm_for_task(TaskPurpose.VISION)

        # Make sure it's a vision LLM
        if not hasattr(llm, "analyze_image"):
            self.log.error("Selected LLM does not support image analysis")
            return "Error: Selected LLM does not support image analysis"

        # Analyze image
        self.log.debug(f"Sending image analysis prompt to LLM: {prompt}")
        result = llm.analyze_image(image_path, prompt)

        return result