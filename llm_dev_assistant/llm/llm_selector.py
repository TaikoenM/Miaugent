# llm_dev_assistant/llm/llm_selector.py
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type
import os
import logging
import json

from .llm_interface import LLMInterface
from ..logging.logger import logger


class TaskPurpose(Enum):
    """Enum representing different purposes for using an LLM."""
    VISION = auto()  # Visual content analysis, OCR, image understanding
    CODE_GENERATION = auto()  # Generate new code from scratch
    CODE_MODIFICATION = auto()  # Modify existing code
    CODE_REVIEW = auto()  # Review and analyze code quality, security, etc.
    CODE_DEBUGGING = auto()  # Debug and fix issues in code
    TEXT_GENERATION = auto()  # Generate natural language text (articles, stories, etc.)
    TEXT_SUMMARIZATION = auto()  # Summarize longer text into concise format
    TRANSLATION = auto()  # Translate between languages
    SENTIMENT_ANALYSIS = auto()  # Analyze sentiment in text
    DATA_ANALYSIS = auto()  # Analyze and interpret data
    PLANNING = auto()  # Plan operations, create roadmaps, strategy
    PROMPT_ENGINEERING = auto()  # Create or optimize prompts for other LLMs
    REASONING = auto()  # Complex reasoning, problem-solving
    QA = auto()  # Question answering
    CREATIVE = auto()  # Creative tasks like storytelling, poetry, etc.
    RAG = auto()  # Retrieval Augmented Generation
    TECHNICAL_WRITING = auto()  # Technical documentation, API docs, etc.
    CONVERSATION = auto()  # Conversational purposes or chat
    DOMAIN_EXPERT = auto()  # Domain-specific expertise (legal, medical, etc.)
    MULTI_MODAL = auto()  # Tasks involving multiple modalities (text, images, audio)


class TaskComplexity(Enum):
    """Enum representing different complexity levels for tasks."""
    LOW = auto()  # Simple, straightforward tasks with minimal context
    NORMAL = auto()  # Moderate complexity, average context requirements
    HARD = auto()  # Complex tasks requiring more context and reasoning
    NIGHTMARE = auto()  # Extremely complex tasks requiring maximum context and capabilities


class ModelType(Enum):
    """Enum representing different types of LLM models."""
    VISION = auto()  # Vision-capable models
    ONLINE = auto()  # Cloud-based powerful models
    LOCAL = auto()  # Locally deployed models
    LMSTUDIO = auto()  # Models deployed via LM Studio


class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create_vision_llm(model_name: str, **kwargs) -> LLMInterface:
        """
        Create a vision-capable LLM.

        Args:
            model_name: Name of the model
            **kwargs: Additional arguments for the specific implementation

        Returns:
            Vision-capable LLM instance
        """
        # Import here to avoid circular imports
        from .vision_llm_adapter import VisionLLMAdapter
        return VisionLLMAdapter(model=model_name, **kwargs)

    @staticmethod
    def create_online_llm(model_name: str, **kwargs) -> LLMInterface:
        """
        Create an online LLM.

        Args:
            model_name: Name of the model
            **kwargs: Additional arguments for the specific implementation

        Returns:
            Online LLM instance
        """
        # Import here to avoid circular imports
        from .openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model=model_name, **kwargs)

    @staticmethod
    def create_local_llm(model_path: str, **kwargs) -> LLMInterface:
        """
        Create a local LLM.

        Args:
            model_path: Path to the model
            **kwargs: Additional arguments for the specific implementation

        Returns:
            Local LLM instance
        """
        # Import here to avoid circular imports
        from .local_llm_adapter import LocalLLMAdapter
        return LocalLLMAdapter(model_path=model_path, **kwargs)

    @staticmethod
    def create_lmstudio_llm(model_name: str, **kwargs) -> LLMInterface:
        """
        Create an LM Studio LLM.

        Args:
            model_name: Name of the model
            **kwargs: Additional arguments for the specific implementation

        Returns:
            LM Studio LLM instance
        """
        # Import here to avoid circular imports
        from .lmstudio_adapter import LMStudioAdapter
        return LMStudioAdapter(model_name=model_name, **kwargs)


class LLMSelector:
    """Class for selecting the appropriate LLM based on task and complexity."""

    def __init__(self, factories: Optional[Dict[ModelType, Callable]] = None):
        """
        Initialize the LLM selector.

        Args:
            factories: Optional dictionary mapping model types to factory functions
        """
        self.log = logger.get_logger("llm_selector")
        self.log.info("Initializing LLM selector")

        # Set up factory functions
        self.factories = factories or {
            ModelType.VISION: LLMFactory.create_vision_llm,
            ModelType.ONLINE: LLMFactory.create_online_llm,
            ModelType.LOCAL: LLMFactory.create_local_llm,
            ModelType.LMSTUDIO: LLMFactory.create_lmstudio_llm
        }

        # Dictionary to store LLM instances
        self.llm_instances = {}

    def select_llm(self, task_purpose: TaskPurpose, task_complexity: TaskComplexity,
                   vision_model_name: str = "gpt-4-vision-preview",
                   online_model_name: str = "gpt-4",
                   local_model_path: str = None,
                   lm_studio_model: str = None) -> Tuple[LLMInterface, str]:
        """
        Select the appropriate LLM based on task purpose and complexity.

        Args:
            task_purpose: The purpose of the task
            task_complexity: The complexity of the task
            vision_model_name: Name of the vision model to use
            online_model_name: Name of the online model to use
            local_model_path: Path to the local model
            lm_studio_model: Name of the LM Studio model

        Returns:
            Tuple of (LLM instance, model_type_description)
        """
        self.log.info(f"Selecting LLM for task purpose: {task_purpose.name}, complexity: {task_complexity.name}")

        model_type, model_args = self._determine_model_type(
            task_purpose,
            task_complexity,
            vision_model_name,
            online_model_name,
            local_model_path,
            lm_studio_model
        )

        # Get or create the LLM instance
        llm = self._get_or_create_llm(model_type, model_args)

        return llm, model_type.name

    def _determine_model_type(self,
                              task_purpose: TaskPurpose,
                              task_complexity: TaskComplexity,
                              vision_model_name: str,
                              online_model_name: str,
                              local_model_path: str,
                              lm_studio_model: str) -> Tuple[ModelType, Dict[str, Any]]:
        """
        Determine the appropriate model type and arguments based on task and complexity.

        Args:
            task_purpose: The purpose of the task
            task_complexity: The complexity of the task
            vision_model_name: Name of the vision model
            online_model_name: Name of the online model
            local_model_path: Path to the local model
            lm_studio_model: Name of the LM Studio model

        Returns:
            Tuple of (model_type, model_args)
        """
        if task_purpose == TaskPurpose.VISION or task_purpose == TaskPurpose.MULTI_MODAL:
            # Vision tasks require a vision-capable model
            self.log.info(f"Task requires vision capabilities, selecting vision LLM: {vision_model_name}")
            return ModelType.VISION, {"model_name": vision_model_name}

        elif task_complexity in [TaskComplexity.HARD, TaskComplexity.NIGHTMARE]:
            # Hard or nightmare complexity tasks require more powerful online models
            self.log.info(f"Task complexity requires online model, selecting: {online_model_name}")
            return ModelType.ONLINE, {"model_name": online_model_name}

        else:
            # For normal or low complexity tasks, use local model or LM Studio
            if lm_studio_model:
                self.log.info(f"Using LM Studio model: {lm_studio_model}")
                return ModelType.LMSTUDIO, {"model_name": lm_studio_model}

            elif local_model_path:
                self.log.info(f"Using local model: {local_model_path}")
                return ModelType.LOCAL, {"model_path": local_model_path}

            else:
                # Fallback to online model if no local or LM Studio model specified
                self.log.info(f"No local or LM Studio model specified, using online model: {online_model_name}")
                return ModelType.ONLINE, {"model_name": online_model_name}

    def _get_or_create_llm(self, model_type: ModelType, model_args: Dict[str, Any]) -> LLMInterface:
        """
        Get an existing LLM instance or create a new one.

        Args:
            model_type: The type of model to use
            model_args: Arguments for creating the model

        Returns:
            LLM instance
        """
        # Create a unique key for the model instance
        key_parts = [model_type.name]
        for k, v in sorted(model_args.items()):
            key_parts.append(f"{k}={v}")
        model_key = ":".join(key_parts)

        # Return existing instance if available
        if model_key in self.llm_instances:
            self.log.debug(f"Using existing LLM instance: {model_key}")
            return self.llm_instances[model_key]

        # Create new instance
        self.log.debug(f"Creating new LLM instance: {model_key}")

        if model_type not in self.factories:
            self.log.error(f"No factory available for model type: {model_type.name}")
            raise ValueError(f"No factory available for model type: {model_type.name}")

        # Create the instance using the appropriate factory
        factory = self.factories[model_type]

        # Prepare arguments based on the model type
        if model_type == ModelType.VISION:
            llm = factory(model_args.get("model_name"))
        elif model_type == ModelType.ONLINE:
            llm = factory(model_args.get("model_name"))
        elif model_type == ModelType.LOCAL:
            llm = factory(model_args.get("model_path"))
        elif model_type == ModelType.LMSTUDIO:
            llm = factory(model_args.get("model_name"))
        else:
            self.log.error(f"Unknown model type: {model_type.name}")
            raise ValueError(f"Unknown model type: {model_type.name}")

        # Cache the instance
        self.llm_instances[model_key] = llm

        return llm

    def clear_instance(self, model_key: str) -> bool:
        """
        Clear a specific LLM instance from the cache.

        Args:
            model_key: The model key to clear

        Returns:
            True if the instance was cleared, False otherwise
        """
        if model_key in self.llm_instances:
            self.log.info(f"Clearing LLM instance: {model_key}")

            # If it's an LM Studio model, unmount it
            if model_key.startswith(ModelType.LMSTUDIO.name):
                try:
                    lm_studio_adapter = self.llm_instances[model_key]
                    if hasattr(lm_studio_adapter, "unmount_model"):
                        lm_studio_adapter.unmount_model()
                except Exception as e:
                    self.log.warning(f"Error unmounting LM Studio model: {str(e)}")

            # Remove the instance
            del self.llm_instances[model_key]
            return True

        return False

    def clear_all_instances(self) -> None:
        """Clear all LLM instances from the cache."""
        self.log.info("Clearing all LLM instances")

        # Unmount LM Studio models
        for model_key in list(self.llm_instances.keys()):
            if model_key.startswith(ModelType.LMSTUDIO.name):
                try:
                    lm_studio_adapter = self.llm_instances[model_key]
                    if hasattr(lm_studio_adapter, "unmount_model"):
                        lm_studio_adapter.unmount_model()
                except Exception as e:
                    self.log.warning(f"Error unmounting LM Studio model {model_key}: {str(e)}")

        # Clear the dictionary
        self.llm_instances.clear()


def choose_llm(task_purpose: Union[TaskPurpose, str],
               task_complexity: Union[TaskComplexity, str],
               vision_model: str = "gpt-4-vision-preview",
               online_model: str = "gpt-4",
               local_model: str = None,
               lm_studio_model: str = None) -> LLMInterface:
    """
    Helper function to choose an appropriate LLM based on task purpose and complexity.

    Args:
        task_purpose: The purpose of the task (string or TaskPurpose enum)
        task_complexity: The complexity of the task (string or TaskComplexity enum)
        vision_model: Name of the vision model to use
        online_model: Name of the online model to use
        local_model: Path to the local model
        lm_studio_model: Name of the LM Studio model

    Returns:
        An appropriate LLM interface instance
    """
    log = logger.get_logger("choose_llm")
    log.info(f"Choosing LLM for task: {task_purpose}, complexity: {task_complexity}")

    # Convert string to enum if needed
    if isinstance(task_purpose, str):
        try:
            task_purpose = TaskPurpose[task_purpose.upper()]
        except KeyError:
            log.warning(f"Unknown task purpose: {task_purpose}, defaulting to CODE_GENERATION")
            task_purpose = TaskPurpose.CODE_GENERATION

    if isinstance(task_complexity, str):
        try:
            task_complexity = TaskComplexity[task_complexity.upper()]
        except KeyError:
            log.warning(f"Unknown task complexity: {task_complexity}, defaulting to NORMAL")
            task_complexity = TaskComplexity.NORMAL

    # Create a selector and choose the appropriate LLM
    selector = LLMSelector()
    llm, model_type = selector.select_llm(
        task_purpose=task_purpose,
        task_complexity=task_complexity,
        vision_model_name=vision_model,
        online_model_name=online_model,
        local_model_path=local_model,
        lm_studio_model=lm_studio_model
    )

    log.info(f"Selected {model_type} for the task")
    return llm