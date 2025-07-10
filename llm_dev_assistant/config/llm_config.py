# llm_dev_assistant/config/llm_config.py
from typing import Dict, Any, Optional
import os
import json

from ..log_system.logger import logger
from ..llm.llm_selector import TaskPurpose, TaskComplexity, ModelType


class LLMConfig:
    """Configuration for the LLM system."""

    DEFAULT_CONFIG = {
        "vision_model": "gpt-4-vision-preview",
        "online_model": "gpt-4",
        "local_model": None,
        "lm_studio_model": None,
        "default_complexity": "NORMAL",
        "task_complexity_mapping": {
            "VISION": "NORMAL",
            "CODE_GENERATION": "NORMAL",
            "CODE_MODIFICATION": "NORMAL",
            "CODE_REVIEW": "NORMAL",
            "CODE_DEBUGGING": "HARD",
            "TEXT_GENERATION": "NORMAL",
            "TEXT_SUMMARIZATION": "NORMAL",
            "TRANSLATION": "NORMAL",
            "SENTIMENT_ANALYSIS": "LOW",
            "DATA_ANALYSIS": "HARD",
            "PLANNING": "HARD",
            "PROMPT_ENGINEERING": "NORMAL",
            "REASONING": "HARD",
            "QA": "NORMAL",
            "CREATIVE": "NORMAL",
            "RAG": "HARD",
            "TECHNICAL_WRITING": "NORMAL",
            "CONVERSATION": "LOW",
            "DOMAIN_EXPERT": "HARD",
            "MULTI_MODAL": "NORMAL"
        },
        "api_keys": {
            "openai": None
        },
        "lm_studio": {
            "host": "localhost",
            "port": 1234,
            "default_context_length": 4096
        },
        "models_config": {
            "max_token_lengths": {
                "gpt-4": 8192,
                "gpt-4-vision-preview": 128000,
                "gpt-3.5-turbo": 4096
            }
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LLM configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.log = logger.get_logger("llm_config")
        self.log.info("Initializing LLM configuration")

        # Start with default configuration
        self.config = self.DEFAULT_CONFIG.copy()

        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

        # Override with environment variables
        self._load_from_env()

        self.log.debug(
            f"Configuration initialized with vision_model={self.config['vision_model']}, online_model={self.config['online_model']}")

    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from a file.

        Args:
            config_path: Path to the configuration file

        Returns:
            True if successful, False otherwise
        """
        self.log.info(f"Loading configuration from {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            # Update configuration
            self.config.update(loaded_config)
            self.log.info(f"Configuration loaded successfully from {config_path}")
            return True

        except Exception as e:
            self.log.error(f"Error loading configuration: {str(e)}")
            return False

    def save_config(self, config_path: str) -> bool:
        """
        Save configuration to a file.

        Args:
            config_path: Path to save the configuration

        Returns:
            True if successful, False otherwise
        """
        self.log.info(f"Saving configuration to {config_path}")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)

            self.log.info(f"Configuration saved successfully to {config_path}")
            return True

        except Exception as e:
            self.log.error(f"Error saving configuration: {str(e)}")
            return False

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        self.log.debug("Loading configuration from environment variables")

        # API keys
        if os.environ.get("OPENAI_API_KEY"):
            self.config["api_keys"]["openai"] = os.environ.get("OPENAI_API_KEY")

        # Models
        if os.environ.get("LLM_VISION_MODEL"):
            self.config["vision_model"] = os.environ.get("LLM_VISION_MODEL")

        if os.environ.get("LLM_ONLINE_MODEL"):
            self.config["online_model"] = os.environ.get("LLM_ONLINE_MODEL")

        if os.environ.get("LLM_LOCAL_MODEL"):
            self.config["local_model"] = os.environ.get("LLM_LOCAL_MODEL")

        if os.environ.get("LLM_LMSTUDIO_MODEL"):
            self.config["lm_studio_model"] = os.environ.get("LLM_LMSTUDIO_MODEL")

        # LM Studio configuration
        if os.environ.get("LMSTUDIO_HOST"):
            self.config["lm_studio"]["host"] = os.environ.get("LMSTUDIO_HOST")

        if os.environ.get("LMSTUDIO_PORT"):
            try:
                self.config["lm_studio"]["port"] = int(os.environ.get("LMSTUDIO_PORT"))
            except (ValueError, TypeError):
                pass

    def get_task_complexity(self, task_purpose: TaskPurpose) -> TaskComplexity:
        """
        Get the default complexity for a task purpose.

        Args:
            task_purpose: Task purpose

        Returns:
            Task complexity
        """
        task_name = task_purpose.name
        complexity_name = self.config["task_complexity_mapping"].get(task_name, self.config["default_complexity"])

        try:
            return TaskComplexity[complexity_name]
        except KeyError:
            self.log.warning(f"Unknown complexity: {complexity_name}, using NORMAL")
            return TaskComplexity.NORMAL

    def get_preferred_model_for_task(self, task_purpose: TaskPurpose,
                                     task_complexity: Optional[TaskComplexity] = None) -> Dict[str, Any]:
        """
        Get the preferred model configuration for a task.

        Args:
            task_purpose: Task purpose
            task_complexity: Optional task complexity (if None, uses default for the task)

        Returns:
            Dictionary with model configuration
        """
        # Get complexity if not provided
        if task_complexity is None:
            task_complexity = self.get_task_complexity(task_purpose)

        # Determine model type based on task and complexity
        if task_purpose in [TaskPurpose.VISION, TaskPurpose.MULTI_MODAL]:
            model_type = ModelType.VISION
            model_name = self.config["vision_model"]
        elif task_complexity in [TaskComplexity.HARD, TaskComplexity.NIGHTMARE]:
            model_type = ModelType.ONLINE
            model_name = self.config["online_model"]
        elif self.config["lm_studio_model"]:
            model_type = ModelType.LMSTUDIO
            model_name = self.config["lm_studio_model"]
        elif self.config["local_model"]:
            model_type = ModelType.LOCAL
            model_name = self.config["local_model"]
        else:
            model_type = ModelType.ONLINE
            model_name = self.config["online_model"]

        # Return model configuration
        return {
            "model_type": model_type,
            "model_name": model_name,
            "task_purpose": task_purpose,
            "task_complexity": task_complexity
        }

    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a service.

        Args:
            service: Service name

        Returns:
            API key if available, None otherwise
        """
        return self.config["api_keys"].get(service)

    def set_api_key(self, service: str, api_key: str) -> None:
        """
        Set API key for a service.

        Args:
            service: Service name
            api_key: API key
        """
        self.config["api_keys"][service] = api_key

    def get_model_max_tokens(self, model_name: str) -> int:
        """
        Get maximum token length for a model.

        Args:
            model_name: Model name

        Returns:
            Maximum token length
        """
        return self.config["models_config"]["max_token_lengths"].get(model_name, 4096)

    def get_lm_studio_config(self) -> Dict[str, Any]:
        """
        Get LM Studio configuration.

        Returns:
            LM Studio configuration
        """
        return self.config["lm_studio"].copy()


# Singleton instance
llm_config = LLMConfig()