# llm_dev_assistant/log_system/logger.py
import os
import sys
import logging
import datetime
from typing import Optional, Dict, Any


class Logger:
    """Advanced log_system system with dual log files and structured directory organization."""

    def __init__(self):
        """Initialize the logger instance."""
        self.loggers = {}
        self.log_dir = None
        self.debug_log_path = None
        self.info_log_path = None

    def setup(self, app_name: str = "llm_dev_assistant", log_dir: str = "logs") -> Dict[str, str]:
        """
        Set up the log_system system with date-based directories and timestamped log files.

        Args:
            app_name: Name of the application for logger naming
            log_dir: Directory to store log files

        Returns:
            Dictionary with paths to created log files
        """
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create date-based subdirectory
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        date_dir = os.path.join(log_dir, today)
        if not os.path.exists(date_dir):
            os.makedirs(date_dir)

        # Generate timestamp for log filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define log file paths
        self.log_dir = date_dir
        self.debug_log_path = os.path.join(date_dir, f"{timestamp}_debug.log")
        self.info_log_path = os.path.join(date_dir, f"{timestamp}_info.log")

        # Set up the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create and configure debug file handler (includes all levels)
        debug_handler = logging.FileHandler(self.debug_log_path)
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        debug_handler.setFormatter(debug_formatter)
        root_logger.addHandler(debug_handler)

        # Create and configure info file handler (includes INFO and above)
        info_handler = logging.FileHandler(self.info_log_path)
        info_handler.setLevel(logging.INFO)
        info_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        info_handler.setFormatter(info_formatter)
        root_logger.addHandler(info_handler)

        # Create and configure console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Store the application-specific logger
        self.loggers[app_name] = logging.getLogger(app_name)

        # Log setup completion
        self.loggers[app_name].info(f"Logging system initialized")
        self.loggers[app_name].debug(f"Debug log: {self.debug_log_path}")
        self.loggers[app_name].debug(f"Info log: {self.info_log_path}")

        return {
            "debug_log": self.debug_log_path,
            "info_log": self.info_log_path
        }

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a named logger.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]

    def log_llm_prompt(self, model: str, prompt: str, context: Optional[str] = None,
                       system_message: Optional[str] = None):
        """
        Log an LLM prompt with detailed information.

        Args:
            model: LLM model name
            prompt: The prompt sent to the LLM
            context: Optional context provided with the prompt
            system_message: Optional system message
        """
        logger = self.get_logger("llm.prompts")

        logger.debug(f"===== LLM PROMPT DETAILS - {model} =====")
        if system_message:
            logger.debug(f"SYSTEM MESSAGE:\n{system_message}")
        if context:
            logger.debug(f"CONTEXT:\n{context}")
        logger.debug(f"PROMPT:\n{prompt}")
        logger.debug("=====================================")

    def log_llm_response(self, model: str, response: str):
        """
        Log an LLM response.

        Args:
            model: LLM model name
            response: The response from the LLM
        """
        logger = self.get_logger("llm.responses")

        logger.debug(f"===== LLM RESPONSE - {model} =====")
        logger.debug(f"RESPONSE:\n{response}")
        logger.debug("=====================================")

    def log_function_call(self, func_name: str, args: Dict[str, Any] = None,
                          result: Any = None, error: Exception = None):
        """
        Log a function call with arguments and result.

        Args:
            func_name: Function name
            args: Function arguments
            result: Function result (if successful)
            error: Exception (if failed)
        """
        logger = self.get_logger("function.calls")

        if args:
            args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
            logger.debug(f"CALL: {func_name}({args_str})")
        else:
            logger.debug(f"CALL: {func_name}()")

        if error:
            logger.error(f"ERROR: {func_name} - {str(error)}")
        elif result:
            if isinstance(result, (dict, list)) and len(str(result)) > 500:
                logger.debug(f"RESULT: {func_name} - [Complex data structure]")
            else:
                logger.debug(f"RESULT: {func_name} - {result}")


# Singleton instance
logger = Logger()