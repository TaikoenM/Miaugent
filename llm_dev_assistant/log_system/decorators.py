# llm_dev_assistant/log_system/decorators.py
import functools
import inspect
import logging
import time
from typing import Any, Callable
from .logger import logger


def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function calls with arguments and results.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        module_name = func.__module__

        # Get logger for the module
        log = logger.get_logger(module_name)

        # Convert arguments to a readable format
        args_str = []
        for i, arg in enumerate(args):
            # Get argument name if available
            param_names = list(inspect.signature(func).parameters.keys())
            if i < len(param_names):
                arg_name = param_names[i]
                args_str.append(f"{arg_name}={repr(arg)}")
            else:
                args_str.append(repr(arg))

        for name, value in kwargs.items():
            args_str.append(f"{name}={repr(value)}")

        args_str = ", ".join(args_str)

        # Log function call
        log.debug(f"CALL: {func_name}({args_str})")

        # Call the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            # Log result (if not too large)
            if result is not None:
                result_str = repr(result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "... [truncated]"
                log.debug(f"RETURN [{elapsed:.3f}s]: {func_name} -> {result_str}")
            else:
                log.debug(f"RETURN [{elapsed:.3f}s]: {func_name} -> None")

            return result
        except Exception as e:
            elapsed = time.time() - start_time
            log.error(f"ERROR [{elapsed:.3f}s]: {func_name} -> {type(e).__name__}: {str(e)}")
            raise

    return wrapper