# llm_dev_assistant/gui/log_handler.py
"""Custom log handler for GUI integration."""

import logging
import queue
from datetime import datetime
from typing import Optional


class QueueLogHandler(logging.Handler):
    """Log handler that puts log records into a queue for GUI consumption."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        """Emit a log record."""
        try:
            # Format the log entry for GUI display
            log_entry = {
                'time': datetime.fromtimestamp(record.created).strftime('%H:%M:%S'),
                'level': record.levelname,
                'message': self.format(record),
                'module': record.name,
                'filename': record.filename,
                'lineno': record.lineno
            }

            # Put it in the queue
            self.log_queue.put(log_entry)

        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def setup_gui_logging(log_queue: queue.Queue, level: int = logging.DEBUG) -> QueueLogHandler:
    """
    Set up log_system to redirect to GUI.

    Args:
        log_queue: Queue to put log entries into
        level: Logging level

    Returns:
        The queue log handler
    """
    # Create the queue handler
    queue_handler = QueueLogHandler(log_queue)
    queue_handler.setLevel(level)

    # Set a simple format for GUI display
    formatter = logging.Formatter('%(message)s')
    queue_handler.setFormatter(formatter)

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)

    return queue_handler


def remove_gui_logging(handler: QueueLogHandler):
    """Remove GUI log_system handler."""
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)