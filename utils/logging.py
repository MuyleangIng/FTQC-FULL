import logging
import os
import queue
from logging.handlers import QueueHandler, QueueListener
from logging import FileHandler
from config import LOGS_DIR

# Ensure logs directory exists
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Set up the queue and listener for asynchronous logging
log_queue = queue.Queue(-1)  # No limit on queue size
queue_handler = QueueHandler(log_queue)

# Configure a FileHandler for the queue listener
global_file_handler = FileHandler(os.path.join(LOGS_DIR, "ftqc_pipeline_async.log"))
global_file_handler.setLevel(logging.DEBUG)
global_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s,%(msecs)d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Start the QueueListener
listener = QueueListener(log_queue, global_file_handler)
listener.start()

# Ensure the listener stops when the application exits
import atexit

atexit.register(listener.stop)


class FlushFileHandler(FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # Force flush after each log message to ensure real-time updates


def setup_logging(job_id: str, user_id: str = None) -> logging.Logger:
    """
    Set up logging to file for a specific job using a queue handler and real-time flushing.
    """
    filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_debug-{job_id}.log")
    logger_name = f"job_{job_id}"
    logger = logging.getLogger(logger_name)

    # Clear any existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # Set logger level to DEBUG to capture all messages
    logger.setLevel(logging.DEBUG)

    # Create a FlushFileHandler for this specific job
    file_handler = FlushFileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s,%(msecs)d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Add both the QueueHandler (thread-safe) and FlushFileHandler
    logger.addHandler(queue_handler)
    logger.addHandler(file_handler)

    # Prevent log messages from propagating to the root logger
    logger.propagate = False

    logger.debug(f"Logger {logger_name} initialized with handlers: {logger.handlers}")
    return logger