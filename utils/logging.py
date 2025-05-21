import logging
import os
import queue
from logging.handlers import QueueHandler, QueueListener
from logging import FileHandler
from config import LOGS_DIR

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

log_queue = queue.Queue(-1)
queue_handler = QueueHandler(log_queue)

global_file_handler = FileHandler(os.path.join(LOGS_DIR, "ftqc_pipeline_async.log"))
global_file_handler.setLevel(logging.DEBUG)
global_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s,%(msecs)d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

listener = QueueListener(log_queue, global_file_handler)
listener.start()

import atexit
atexit.register(listener.stop)

class FlushFileHandler(FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logging(job_id: str, user_id: str = None) -> logging.Logger:
    filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_debug-{job_id}.log")
    logger_name = f"job_{job_id}"
    logger = logging.getLogger(logger_name)

    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    file_handler = FlushFileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s,%(msecs)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    logger.addHandler(queue_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.debug(f"Logger {logger_name} initialized with handlers: {logger.handlers}")
    return logger