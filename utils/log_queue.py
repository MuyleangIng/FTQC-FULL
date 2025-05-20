import logging
import queue
from logging.handlers import QueueHandler, QueueListener
from logging import FileHandler
from config import LOGS_DIR

# Set up the queue and listener for asynchronous logging
log_queue = queue.Queue(-1)  # No limit on queue size
queue_handler = QueueHandler(log_queue)

# Configure a FileHandler for the queue listener
file_handler = FileHandler(os.path.join(LOGS_DIR, "ftqc_pipeline_async.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s,%(msecs)d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Start the QueueListener
listener = QueueListener(log_queue, file_handler)
listener.start()

# Ensure the listener stops when the application exits
import atexit
atexit.register(listener.stop)