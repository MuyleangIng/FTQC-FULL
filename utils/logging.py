import logging
import os
from datetime import datetime
from config import LOGS_DIR


def setup_logging(filename: str = None) -> None:
    """
    Set up logging to file and console.
    """
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    if filename is None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(LOGS_DIR, f"ftqc_debug-{current_time}.log")

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(filename),
            logging.StreamHandler()
        ],
        format='%(asctime)s,%(msecs)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )