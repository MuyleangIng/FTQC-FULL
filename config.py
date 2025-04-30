import os

# Directory for logs and outputs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Default simulation parameters
DEFAULT_NOISE = 0.001
DEFAULT_ITERATIONS = 10
DEFAULT_CODE_DISTANCE = 3
DEFAULT_MSD_ROUNDS = 2
DEFAULT_ERROR_RATE = 0.01
DEFAULT_PARALLEL_UNITS = 2