import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

DATABASE_URL = "postgresql://postgres:12345@localhost:5443/ftqc_db"
MAX_THREADS = 8

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEFAULT_NOISE = 0.001
DEFAULT_ITERATIONS = 10
DEFAULT_CODE_DISTANCE = 3
DEFAULT_MSD_ROUNDS = 2
DEFAULT_ERROR_RATE = 0.01
DEFAULT_PARALLEL_UNITS = 2