from celery import Celery

import os
import json
from datetime import datetime

from core.pipeline import FTQCPipeline
from utils.logging import setup_logging

app = Celery('ftqc_tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@app.task
def run_simulation(user_id: str, job_id: str, params: dict) -> dict:
    logger = setup_logging(user_id, job_id)
    try:
        pipeline = FTQCPipeline(
            circuit=params["source_code"],
            iterations=params["iterations"],
            noise=params["noise"],
            code_distance=params["distance"],
            msd_rounds=params["rounds"],
            logger=logger
        )
        result = pipeline.run_simulation()

        # Save JSON response
        response_dir = f"logs/user_{user_id}/responses"
        os.makedirs(response_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        response_path = f"{response_dir}/{job_id}.json"
        with open(response_path, "w") as f:
            json.dump(result, f, indent=2)

        return {"status": "completed", "response_path": response_path, "result": result}
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        return {"status": "failed", "error": str(e)}