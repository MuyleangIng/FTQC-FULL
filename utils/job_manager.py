import threading
import uuid
from queue import Queue
from typing import Dict, Optional

from core.pipeline import FTQCPipeline
from utils.logging import setup_logging


class SimulationJob:
    def __init__(self, job_id: str, user_id: str, params: dict):
        self.job_id = job_id
        self.user_id = user_id
        self.params = params
        self.thread = None
        self.event = threading.Event()
        self.stop_flag = False
        self.status = "pending"
        self.result = None

    def run(self, logger):
        self.status = "running"
        self.event.set()  # Allow running
        try:
            pipeline = FTQCPipeline(
                circuit=self.params["source_code"],
                iterations=self.params["iterations"],
                noise=self.params["noise"],
                code_distance=self.params["distance"],
                msd_rounds=self.params["rounds"],
                logger=logger
            )
            self.result = pipeline.run_simulation()
            self.status = "completed"
        except Exception as e:
            logger.error(f"Job {self.job_id} failed: {str(e)}")
            self.status = "failed"
            self.result = {"error": str(e)}

    def pause(self):
        self.event.clear()
        self.status = "paused"

    def resume(self):
        self.event.set()
        self.status = "running"

    def stop(self):
        self.stop_flag = True
        self.event.set()
        self.status = "stopped"

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, SimulationJob] = {}
        self.lock = threading.Lock()
        self.logger_queue = Queue()

    def submit_job(self, user_id: str, params: dict) -> str:
        job_id = str(uuid.uuid4())
        logger = setup_logging(user_id=user_id, job_id=job_id)
        job = SimulationJob(job_id, user_id, params)
        with self.lock:
            self.jobs[job_id] = job
        job.thread = threading.Thread(target=job.run, args=(logger,))
        job.thread.start()
        return job_id

    def get_job_status(self, job_id: str) -> Optional[dict]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                return {"job_id": job_id, "user_id": job.user_id, "status": job.status}
            return None

    def control_job(self, job_id: str, action: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            if action == "pause":
                job.pause()
            elif action == "resume":
                job.resume()
            elif action == "stop":
                job.stop()
            else:
                return False
            return True

    def get_job_result(self, job_id: str) -> Optional[dict]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job.status in ["completed", "failed"]:
                return job.result
            return None