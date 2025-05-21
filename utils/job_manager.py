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
        self.event.set()
        try:
            pipeline = FTQCPipeline(
                circuit=self.params["source_code"],
                iterations=self.params["iterations"],
                noise=self.params["noise"],
                distance=self.params["distance"],
                rounds=self.params["rounds"],
                logger=logger
            )
            self.result = pipeline.run()
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
    def __init__(self, max_jobs: int = 50):
        self.jobs: Dict[str, SimulationJob] = {}
        self.lock = threading.Lock()
        self.max_jobs = max_jobs
        self.job_queue = Queue()

    def submit_job(self, user_id: str, params: dict) -> str:
        with self.lock:
            if len(self.jobs) >= self.max_jobs:
                raise RuntimeError("Maximum number of concurrent jobs reached")
            job_id = str(uuid.uuid4())
            logger = setup_logging(user_id=user_id, job_id=job_id)
            job = SimulationJob(job_id, user_id, params)
            self.jobs[job_id] = job
        job.thread = threading.Thread(target=self._run_job, args=(job, logger))
        job.thread.start()
        return job_id

    def _run_job(self, job: SimulationJob, logger):
        try:
            job.run(logger)
        finally:
            with self.lock:
                if job.status in ["completed", "failed", "stopped"]:
                    self.jobs.pop(job.job_id, None)

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
                result = job.result
                self.jobs.pop(job_id, None)
                return result
            return None