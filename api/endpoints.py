import traceback
import base64
import re
import logging
import time
import os
import json
from datetime import datetime
import numpy as np
import math
from typing import Dict, List, AsyncGenerator
from qiskit import QuantumCircuit
from core.pipeline import FTQCPipeline
import asyncio
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session as SQLSession
from fastapi import APIRouter, HTTPException, Depends, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware  # Add CORS middleware
from pydantic import BaseModel, Field, PositiveInt
from config import LOGS_DIR, DATABASE_URL, MAX_THREADS
import matplotlib

matplotlib.use('Agg')

app = FastAPI(
    title="FTQC Simulation API",
    description="API for running fault-tolerant quantum computing simulations.",
    version="1.0.0"
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

class SimulationResult(Base):
    __tablename__ = "simulation_results"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    job_id = Column(String, unique=True, nullable=False)
    simulation_id = Column(String, unique=True)
    result = Column(JSON)
    timestamp = Column(String, nullable=False)
    status = Column(String)

class JobLog(Base):
    __tablename__ = "job_logs"
    id = Column(Integer, primary_key=True)
    job_id = Column(String, unique=True, nullable=False)
    log_file = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ThreadManager:
    def __init__(self, max_threads: int = MAX_THREADS):
        self.job_queue = queue.Queue()
        self.running_jobs = {}
        self.max_threads = max_threads
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_threads)
        self.logger = logging.getLogger("ThreadManager")
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(os.path.join(LOGS_DIR, "thread_manager.log"))
            handler.setFormatter(logging.Formatter(
                '%(asctime)s,%(msecs)d - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(handler)

    def add_job(self, job_id: str, user_id: int, job_func, *args):
        with self.lock:
            self._store_job_in_db(user_id, job_id)
            self.job_queue.put((job_id, user_id, job_func, args))
            self.running_jobs[job_id] = {"status": "queued", "result": None, "logger": None}
            self.logger.debug(f"Job {job_id} queued for user {user_id}")
        self.executor.submit(self._worker)
        return job_id

    def _store_job_in_db(self, user_id: int, job_id: str):
        try:
            session = SessionLocal()
            new_result = SimulationResult(
                user_id=user_id,
                job_id=job_id,
                simulation_id=None,
                result=None,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                status="queued"
            )
            job_log = JobLog(
                job_id=job_id,
                log_file=os.path.join(LOGS_DIR, f"ftqc_pipeline_debug-{job_id}.log"),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            session.add(new_result)
            session.add(job_log)
            session.commit()
            self.logger.debug(f"Job {job_id} stored in database with status 'queued'")
        except Exception as e:
            self.logger.error(f"Failed to store job {job_id} in database: {str(e)}")
        finally:
            session.close()

    def _worker(self):
        try:
            job_id, user_id, job_func, args = self.job_queue.get(timeout=1)
            with self.lock:
                self.running_jobs[job_id]["status"] = "running"
                self.logger.debug(f"Job {job_id} started running")
                session = SessionLocal()
                job_record = session.query(SimulationResult).filter_by(job_id=job_id).first()
                if job_record:
                    job_record.status = "running"
                    session.commit()
                else:
                    self.logger.warning(f"No job record found for job_id {job_id} when updating status to running")
                session.close()

            request, logger = args
            self.running_jobs[job_id]["logger"] = logger

            result = job_func(job_id, request, logger)
            with self.lock:
                if result is None:
                    self.running_jobs[job_id]["status"] = "failed"
                    self.running_jobs[job_id]["result"] = {"error": "Simulation returned None"}
                    self.logger.error(f"Job {job_id} failed: Simulation returned None")
                else:
                    self.running_jobs[job_id]["status"] = "completed"
                    self.running_jobs[job_id]["result"] = result
                    self.logger.debug(f"Job {job_id} completed")
                session = SessionLocal()
                job_record = session.query(SimulationResult).filter_by(job_id=job_id).first()
                if job_record:
                    job_record.status = self.running_jobs[job_id]["status"]
                    job_record.simulation_id = result.simulation_id if result else None
                    job_record.result = result.__dict__ if result else {"error": "Simulation returned None"}
                    session.commit()
                session.close()

                if self.running_jobs[job_id]["logger"]:
                    for handler in self.running_jobs[job_id]["logger"].handlers:
                        handler.flush()
                        handler.close()
        except queue.Empty:
            self.logger.debug("Worker stopped due to empty queue")
        except Exception as e:
            with self.lock:
                self.running_jobs[job_id]["status"] = "failed"
                self.running_jobs[job_id]["result"] = {"error": str(e)}
                self.logger.error(f"Job {job_id} failed: {str(e)}")
                session = SessionLocal()
                job_record = session.query(SimulationResult).filter_by(job_id=job_id).first()
                if job_record:
                    job_record.status = "failed"
                    job_record.result = {"error": str(e)}
                    session.commit()
                session.close()

                if self.running_jobs[job_id]["logger"]:
                    for handler in self.running_jobs[job_id]["logger"].handlers:
                        handler.flush()
                        handler.close()
        finally:
            self.job_queue.task_done()
            with self.lock:
                self.running_jobs.pop(job_id, None)

    def get_job_status(self, job_id: str):
        with self.lock:
            return self.running_jobs.get(job_id, {}).get("status", "not found")

    def get_job_result(self, job_id: str):
        with self.lock:
            return self.running_jobs.get(job_id, {}).get("result")

thread_manager = ThreadManager()

validation_logger = logging.getLogger("validation_logger")
if not validation_logger.handlers:
    validation_logger.setLevel(logging.DEBUG)
    validation_handler = logging.FileHandler(os.path.join(LOGS_DIR, "validation_errors.log"))
    validation_handler.setFormatter(
        logging.Formatter('%(asctime)s,%(msecs)d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    validation_logger.addHandler(validation_handler)

async def log_streamer(job_id: str) -> AsyncGenerator[str, None]:
    session = SessionLocal()
    job_log = session.query(JobLog).filter_by(job_id=job_id).first()
    session.close()
    if not job_log or not os.path.exists(job_log.log_file):
        yield f"data: Log file for job {job_id} not found\n\n"
        return

    # Read all existing lines from the log file
    with open(job_log.log_file, "r") as f:
        # Start from the beginning to read all existing logs
        lines = f.readlines()
        for line in lines:
            yield f"data: {line.strip()}\n\n"
            await asyncio.sleep(0.01)  # Small delay to ensure proper streaming

        # Now monitor for new lines if the job is still running
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                yield f"data: {line.strip()}\n\n"
                await asyncio.sleep(0.01)  # Small delay to ensure proper streaming
            else:
                # Check if the job is still running
                if thread_manager.get_job_status(job_id) not in ["completed", "failed"]:
                    await asyncio.sleep(0.1)  # Wait for more logs
                else:
                    # Keep the stream open for a bit longer to ensure all logs are flushed
                    for _ in range(50):  # Keep the stream alive for 5 seconds
                        line = f.readline()
                        if line:
                            yield f"data: {line.strip()}\n\n"
                            await asyncio.sleep(0.01)
                        await asyncio.sleep(0.1)
                    yield "data: [Streaming Complete]\n\n"
                    break

class SimulationRequest(BaseModel):
    source_code: str = Field(..., description="Quantum circuit source code.")
    iterations: int = Field(10, ge=1, description="Number of simulation iterations")
    noise: float = Field(0.001, ge=0.0, le=1.0, description="Noise level")
    distance: int = Field(3, ge=3, le=6, description="Distance parameter")
    rounds: int = Field(2, ge=1, description="Number of rounds")
    error_rate: float = Field(0.01, ge=0.0, le=1.0, description="Error rate")
    debug: bool = Field(True, description="Enable debug mode")
    user_id: PositiveInt = Field(..., description="User ID")

class SimulationUpdate(BaseModel):
    source_code: str | None = Field(None, description="Quantum circuit source code")
    iterations: int | None = Field(None, ge=1)
    noise: float | None = Field(None, ge=0.0, le=1.0)
    distance: int | None = Field(None, ge=3, le=6)
    rounds: int | None = Field(None, ge=1)
    error_rate: float | None = Field(None, ge=0.0, le=1.0)
    debug: bool | None = Field(None)

class SimulationResponse(BaseModel):
    simulation_id: str
    result: dict
    timestamp: str

router = APIRouter()

def run_simulation_with_timeout(job_id: str, request: SimulationRequest, logger: logging.Logger):
    logger.info(f"Starting simulation for user {request.user_id}")

    try:
        logger.debug("Decoding source code...")
        source_code = request.source_code
        if re.match(r'^[A-Za-z0-9+/=]+$', source_code):
            try:
                source_code = base64.b64decode(source_code).decode('utf-8')
                logger.debug(f"Decoded Base64 source code: {source_code[:100]}...")
            except Exception as e:
                logger.error(f"Invalid Base64-encoded source code: {str(e)}")
                raise ValueError(f"Invalid Base64-encoded source code: {str(e)}")

        logger.debug("Validating source code syntax...")
        try:
            compile(source_code, '<string>', 'exec')
        except SyntaxError as e:
            logger.error(f"Invalid Python syntax in source code: {str(e)}")
            raise ValueError(f"Invalid Python syntax in source code: {str(e)}")

        logger.debug("Executing source code to extract QuantumCircuit...")
        local_vars = {}
        exec(source_code, {"QuantumCircuit": QuantumCircuit, "print": print, "np": np, "math": math, "pi": math.pi},
             local_vars)
        if 'qc' not in local_vars:
            logger.error("Source code must define a QuantumCircuit named 'qc'")
            raise ValueError("Source code must define a QuantumCircuit named 'qc'")
        circuit = local_vars['qc']
        if not isinstance(circuit, QuantumCircuit) or circuit.num_qubits < 1:
            logger.error("Invalid QuantumCircuit: must have at least 1 qubit")
            raise ValueError("Invalid QuantumCircuit: must have at least 1 qubit")
        logger.debug(f"Extracted QuantumCircuit with {circuit.num_qubits} qubits")

        logger.debug("Initializing FTQCPipeline...")
        pipeline = FTQCPipeline(
            circuit=circuit,
            iterations=request.iterations,
            noise=request.noise,
            distance=request.distance,
            rounds=request.rounds,
            error_rate=request.error_rate,
            debug=request.debug,
            job_id=job_id
        )
        pipeline.logger = logger
        logger.debug(f"FTQCPipeline initialized with job_id {job_id}")

        logger.debug("Running FTQCPipeline...")
        avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts, visualizations = pipeline.run()
        logger.debug("FTQCPipeline execution completed")

        if avg_fidelity is None:
            logger.error("Simulation failed. Check logs for details.")
            raise ValueError("Simulation failed. Check logs for details.")

        logger.debug("Generating JSON response...")
        result = json.loads(
            pipeline.generate_json_response(avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts)
        )
        result["visualizations"] = visualizations

        simulation_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"Saving JSON response to {LOGS_DIR}/ftqc_pipeline_response-{simulation_id}.json...")
        json_path = os.path.join(LOGS_DIR, f"ftqc_pipeline_response-{simulation_id}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Simulation completed with ID {simulation_id}")
        return SimulationResponse(simulation_id=simulation_id, result=result, timestamp=timestamp)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@router.post("/simulate")
async def simulate(request: SimulationRequest, db: SQLSession = Depends(get_db)):
    from utils.logging import setup_logging
    job_id = f"job_{request.user_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    logger = setup_logging(job_id=job_id, user_id=str(request.user_id))
    thread_manager.add_job(job_id, request.user_id, run_simulation_with_timeout, request, logger)
    return {"job_id": job_id, "status": "queued"}

@router.post("/simulate_with_distances")
async def simulate_with_distances(request: SimulationRequest, db: SQLSession = Depends(get_db)):
    from utils.logging import setup_logging
    distances = [3, 5, 7]
    results = []

    for distance in distances:
        modified_request = SimulationRequest(
            source_code=request.source_code,
            iterations=request.iterations,
            noise=request.noise,
            distance=distance,
            rounds=request.rounds,
            error_rate=request.error_rate,
            debug=request.debug,
            user_id=request.user_id
        )

        job_id = f"job_{request.user_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_distance_{distance}"
        logger = setup_logging(job_id=job_id, user_id=str(request.user_id))
        thread_manager.add_job(job_id, request.user_id, run_simulation_with_timeout, modified_request, logger)

        while thread_manager.get_job_status(job_id) not in ["completed", "failed"]:
            await asyncio.sleep(0.5)

        result = thread_manager.get_job_result(job_id)
        if result is None or isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=f"Simulation failed for distance {distance}: {result.get('error') if result else 'Unknown error'}")

        result_dict = result.result
        results.append({
            "distance": distance,
            "logicalErrorRate": result_dict["performance"]["actualLogicalErrorRate"],
            "executionTime": result_dict["performance"]["executionTime"],
            "job_id": job_id
        })

    return {
        "results": results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@router.put("/simulate/{job_id}", response_model=SimulationResponse)
async def update_simulation(job_id: str, update: SimulationUpdate, db: SQLSession = Depends(get_db)):
    simulation = db.query(SimulationResult).filter(SimulationResult.job_id == job_id).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")

    original_request = SimulationRequest(
        source_code=simulation.result.get("source_code") if simulation.result else None,
        iterations=simulation.result.get("iterations", 10) if simulation.result else 10,
        noise=simulation.result.get("noise", 0.001) if simulation.result else 0.001,
        distance=simulation.result.get("distance", 3) if simulation.result else 3,
        rounds=simulation.result.get("rounds", 2) if simulation.result else 2,
        error_rate=simulation.result.get("error_rate", 0.01) if simulation.result else 0.01,
        debug=simulation.result.get("debug", True) if simulation.result else True,
        user_id=simulation.user_id
    )

    updated_request = SimulationRequest(
        source_code=update.source_code if update.source_code else original_request.source_code,
        iterations=update.iterations if update.iterations is not None else original_request.iterations,
        noise=update.noise if update.noise is not None else original_request.noise,
        distance=update.distance if update.distance is not None else original_request.distance,
        rounds=update.rounds if update.rounds is not None else original_request.rounds,
        error_rate=update.error_rate if update.error_rate is not None else original_request.error_rate,
        debug=update.debug if update.debug is not None else original_request.debug,
        user_id=simulation.user_id
    )

    from utils.logging import setup_logging
    new_job_id = f"job_{updated_request.user_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    logger = setup_logging(job_id=new_job_id, user_id=str(updated_request.user_id))
    thread_manager.add_job(new_job_id, updated_request.user_id, run_simulation_with_timeout, updated_request, logger)
    return {"job_id": new_job_id, "status": "queued"}

@router.get("/simulate/{job_id}/logs")
async def stream_logs(job_id: str):
    return StreamingResponse(log_streamer(job_id), media_type="text/event-stream")

@router.get("/simulations/{user_id}")
async def get_simulations(user_id: int, db: SQLSession = Depends(get_db)):
    results = db.query(SimulationResult).filter_by(user_id=user_id).all()
    return [
        {
            "job_id": r.job_id,
            "simulation_id": r.simulation_id,
            "result": r.result,
            "timestamp": r.timestamp,
            "status": r.status
        }
        for r in results
    ]

@router.get("/simulation/{user_id}/{job_id}")
async def get_simulation_by_job(user_id: int, job_id: str, db: SQLSession = Depends(get_db)):
    result = db.query(SimulationResult).filter_by(user_id=user_id, job_id=job_id).first()
    if result:
        return {
            "job_id": result.job_id,
            "simulation_id": result.simulation_id,
            "result": result.result,
            "timestamp": result.timestamp,
            "status": result.status
        }
    raise HTTPException(status_code=404, detail="Simulation not found")

@router.get("/json/{user_id}/{job_id}")
async def get_json(user_id: int, job_id: str, db: SQLSession = Depends(get_db)):
    result = db.query(SimulationResult).filter_by(user_id=user_id, job_id=job_id).first()
    if result:
        json_path = os.path.join(LOGS_DIR, f"ftqc_pipeline_response-{result.simulation_id}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="JSON file not found")
    raise HTTPException(status_code=404, detail="Simulation not found")

@router.get("/playback/{job_id}")
async def playback_logs(job_id: str, db: SQLSession = Depends(get_db)):
    job_log = db.query(JobLog).filter_by(job_id=job_id).first()
    if not job_log or not os.path.exists(job_log.log_file):
        raise HTTPException(status_code=404, detail=f"Log file for job {job_id} not found")

    logs = []
    with open(job_log.log_file, "r") as f:
        for line in f:
            logs.append(line.strip())

    return {
        "job_id": job_id,
        "logs": logs,
        "timestamp": job_log.timestamp
    }

@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}