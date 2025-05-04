import traceback

DATABASE_URL = "postgresql://postgres:12345@localhost:5443/ftqc_db"
from fastapi import APIRouter, HTTPException, Depends, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
import base64
import re
import logging
import time
import os
import json
from datetime import datetime
import numpy as np
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
from pydantic import BaseModel, Field, PositiveInt, validator
import matplotlib

matplotlib.use('Agg')  # Force non-GUI backend for thread safety

# FastAPI app setup with compression
app = FastAPI(
    title="FTQC Simulation API",
    description="API for running fault-tolerant quantum computing simulations.",
    version="1.0.0"
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Database setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class SimulationResult(Base):
    __tablename__ = "simulation_results"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    job_id = Column(String, unique=True, nullable=False)
    simulation_id = Column(String, unique=True)  # Nullable
    result = Column(JSON)  # Nullable
    timestamp = Column(String, nullable=False)
    status = Column(String)  # Nullable


class JobLog(Base):
    __tablename__ = "job_logs"
    id = Column(Integer, primary_key=True)
    job_id = Column(String, unique=True, nullable=False)
    log_file = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)


# Drop and recreate tables to ensure the schema is up-to-date
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Thread Manager for handling multiple simulation jobs
class ThreadManager:
    def __init__(self, max_threads: int = 5):
        self.job_queue = queue.Queue()
        self.running_jobs = {}
        self.max_threads = max_threads
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_threads)
        self.logger = logging.getLogger("ThreadManager")
        self.logger.setLevel(logging.DEBUG)

    def add_job(self, user_id: int, job_func, *args):
        with self.lock:
            job_id = f"job_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            self.job_queue.put((job_id, user_id, job_func, args))
            self.running_jobs[job_id] = {"status": "queued", "result": None}
            self.logger.debug(f"Job {job_id} queued for user {user_id}")
        # Perform database operation in a separate thread to avoid blocking
        threading.Thread(target=self._store_job_in_db, args=(user_id, job_id), daemon=True).start()
        # Submit the job to the thread pool executor
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
            session.add(new_result)
            session.commit()
            self.logger.debug(f"Job {job_id} stored in database with status 'queued'")
        except Exception as e:
            self.logger.error(f"Failed to store job {job_id} in database: {str(e)}")
        finally:
            session.close()

    def _worker(self):
        try:
            # Get the next job from the queue
            job_id, user_id, job_func, args = self.job_queue.get(timeout=1)
            with self.lock:
                self.running_jobs[job_id]["status"] = "running"
                self.logger.debug(f"Job {job_id} started running")
                # Update status in database
                session = SessionLocal()
                job_record = session.query(SimulationResult).filter_by(job_id=job_id).first()
                if job_record:
                    job_record.status = "running"
                    session.commit()
                session.close()

            # Run the job
            result = job_func(job_id, *args)
            with self.lock:
                self.running_jobs[job_id]["status"] = "completed"
                self.running_jobs[job_id]["result"] = result
                self.logger.debug(f"Job {job_id} completed")
                # Update database with result
                session = SessionLocal()
                job_record = session.query(SimulationResult).filter_by(job_id=job_id).first()
                if job_record:
                    job_record.status = "completed"
                    job_record.simulation_id = result.simulation_id
                    job_record.result = result.result
                    session.commit()
                session.close()
        except queue.Empty:
            self.logger.debug("Worker stopped due to empty queue")
        except Exception as e:
            with self.lock:
                self.running_jobs[job_id]["status"] = "failed"
                self.running_jobs[job_id]["result"] = str(e)
                self.logger.error(f"Job {job_id} failed: {str(e)}")
                # Update status in database
                session = SessionLocal()
                job_record = session.query(SimulationResult).filter_by(job_id=job_id).first()
                if job_record:
                    job_record.status = "failed"
                    job_record.result = {"error": str(e)}
                    session.commit()
                session.close()
        finally:
            self.job_queue.task_done()

    def get_job_status(self, job_id: str):
        with self.lock:
            return self.running_jobs.get(job_id, {}).get("status", "not found")

    def get_job_result(self, job_id: str):
        with self.lock:
            return self.running_jobs.get(job_id, {}).get("result")


thread_manager = ThreadManager(max_threads=5)

# Logging setup
LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
log_queue = queue.Queue()

# Validation logger (separate from simulation logs)
validation_logger = logging.getLogger("validation_logger")
validation_logger.setLevel(logging.DEBUG)
validation_handler = logging.FileHandler(os.path.join(LOGS_DIR, "validation_errors.log"))
validation_handler.setFormatter(
    logging.Formatter('%(asctime)s,%(msecs)d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
validation_logger.addHandler(validation_handler)


# Simulation logger
def setup_job_logger(job_id: str):
    log_filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_debug-{job_id}.log")
    logger = logging.getLogger(job_id)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(
        logging.Formatter('%(asctime)s,%(msecs)d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    # Store log file in database
    session = SessionLocal()
    job_log = JobLog(job_id=job_id, log_file=log_filename, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    session.add(job_log)
    session.commit()
    session.close()
    return logger


async def log_streamer(job_id: str) -> AsyncGenerator[str, None]:
    session = SessionLocal()
    job_log = session.query(JobLog).filter_by(job_id=job_id).first()
    session.close()
    if not job_log or not os.path.exists(job_log.log_file):
        yield f"data: Log file for job {job_id} not found\n\n"
        return
    with open(job_log.log_file, "r") as f:
        while True:
            line = f.readline()
            if line:
                yield f"data: {line.strip()}\n\n"
            else:
                if thread_manager.get_job_status(job_id) not in ["completed", "failed"]:
                    await asyncio.sleep(0.1)
                else:
                    break


# Pydantic models for validation
class SimulationRequest(BaseModel):
    source_code: str = Field(
        ...,
        description="Base64-encoded quantum circuit source code. Example: 'ZnJvbSBxaXNraXQgaW1wb3J0IFF1YW50dW1DaXJjdWl0CnFjID0gUXVhbnR1bUNpcmN1aXQoMiwgMikKcWMuaCgwKQpxYy5jeCgwLCAxKQpxYy50KDApCnFjLnQoMSkKcWMubWVhc3VyZShbMCwgMV0sIFswLCAxXSkK'"
    )
    iterations: int = Field(10, ge=1, description="Number of simulation iterations")
    noise: float = Field(0.001, ge=0.0, le=1.0, description="Noise level")
    distance: int = Field(3, ge=1, description="Distance parameter")
    rounds: int = Field(2, ge=1, description="Number of rounds")
    error_rate: float = Field(0.01, ge=0.0, le=1.0, description="Error rate")
    debug: bool = Field(False, description="Enable debug mode")
    user_id: PositiveInt = Field(..., description="User ID")

    @validator('source_code')
    def validate_base64(cls, v):
        validation_logger.debug(f"Validating Base64 string: {v}")

        # Check if the string contains only valid Base64 characters
        if not re.match(r'^[A-Za-z0-9+/=]+$', v):
            validation_logger.debug(f"Invalid Base64 characters in string: {v}")
            error_msg = (
                f"Source code must be a valid Base64-encoded string. "
                f"Input '{v}' contains invalid characters. "
                f"Allowed characters are A-Z, a-z, 0-9, +, /, and =. "
                f"Example of a valid Base64 string: 'ZnJvbSBxaXNraXQgaW1wb3J0IFF1YW50dW1DaXJjdWl0CnFjID0gUXVhbnR1bUNpcmN1aXQoMiwgMikKcWMuaCgwKQpxYy5jeCgwLCAxKQpxYy50KDApCnFjLnQoMSkKcWMubWVhc3VyZShbMCwgMV0sIFswLCAxXSkK'. "
                f"Use the /encode endpoint to convert your Python code to Base64."
            )
            raise ValueError(error_msg)

        # Check length before padding
        original_length = len(v)
        padding_needed = (4 - original_length % 4) % 4
        if padding_needed > 0 and v[-padding_needed:] != '=' * padding_needed:
            validation_logger.debug(f"Base64 string missing padding: {v}")
            v_padded = v + '=' * padding_needed
        else:
            v_padded = v
        validation_logger.debug(f"Base64 string after padding: {v_padded}")

        # Validate length after padding
        if len(v_padded) % 4 != 0:
            validation_logger.debug(f"Base64 string length not a multiple of 4 after padding: {v_padded}")
            error_msg = (
                f"Base64 string length must be a multiple of 4. "
                f"Input '{v}' has length {original_length}, after padding: {len(v_padded)}. "
                f"Ensure proper padding with '=' characters. "
                f"Example of a valid Base64 string: 'ZnJvbSBxaXNraXQgaW1wb3J0IFF1YW50dW1DaXJjdWl0CnFjID0gUXVhbnR1bUNpcmN1aXQoMiwgMikKcWMuaCgwKQpxYy5jeCgwLCAxKQpxYy50KDApCnFjLnQoMSkKcWMubWVhc3VyZShbMCwgMV0sIFswLCAxXSkK'. "
                f"Use the /encode endpoint to convert your Python code to Base64."
            )
            raise ValueError(error_msg)

        try:
            decoded = base64.b64decode(v_padded, validate=True)
            decoded_str = decoded.decode('utf-8')
            validation_logger.debug(f"Decoded Base64 string: {decoded_str}")

            # Validate quantum circuit
            local_vars = {}
            # Provide numpy in the globals for exec
            exec(decoded_str, {"QuantumCircuit": QuantumCircuit, "print": print, "np": np}, local_vars)
            if 'qc' not in local_vars:
                validation_logger.debug("Decoded code does not define a QuantumCircuit named 'qc'")
                raise ValueError("Source code must define a QuantumCircuit named 'qc'")
            circuit = local_vars['qc']

            if circuit.num_qubits < 1:
                validation_logger.debug("Circuit has no qubits")
                raise ValueError("Quantum circuit must have at least 1 qubit")
            if circuit.num_clbits < 1 and any(instr.operation.name == 'measure' for instr in circuit):
                validation_logger.debug("Circuit has measurement gates but no classical bits")
                raise ValueError("Quantum circuit with measurement gates must have at least 1 classical bit")

            allowed_gates = {'h', 't', 'cx', 'swap', 'measure'}
            for instruction in circuit:
                gate = instruction.operation
                if gate.name not in allowed_gates:
                    validation_logger.debug(f"Unsupported gate found: {gate.name}")
                    raise ValueError(
                        f"Unsupported gate '{gate.name}' in circuit. Allowed gates: {allowed_gates}. "
                        f"If using gates like 'cp' or 'tdg', decompose them into allowed gates (H, T, CX, etc.) before encoding."
                    )
                qubits = [circuit.find_bit(q).index for q in instruction.qubits]
                for qubit in qubits:
                    if qubit >= circuit.num_qubits:
                        validation_logger.debug(f"Invalid qubit index: {qubit}")
                        raise ValueError(
                            f"Qubit index {qubit} is out of range for circuit with {circuit.num_qubits} qubits")
                if gate.name == 'measure':
                    clbits = [circuit.find_bit(c).index for c in instruction.clbits]
                    for clbit in clbits:
                        if clbit >= circuit.num_clbits:
                            validation_logger.debug(f"Invalid classical bit index: {clbit}")
                            raise ValueError(
                                f"Classical bit index {clbit} is out of range for circuit with {circuit.num_clbits} classical bits")

            validation_logger.debug("Base64 and circuit validation successful")
            return v
        except base64.binascii.Error as e:
            validation_logger.debug(f"Base64 decoding failed: {str(e)}")
            example_base64 = "ZnJvbSBxaXNraXQgaW1wb3J0IFF1YW50dW1DaXJjdWl0CnFjID0gUXVhbnR1bUNpcmN1aXQoMiwgMikKcWMuaCgwKQpxYy5jeCgwLCAxKQpxYy50KDApCnFjLnQoMSkKcWMubWVhc3VyZShbMCwgMV0sIFswLCAxXSkK"
            error_msg = (
                f"Invalid Base64 encoding: {str(e)}. "
                f"Input length: {original_length} characters, after padding: {len(v_padded)} characters. "
                f"Base64 strings must encode valid data. "
                f"Ensure the source code is properly Base64-encoded. Example of a valid Base64 string: '{example_base64}'. "
                f"Use the /encode endpoint to convert your Python code to Base64."
            )
            raise ValueError(error_msg)
        except Exception as e:
            validation_logger.debug(f"Quantum circuit validation failed: {str(e)}")
            raise ValueError(f"Invalid quantum circuit code: {str(e)}")


class SimulationUpdate(BaseModel):
    source_code: str | None = Field(None, description="Base64-encoded quantum circuit source code")
    iterations: int | None = Field(None, ge=1, description="Number of simulation iterations")
    noise: float | None = Field(None, ge=0.0, le=1.0, description="Noise level")
    distance: int | None = Field(None, ge=1, description="Distance parameter")
    rounds: int | None = Field(None, ge=1, description="Number of rounds")
    error_rate: float | None = Field(None, ge=0.0, le=1.0, description="Error rate")
    debug: bool | None = Field(None, description="Enable debug mode")

    @validator('source_code')
    def validate_base64(cls, v):
        if v is None:
            return v
        validation_logger.debug(f"Validating Base64 string in update: {v}")
        if not re.match(r'^[A-Za-z0-9+/=]+$', v):
            raise ValueError(
                "Source code must be a valid Base64-encoded string. It should only contain characters A-Z, a-z, 0-9, +, /, and =."
            )
        padding_needed = (4 - len(v) % 4) % 4
        if padding_needed > 0 and v[-padding_needed:] != '=' * padding_needed:
            v_padded = v + '=' * padding_needed
        else:
            v_padded = v
        if len(v_padded) % 4 != 0:
            error_msg = (
                f"Base64 string length must be a multiple of 4. "
                f"Input '{v}' has length {len(v)}, after padding: {len(v_padded)}. "
                f"Ensure proper padding with '=' characters. "
                f"Example of a valid Base64 string: 'ZnJvbSBxaXNraXQgaW1wb3J0IFF1YW50dW1DaXJjdWl0CnFjID0gUXVhbnR1bUNpcmN1aXQoMiwgMikKcWMuaCgwKQpxYy5jeCgwLCAxKQpxYy50KDApCnFjLnQoMSkKcWMubWVhc3VyZShbMCwgMV0sIFswLCAxXSkK'. "
                f"Use the /encode endpoint to convert your Python code to Base64."
            )
            raise ValueError(error_msg)
        try:
            decoded = base64.b64decode(v_padded, validate=True)
            decoded_str = decoded.decode('utf-8')
            local_vars = {}
            exec(decoded_str, {"QuantumCircuit": QuantumCircuit, "print": print, "np": np}, local_vars)
            if 'qc' not in local_vars:
                raise ValueError("Source code must define a QuantumCircuit named 'qc'")
            circuit = local_vars['qc']
            allowed_gates = {'h', 't', 'cx', 'swap', 'measure'}
            for instruction in circuit:
                gate = instruction.operation
                if gate.name not in allowed_gates:
                    raise ValueError(
                        f"Unsupported gate '{gate.name}' in circuit. Allowed gates: {allowed_gates}. "
                        f"If using gates like 'cp' or 'tdg', decompose them into allowed gates (H, T, CX, etc.) before encoding."
                    )
                qubits = [circuit.find_bit(q).index for q in instruction.qubits]
                for qubit in qubits:
                    if qubit >= circuit.num_qubits:
                        raise ValueError(
                            f"Qubit index {qubit} is out of range for circuit with {circuit.num_qubits} qubits")
                if gate.name == 'measure':
                    clbits = [circuit.find_bit(c).index for c in instruction.clbits]
                    for clbit in clbits:
                        if clbit >= circuit.num_clbits:
                            raise ValueError(
                                f"Classical bit index {clbit} is out of range for circuit with {circuit.num_clbits} classical bits")
            return v
        except base64.binascii.Error as e:
            example_base64 = "ZnJvbSBxaXNraXQgaW1wb3J0IFF1YW50dW1DaXJjdWl0CnFjID0gUXVhbnR1bUNpcmN1aXQoMiwgMikKcWMuaCgwKQpxYy5jeCgwLCAxKQpxYy50KDApCnFjLnQoMSkKcWMubWVhc3VyZShbMCwgMV0sIFswLCAxXSkK"
            error_msg = (
                f"Invalid Base64 encoding: {str(e)}. "
                f"Input length: {len(v)} characters, after padding: {len(v_padded)} characters. "
                f"Base64 strings must encode valid data. "
                f"Ensure the source code is properly Base64-encoded. Example of a valid Base64 string: '{example_base64}'. "
                f"Use the /encode endpoint to convert your Python code to Base64."
            )
            raise ValueError(error_msg)
        except Exception as e:
            raise ValueError(f"Invalid quantum circuit code: {str(e)}")


class SimulationResponse(BaseModel):
    simulation_id: str
    result: dict
    timestamp: str


# Router
router = APIRouter()


def run_simulation_with_timeout(job_id: str, request: SimulationRequest):
    # Setup logger after validation
    logger = setup_job_logger(job_id)
    logger.info(f"Starting simulation for user {request.user_id}")
    try:
        source_code = base64.b64decode(request.source_code).decode('utf-8')
        local_vars = {}
        exec(source_code, {"QuantumCircuit": QuantumCircuit, "print": print, "np": np}, local_vars)
        circuit = local_vars['qc']  # Already validated in the model

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

        avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts = pipeline.run()

        if avg_fidelity is None:
            raise ValueError("Simulation failed. Check logs for details.")

        # Use the detailed JSON response from pipeline.py
        result = json.loads(
            pipeline.generate_json_response(avg_fidelity, logical_error_rate, success_rate, avg_success_prob,
                                            msd_attempts))

        simulation_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Store JSON
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
    job_id = thread_manager.add_job(request.user_id, run_simulation_with_timeout, request)
    return {"job_id": job_id, "status": "queued"}


@router.put("/simulate/{job_id}", response_model=SimulationResponse)
async def update_simulation(job_id: str, update: SimulationUpdate, db: SQLSession = Depends(get_db)):
    # Check if job exists
    simulation = db.query(SimulationResult).filter(SimulationResult.job_id == job_id).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Retrieve original request parameters (we'll use default values as a fallback)
    original_request = SimulationRequest(
        source_code=None,
        iterations=simulation.result.get("iterations", 10) if simulation.result else 10,
        noise=simulation.result.get("noise", 0.001) if simulation.result else 0.001,
        distance=simulation.result.get("distance", 3) if simulation.result else 3,
        rounds=simulation.result.get("rounds", 2) if simulation.result else 2,
        error_rate=simulation.result.get("error_rate", 0.01) if simulation.result else 0.01,
        debug=simulation.result.get("debug", False) if simulation.result else False,
        user_id=simulation.user_id
    )

    # Update fields if provided
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

    # Re-run simulation with updated parameters
    job_id = thread_manager.add_job(updated_request.user_id, run_simulation_with_timeout, updated_request)
    return {"job_id": job_id, "status": "queued"}


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


@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@router.post("/encode")
async def encode_source_code(code: str):
    try:
        encoded = base64.b64encode(code.encode('utf-8')).decode('utf-8')
        return {"base64_encoded": encoded}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to encode source code: {str(e)}")