from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import base64
import traceback
import logging
from datetime import datetime
import numpy as np
from typing import Dict, List
from qiskit import QuantumCircuit
from core.pipeline import FTQCPipeline

# Create an APIRouter instead of a FastAPI app
router = APIRouter()

# In-memory storage for simulation results
simulations: Dict[str, dict] = {}


class SimulationRequest(BaseModel):
    source_code: str
    iterations: int = 10
    noise: float = 0.001
    distance: int = 3
    rounds: int = 2
    error_rate: float = 0.01
    debug: bool = False


class SimulationResponse(BaseModel):
    simulation_id: str
    result: dict
    timestamp: str


@router.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    try:
        # Decode the base64 source code
        source_code = base64.b64decode(request.source_code).decode('utf-8')

        # Safely execute the source code to create a QuantumCircuit
        local_vars = {}
        exec(source_code, {"QuantumCircuit": QuantumCircuit, "print": print}, local_vars)
        if 'qc' not in local_vars:
            raise ValueError("Source code must define a QuantumCircuit named 'qc'")
        circuit = local_vars['qc']

        # Run the pipeline
        pipeline = FTQCPipeline(
            circuit=circuit,
            iterations=request.iterations,
            noise=request.noise,
            distance=request.distance,
            rounds=request.rounds,
            error_rate=request.error_rate,
            debug=request.debug
        )
        avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts = pipeline.run()

        if avg_fidelity is None:
            raise ValueError("Simulation failed. Check logs for details.")

        # Prepare the result
        result = {
            "avg_fidelity": float(avg_fidelity),
            "logical_error_rate": float(logical_error_rate),
            "success_rate": float(success_rate),
            "avg_success_prob": float(avg_success_prob),
            "msd_attempts": msd_attempts if msd_attempts is not None else [],
            "circuit_info": {
                "num_qubits": circuit.num_qubits,
                "num_gates": len(pipeline.gates),
                "num_t_gates": len(pipeline.t_gates)
            },
            "visualization_files": [
                f"figures/surface_code_{logical_qubit.logical_qubit_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
                for logical_qubit in pipeline.logical_qubits
            ]
        }

        # Generate a unique simulation ID
        simulation_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Store the result
        simulations[simulation_id] = {
            "result": result,
            "timestamp": timestamp
        }

        return SimulationResponse(
            simulation_id=simulation_id,
            result=result,
            timestamp=timestamp
        )

    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.get("/simulations", response_model=List[SimulationResponse])
async def get_all_simulations():
    """
    Retrieve all simulation results.
    """
    return [
        SimulationResponse(
            simulation_id=sim_id,
            result=sim_data["result"],
            timestamp=sim_data["timestamp"]
        )
        for sim_id, sim_data in simulations.items()
    ]


@router.get("/simulations/{simulation_id}", response_model=SimulationResponse)
async def get_simulation_by_id(simulation_id: str):
    """
    Retrieve a specific simulation result by ID.
    """
    if simulation_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    sim_data = simulations[simulation_id]
    return SimulationResponse(
        simulation_id=simulation_id,
        result=sim_data["result"],
        timestamp=sim_data["timestamp"]
    )


@router.delete("/simulations/{simulation_id}")
async def delete_simulation(simulation_id: str):
    """
    Delete a specific simulation result by ID.
    """
    if simulation_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    del simulations[simulation_id]
    return {"message": f"Simulation {simulation_id} deleted successfully"}


