from qiskit import QuantumCircuit
from core.pipeline import FTQCPipeline
from typing import Dict, Any
import json

def simulate_circuit(circuit: QuantumCircuit, iterations: int, noise: float, distance: int, rounds: int, error_rate: float, debug: bool) -> Dict[str, Any]:
    """
    Simulate a fault-tolerant quantum circuit using the FTQC pipeline.
    """
    pipeline = FTQCPipeline(
        circuit=circuit,
        iterations=iterations,
        noise=noise,
        distance=distance,
        rounds=rounds,
        error_rate=error_rate,
        debug=debug
    )
    # Run the pipeline and get results
    result = pipeline.run()
    if result is None:
        return {"error": "Magic state distillation failed"}
    avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts = result
    # Generate JSON response
    json_response = pipeline.generate_json_response(
        avg_fidelity=avg_fidelity,
        logical_error_rate=logical_error_rate,
        success_rate=success_rate,
        avg_success_prob=avg_success_prob,
        msd_attempts=msd_attempts
    )
    return json.loads(json_response)