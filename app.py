from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from core.pipeline import FTQCPipeline
from api.endpoints import simulate_circuit
import base64
from qiskit import QuantumCircuit

app = FastAPI(title="Fault-Tolerant Quantum Computing API")


class SimulateRequest(BaseModel):
    source_code: str
    iterations: int = 10
    noise: float = 0.001
    distance: int = 3
    rounds: int = 2
    error_rate: float = 0.01
    debug: bool = True


@app.post("/simulate")
async def simulate(request: SimulateRequest) -> Dict[str, Any]:
    """
    Simulate a fault-tolerant quantum circuit from Base64-encoded Python source code.
    """
    try:
        # Decode Base64 source code
        source_code = base64.b64decode(request.source_code).decode('utf-8')

        # Execute source code in a restricted environment
        globals_dict = {'QuantumCircuit': QuantumCircuit}
        locals_dict = {}
        exec(source_code, globals_dict, locals_dict)

        # Extract QuantumCircuit
        qc = locals_dict.get('qc')
        if not isinstance(qc, QuantumCircuit):
            raise ValueError("Source code must define a variable 'qc' of type QuantumCircuit")

        # Run simulation
        result = simulate_circuit(
            circuit=qc,
            iterations=request.iterations,
            noise=request.noise,
            distance=request.distance,
            rounds=request.rounds,
            error_rate=request.error_rate,
            debug=request.debug
        )
        return result
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid Base64 encoding in source_code")
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Syntax error in source code: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing source code: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)