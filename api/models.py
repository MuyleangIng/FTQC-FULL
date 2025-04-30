from pydantic import BaseModel
from typing import Optional

class CircuitGate(BaseModel):
    gate: str  # e.g., "h", "t", "cx", "swap", "measure"
    target: int
    control: Optional[int] = None  # For two-qubit gates like cx, swap