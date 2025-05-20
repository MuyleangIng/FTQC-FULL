from pydantic import BaseModel
from typing import Optional

class CircuitGate(BaseModel):
    gate: str
    target: int
    control: Optional[int] = None