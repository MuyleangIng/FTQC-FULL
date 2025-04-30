# Placeholder for API-specific utilities
def validate_circuit(circuit: list) -> bool:
    """
    Validate a circuit specification.
    """
    valid_gates = {"h", "t", "cx", "swap", "measure"}
    for gate in circuit:
        if gate["gate"] not in valid_gates:
            return False
        if not isinstance(gate.get("target"), int):
            return False
        if gate.get("control") is not None and not isinstance(gate["control"], int):
            return False
    return True