from qiskit import QuantumCircuit

def validate_qiskit_circuit(circuit: QuantumCircuit) -> bool:
    """
    Validate a Qiskit QuantumCircuit.
    """
    if not isinstance(circuit, QuantumCircuit):
        return False
    if circuit.num_qubits < 1:
        return False
    return True