from qiskit import QuantumCircuit

def create_shor_circuit():
    qc = QuantumCircuit(6, 2)

    # Step 1: Prepare counting qubits in superposition
    qc.h(0)
    qc.h(1)

    # Step 2: Initialize work qubits to |1> (0001)
    qc.x(2)

    # Step 3: Controlled-U^1 (multiplication by 7 mod 15, controlled by q_0)
    # For y=1 (0001), set to 7 (0111)
    # If q_0=1 and q_2=1, q_3=q_4=q_5=0, set q_3=1, q_4=1
    qc.cx(0, 3)  # If q_0=1, set q_3=1
    qc.cx(0, 4)  # If q_0=1, set q_4=1

    # Step 4: Controlled-U^2 (multiplication by 4 mod 15, controlled by q_1)
    # For y=1 (0001), set to 4 (0100)
    # If q_1=1 and q_2=1, q_3=q_4=q_5=0, set q_3=1, clear q_2
    qc.cx(1, 3)  # If q_1=1, set q_3=1
    qc.cx(1, 2)  # If q_1=1, clear q_2

    # Step 5: Inverse QFT on q_0, q_1
    qc.swap(0, 1)
    qc.t(1)  # T†
    qc.t(1)
    qc.t(1)
    qc.cx(0, 1)
    qc.t(1)  # T
    qc.cx(0, 1)
    qc.h(0)
    qc.t(1)  # (T†)^2
    qc.t(1)
    qc.t(1)
    qc.t(1)
    qc.t(1)
    qc.t(1)
    qc.t(1)
    qc.cx(0, 1)
    qc.t(1)  # T^2
    qc.t(1)
    qc.cx(0, 1)
    qc.h(1)

    # Step 6: Measure counting qubits
    qc.measure([0, 1], [0, 1])

    return qc

# Create and save the circuit
if __name__ == "__main__":
    circuit = create_shor_circuit()
    print(circuit)