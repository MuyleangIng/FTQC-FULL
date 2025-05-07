from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

# Create the 4-qubit quantum circuit
qc = QuantumCircuit(4, 4)
qc.h(0)
qc.cx(0, 1)
qc.t(0)
qc.t(1)
qc.h(2)
qc.h(3)
qc.cx(2, 0)
qc.cx(3, 1)
qc.swap(2, 3)

qc.t(3)
qc.t(3)
qc.t(3)
qc.cx(2, 3)
qc.t(3)
qc.cx(2, 3)

qc.h(2)

for _ in range(7): qc.t(3)
qc.cx(2, 3)
qc.t(3)
qc.t(3)
qc.cx(2, 3)

qc.h(3)
qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

# Save the circuit as an image
circuit_drawer(qc, output='mpl', filename='quantum_circuit.png')
