from qiskit import QuantumCircuit, transpile
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

# === Step 1: Build a mock version of Shor's algorithm circuit ===
n_count = 4  # Number of counting qubits
qc = QuantumCircuit(7, 4)  # 4 counting qubits + 3 work qubits (simplified)

# Apply Hadamard to counting qubits
for q in range(n_count):
    qc.h(q)

# Mock modular exponentiation (not real modexp but has CX and CCX to trigger T gates later)
qc.cx(3, 4)
qc.cx(2, 5)
qc.ccx(1, 5, 6)
qc.ccx(0, 4, 6)

# Inverse QFT definition
def qft_dagger(circ, n):
    for qubit in range(n // 2):
        circ.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            circ.cp(-3.14159 / float(2 ** (j - m)), m, j)
        circ.h(j)

# Apply inverse QFT
qft_dagger(qc, n_count)

# Measure counting qubits
qc.measure(range(n_count), range(n_count))

# === Step 2: Transpile to Clifford+T-compatible gates ===
t_qc = transpile(
    qc,
    basis_gates=['h', 't', 'tdg', 's', 'sdg', 'x', 'cx', 'measure', 'swap'],
    optimization_level=1
)

# === Step 3: Display counts of T and other gates ===
print("=== Transpiled Gate Counts ===")
print(t_qc.count_ops())

# === Step 4: Draw and save the transpiled circuit ===
t_qc.draw(output='mpl')
plt.title("Transpiled Circuit with T Gates (Mock Shor's Algorithm)")
plt.tight_layout()
plt.savefig("shor_transpiled_circuit.png")
plt.show()
