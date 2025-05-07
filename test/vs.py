from qiskit import QuantumCircuit,  transpile, assemble
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error


# Step 1: Prepare magic state (|A> = T|+>)
def prepare_magic_state():
    magic = QuantumCircuit(1, name='magic')
    magic.h(0)
    magic.t(0)
    return magic


# Step 2: Build teleportation circuit to inject T-gate
def build_ftqc_circuit():
    # Registers: data (logical qubit), ancilla (magic), classical bit
    qc = QuantumCircuit(2, 1)

    # Step 1: Initialize |+> on data qubit (like H|0>)
    qc.h(0)

    # Step 2: Prepare ancilla = magic state
    qc.h(1)
    qc.t(1)

    # Step 3: Teleportation injection (CX, H, measure, conditional Z)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.z(1).c_if(qc.cregs[0], 1)

    # Final H (to complete H-T-H from original circuit)
    qc.h(1)

    # Measure logical output
    qc.measure_all()

    return qc


# Step 3: Add noise model (depolarizing noise on gates)
def get_noise_model():
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.01, 1)
    error_2q = depolarizing_error(0.02, 2)

    noise_model.add_all_qubit_quantum_error(error_1q, ['h', 't', 'z'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

    return noise_model


# Step 4: Simulate and visualize
def run_simulation():
    qc = build_ftqc_circuit()
    noise_model = get_noise_model()
    backend = Aer.get_backend('qasm_simulator')

    transpiled = transpile(qc, backend)
    qobj = assemble(transpiled, shots=1024)
    result = backend.run(qobj, noise_model=noise_model).result()
    counts = result.get_counts()

    print("\n🧠 Final Logical Result (with FTQ-style protection):")
    print(counts)

    plot_histogram(counts)
    plt.title("FTQC-Style T-Gate Injection (Teleportation + Noise)")
    plt.show()


# Run the full pipeline
if __name__ == "__main__":
    run_simulation()
