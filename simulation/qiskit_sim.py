from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError

from utils import logging


def run_qiskit_simulation(circuit: QuantumCircuit, noise: float = 0.001, shots: int = 1024, method='statevector') -> dict:
    """
    Parameters
    ----------
    circuit: QuantumCircuit from qiskit
    noise: float depolarizing error
    shots: int number of shots
    method: configurable with ['statevector', 'tensor_network', ....] Ref: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html
            tensor_network mode is only available when nvidia-gpu is detected and through qiskit-aer-gpu.
    Returns
    -------
    """
    noise_model = NoiseModel()
    error_1q = depolarizing_error(noise, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['h', 't', 's'])
    error_2q = depolarizing_error(noise * 2, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'swap'])

    simulator = AerSimulator(noise_model=noise_model, method=method)

    try:
        simulator.set_options(device="GPU")
    except QiskitError as e:
        logging.warning(f"qiskit_sim:run_qiskit_simulation, executing on GPU is not supported on this device with error {e}")

    if not any(instr.operation.name == 'measure' for instr in circuit):
        circuit.measure_all()

    result = simulator.run(circuit, shots=shots).result()
    counts = result.get_counts()
    return counts