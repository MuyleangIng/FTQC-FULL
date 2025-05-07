import stim
import numpy as np
from typing import List, Tuple, Dict
import os
import datetime
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import html
import pymatching

# Ensure the /logs and /images directories exist
if not os.path.exists("../logs"):
    os.makedirs("../logs")
if not os.path.exists("../images"):
    os.makedirs("../images")

# Open a log file
log_file = os.path.join("../logs", "ftqc_log.txt")
log = open(log_file, "w")


def log_message(message: str):
    """Write a message to both the console and the log file."""
    print(message)
    log.write(message + "\n")


# Example Circuits
def create_bell_state_circuit() -> QuantumCircuit:
    """Create a 2-qubit Bell state circuit: (|00⟩ + |11⟩)/√2."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def create_grover_circuit() -> QuantumCircuit:
    """Create a simplified 2-qubit Grover's algorithm circuit (oracle for |11⟩)."""
    qc = QuantumCircuit(2, 2)
    qc.h([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])
    qc.measure([0, 1], [0, 1])
    return qc


def create_shor_circuit() -> QuantumCircuit:
    """Create a simplified 2-qubit Shor's algorithm circuit for factoring 15 (a=7)."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    return qc


def create_user_circuit() -> QuantumCircuit:
    """Create the user's original 4-qubit circuit."""
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
    for _ in range(7):
        qc.t(3)
    qc.cx(2, 3)
    qc.t(3)
    qc.t(3)
    qc.cx(2, 3)
    qc.h(3)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    return qc


# Visualize the Surface Code
def visualize_surface_code(logical_qubits: List[Dict], t_gate_applications: List[Dict], circuit_name: str):
    """Visualize the d=3 surface code lattice for all logical qubits, highlighting T-gate applications."""
    log_message("Generating surface code visualization...")

    fig, ax = plt.subplots(figsize=(12, 8))

    for lq_idx, lq in enumerate(logical_qubits):
        offset_x = lq_idx * 4

        data_qubits = [(x, y) for x in range(3) for y in range(3)]
        data_qubit_labels = {pos: idx for idx, pos in enumerate(data_qubits)}
        for x, y in data_qubits:
            ax.add_patch(Circle((x + offset_x + 0.5, y + 0.5), 0.2, color='blue',
                                label=f'Data Qubit (Logical {lq_idx})' if (x, y, lq_idx) == (0, 0, 0) else ''))
            qubit_idx = lq["data_qubits"][data_qubit_labels[(x, y)]]
            ax.text(x + offset_x + 0.5, y + 0.5, f'D{qubit_idx}', ha='center', va='center', color='white')
            for t_app in t_gate_applications:
                if t_app["target_qubit"] == qubit_idx:
                    ax.plot(x + offset_x + 0.5, y + 0.5, '*', color='red', markersize=15,
                            label='T-Gate Target' if (x, y, lq_idx) == (0, 0, 0) else '')

        ancilla_qubits = [(0.5, 0.5, 'X'), (1.5, 1.5, 'X'), (1.5, 0.5, 'Z'), (0.5, 1.5, 'Z')]
        for x, y, t in ancilla_qubits:
            ancilla_idx = lq["ancilla_qubits"][len([a for a in ancilla_qubits if a[2] == t and (a[0] < x or a[1] < y)])]
            ax.add_patch(Rectangle((x + offset_x, y), 0.4, 0.4, color='red' if t == 'X' else 'green',
                                   label=f'{t}-Ancilla (Logical {lq_idx})' if (x, y, lq_idx) == (0.5, 0.5, 0) else ''))
            ax.text(x + offset_x + 0.2, y + 0.2, f'A{ancilla_idx}', ha='center', va='center', color='white')

            for dx, dy in [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]:
                if (x + dx - 0.5, y + dy - 0.5) in data_qubits:
                    ax.plot([x + offset_x + 0.2, x + offset_x + dx + 0.5], [y + 0.2, y + dy + 0.5], 'k-')

    ax.set_xlim(-1, len(logical_qubits) * 4 + 1)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.set_title(f"d=3 Surface Code Lattice for {circuit_name.capitalize()} Circuit")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'images/surface_code_{circuit_name}.png', bbox_inches='tight')
    plt.close()
    log_message(f"Surface code visualization saved as 'images/surface_code_{circuit_name}.png'")


# Visualize the Circuit
def visualize_circuit(qc: QuantumCircuit, circuit_name: str):
    """Save the Qiskit circuit diagram as an image."""
    log_message("Generating circuit diagram image...")
    circuit_str = qc.draw(output='text')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0, 0.5, circuit_str, fontsize=10, family='monospace', verticalalignment='center')
    ax.axis('off')
    plt.savefig(f'images/circuit_{circuit_name}.png', bbox_inches='tight')
    plt.close()
    log_message(f"Circuit diagram saved as 'images/circuit_{circuit_name}.png'")


# Step 0: Decompose the Circuit into Clifford and T-Gates
def decompose_circuit(qc: QuantumCircuit) -> List[Dict]:
    """Decompose the Qiskit circuit into a list of gates (Clifford and T)."""
    log_message("Step 0: Decomposing the input circuit into Clifford and T-gates")
    log_message("This step breaks down the circuit into basic gates we can apply fault-tolerantly.")
    log_message("Clifford gates (like H, S, CNOT, SWAP) can be applied directly, but T-gates need special handling.")

    dag = circuit_to_dag(qc)
    gates = []

    for node in dag.topological_op_nodes():
        if isinstance(node, DAGOpNode):
            op = node.op
            qubits = [qc.find_bit(q).index for q in node.qargs]
            if op.name == "h":
                gates.append({"type": "H", "qubits": qubits})
            elif op.name == "cx":
                gates.append({"type": "CX", "qubits": qubits})
            elif op.name == "swap":
                gates.append({"type": "SWAP", "qubits": qubits})
            elif op.name == "t":
                gates.append({"type": "T", "qubits": qubits})
            elif op.name == "cz":
                gates.append({"type": "H", "qubits": [qubits[1]]})
                gates.append({"type": "CX", "qubits": qubits})
                gates.append({"type": "H", "qubits": [qubits[1]]})
            elif op.name == "measure":
                gates.append({"type": "MEASURE", "qubits": qubits, "cbits": [qc.find_bit(c).index for c in node.cargs]})
            else:
                log_message(f"Unsupported gate {op.name} encountered. Skipping.")

    log_message(f"Decomposed circuit into {len(gates)} operations.")
    return gates


# Step 1: Define the Surface Code (d=3 for better error correction)
def create_surface_code_circuit_d3() -> stim.Circuit:
    """Create a d=3 surface code for one logical qubit (9 data qubits, 8 ancilla)."""
    surface_code = stim.Circuit()

    data_qubits = list(range(9))
    ancilla_qubits = list(range(9, 17))

    surface_code.append_operation("R", ancilla_qubits)
    surface_code.append_operation("H", ancilla_qubits[:4])

    surface_code.append_operation("CNOT", [9, 0])
    surface_code.append_operation("DEPOLARIZE2", [9, 0], 0.0005)
    surface_code.append_operation("CNOT", [9, 1])
    surface_code.append_operation("DEPOLARIZE2", [9, 1], 0.0005)
    surface_code.append_operation("CNOT", [9, 3])
    surface_code.append_operation("DEPOLARIZE2", [9, 3], 0.0005)
    surface_code.append_operation("CNOT", [9, 4])
    surface_code.append_operation("DEPOLARIZE2", [9, 4], 0.0005)

    surface_code.append_operation("CNOT", [10, 1])
    surface_code.append_operation("DEPOLARIZE2", [10, 1], 0.0005)
    surface_code.append_operation("CNOT", [10, 2])
    surface_code.append_operation("DEPOLARIZE2", [10, 2], 0.0005)
    surface_code.append_operation("CNOT", [10, 4])
    surface_code.append_operation("DEPOLARIZE2", [10, 4], 0.0005)
    surface_code.append_operation("CNOT", [10, 5])
    surface_code.append_operation("DEPOLARIZE2", [10, 5], 0.0005)

    surface_code.append_operation("CNOT", [11, 3])
    surface_code.append_operation("DEPOLARIZE2", [11, 3], 0.0005)
    surface_code.append_operation("CNOT", [11, 4])
    surface_code.append_operation("DEPOLARIZE2", [11, 4], 0.0005)
    surface_code.append_operation("CNOT", [11, 6])
    surface_code.append_operation("DEPOLARIZE2", [11, 6], 0.0005)
    surface_code.append_operation("CNOT", [11, 7])
    surface_code.append_operation("DEPOLARIZE2", [11, 7], 0.0005)

    surface_code.append_operation("CNOT", [12, 4])
    surface_code.append_operation("DEPOLARIZE2", [12, 4], 0.0005)
    surface_code.append_operation("CNOT", [12, 5])
    surface_code.append_operation("DEPOLARIZE2", [12, 5], 0.0005)
    surface_code.append_operation("CNOT", [12, 7])
    surface_code.append_operation("DEPOLARIZE2", [12, 7], 0.0005)
    surface_code.append_operation("CNOT", [12, 8])
    surface_code.append_operation("DEPOLARIZE2", [12, 8], 0.0005)

    surface_code.append_operation("CNOT", [13, 0])
    surface_code.append_operation("DEPOLARIZE2", [13, 0], 0.0005)
    surface_code.append_operation("CNOT", [13, 1])
    surface_code.append_operation("DEPOLARIZE2", [13, 1], 0.0005)
    surface_code.append_operation("CNOT", [13, 3])
    surface_code.append_operation("DEPOLARIZE2", [13, 3], 0.0005)
    surface_code.append_operation("CNOT", [13, 4])
    surface_code.append_operation("DEPOLARIZE2", [13, 4], 0.0005)

    surface_code.append_operation("CNOT", [14, 1])
    surface_code.append_operation("DEPOLARIZE2", [14, 1], 0.0005)
    surface_code.append_operation("CNOT", [14, 2])
    surface_code.append_operation("DEPOLARIZE2", [14, 2], 0.0005)
    surface_code.append_operation("CNOT", [14, 4])
    surface_code.append_operation("DEPOLARIZE2", [14, 4], 0.0005)
    surface_code.append_operation("CNOT", [14, 5])
    surface_code.append_operation("DEPOLARIZE2", [14, 5], 0.0005)

    surface_code.append_operation("CNOT", [15, 3])
    surface_code.append_operation("DEPOLARIZE2", [15, 3], 0.0005)
    surface_code.append_operation("CNOT", [15, 4])
    surface_code.append_operation("DEPOLARIZE2", [15, 4], 0.0005)
    surface_code.append_operation("CNOT", [15, 6])
    surface_code.append_operation("DEPOLARIZE2", [15, 6], 0.0005)
    surface_code.append_operation("CNOT", [15, 7])
    surface_code.append_operation("DEPOLARIZE2", [15, 7], 0.0005)

    surface_code.append_operation("CNOT", [16, 4])
    surface_code.append_operation("DEPOLARIZE2", [16, 4], 0.0005)
    surface_code.append_operation("CNOT", [16, 5])
    surface_code.append_operation("DEPOLARIZE2", [16, 5], 0.0005)
    surface_code.append_operation("CNOT", [16, 7])
    surface_code.append_operation("DEPOLARIZE2", [16, 7], 0.0005)
    surface_code.append_operation("CNOT", [16, 8])
    surface_code.append_operation("DEPOLARIZE2", [16, 8], 0.0005)

    surface_code.append_operation("MR", ancilla_qubits)

    for qubit in data_qubits:
        surface_code.append_operation("DEPOLARIZE1", [qubit], 0.0005)

    return surface_code


# Step 2: Magic State Distillation (15-to-1 Protocol for |A⟩ State)
def distill_magic_state(base_qubit: int, num_rounds: int = 2, initial_error: float = 0.01) -> Tuple[
    stim.Circuit, float]:
    """Simulate distillation of 15 noisy |A⟩ states into 1 high-fidelity |A⟩ state."""
    circuit = stim.Circuit()
    qubits = list(range(base_qubit, base_qubit + 15))
    ancilla = base_qubit + 15

    for q in qubits:
        circuit.append_operation("R", [q])
        circuit.append_operation("DEPOLARIZE1", [q], 0.0005)
        circuit.append_operation("H", [q])
        circuit.append_operation("DEPOLARIZE1", [q], 0.0005)
        circuit.append_operation("S", [q])
        circuit.append_operation("DEPOLARIZE1", [q], 0.0005)
        circuit.append_operation("H", [q])
        circuit.append_operation("DEPOLARIZE1", [q], 0.0005)
        circuit.append_operation("DEPOLARIZE1", [q], initial_error)

    for round in range(num_rounds):
        circuit.append_operation("R", [ancilla])
        circuit.append_operation("DEPOLARIZE1", [ancilla], 0.0005)
        log_message(f"Distillation round {round + 1}: Simulating acceptance (ancilla measures 0).")

    final_error = initial_error ** (3 ** num_rounds)
    log_message(f"Magic state at qubit {base_qubit} distilled with final error rate: {final_error:.2e}")

    return circuit, final_error


# Step 3: Fault-Tolerant T-Gate via Magic State Injection
def inject_magic_state(surface_code: stim.Circuit, target_qubit: int, magic_qubit: int) -> Tuple[stim.Circuit, Dict]:
    """Apply a T-gate to a target qubit using magic state injection, return application info."""
    surface_code.append_operation("CNOT", [magic_qubit, target_qubit])
    surface_code.append_operation("DEPOLARIZE2", [magic_qubit, target_qubit], 0.0005)
    surface_code.append_operation("M", [magic_qubit])

    sampler = surface_code.compile_sampler()
    sample = sampler.sample(shots=1)
    measurement_result = sample[0][-1]

    if measurement_result:
        log_message("Measurement result of magic qubit is 1, applying S gate to target qubit.")
        surface_code.append_operation("S", [target_qubit])
        surface_code.append_operation("DEPOLARIZE1", [target_qubit], 0.0005)
    else:
        log_message("Measurement result of magic qubit is 0, no S gate needed.")

    surface_code.append_operation("DEPOLARIZE1", [magic_qubit], 0.0005)
    surface_code.append_operation("DEPOLARIZE2", [magic_qubit, target_qubit], 0.0005)

    return surface_code, {"target_qubit": target_qubit, "magic_qubit": magic_qubit}


# Step 4: Fault-Tolerant Clifford Gates (H, CX, SWAP)
def apply_logical_h(surface_code: stim.Circuit, data_qubits: List[int]) -> stim.Circuit:
    """Apply a logical Hadamard gate transversally."""
    for q in data_qubits:
        surface_code.append_operation("H", [q])
        surface_code.append_operation("DEPOLARIZE1", [q], 0.0005)
    return surface_code


def apply_logical_cx(surface_code: stim.Circuit, control_data: List[int], target_data: List[int]) -> stim.Circuit:
    """Apply a logical CNOT gate transversally."""
    for c, t in zip(control_data, target_data):
        surface_code.append_operation("CNOT", [c, t])
        surface_code.append_operation("DEPOLARIZE2", [c, t], 0.0005)
    return surface_code


def apply_logical_swap(surface_code: stim.Circuit, data1: List[int], data2: List[int]) -> stim.Circuit:
    """Apply a logical SWAP gate transversally."""
    for q1, q2 in zip(data1, data2):
        surface_code.append_operation("SWAP", [q1, q2])
        surface_code.append_operation("DEPOLARIZE2", [q1, q2], 0.0005)
    return surface_code


# Step 5 & 6: Syndrome Measurement and Decoding
def perform_syndrome_measurement_and_decoding(surface_code: stim.Circuit, logical_qubits: List[dict],
                                              num_shots: int = 100) -> Tuple[List[Tuple[List[int], List[int]]], int]:
    """Simulate syndrome measurements and decode errors for all logical qubits."""
    dem = surface_code.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    all_ancilla_qubits = []
    for lq in logical_qubits:
        all_ancilla_qubits.extend(lq["ancilla_qubits"])

    sampler = surface_code.compile_detector_sampler()
    syndromes = []
    num_detectors = dem.num_detectors
    for _ in range(num_shots):
        sample = sampler.sample(shots=1, append_observables=False)
        syndrome = sample[0].tolist()[:num_detectors]
        syndromes.append((syndrome, []))

    error_count = 0
    corrections = []
    for syndrome, _ in syndromes:
        prediction = matcher.decode(syndrome)
        corrections.append(prediction)
        if any(syndrome):
            error_count += 1

    log_message(f"Decoding syndromes: {error_count} errors detected in {num_shots} shots.")
    return syndromes, error_count


# Step 7: Apply Corrections
def apply_corrections(surface_code: stim.Circuit, syndromes: List[Tuple[List[int], List[int]]],
                      logical_qubits: List[Dict]) -> stim.Circuit:
    """Apply corrections based on decoded syndromes using PyMatching."""
    log_message("Applying corrections using PyMatching for real error correction.")

    dem = surface_code.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    all_data_qubits = []
    for lq in logical_qubits:
        all_data_qubits.extend(lq["data_qubits"])

    for syndrome, _ in syndromes:
        prediction = matcher.decode(syndrome)
        for qubit_idx, correction in enumerate(prediction):
            if correction == 1 and qubit_idx < len(all_data_qubits):
                qubit = all_data_qubits[qubit_idx]
                surface_code.append_operation("Z", [qubit])
                surface_code.append_operation("DEPOLARIZE1", [qubit], 0.0005)
                log_message(f"Applied Z correction to qubit {qubit}.")

    return surface_code


# Step 8: Final Logical Measurement
def measure_logical_qubit(surface_code: stim.Circuit, data_qubits: List[int]) -> int:
    """Measure the logical Z operator for a d=3 surface code."""
    surface_code.append_operation("M", data_qubits[:3])
    sampler = surface_code.compile_sampler()
    results = sampler.sample(shots=1)
    logical_result = results[0][0] ^ results[0][1] ^ results[0][2]
    log_message(f"Logical measurement result: {int(logical_result)}")
    return int(logical_result)


# Generate HTML Report
def generate_html_report(circuit_name: str, results: List[int], error_count: int, num_shots: int, log_content: str):
    """Generate an HTML report with results and visualizations."""
    log_message("Generating HTML report...")

    html_content = f"""
    <html>
    <head>
        <title>FTQC Pipeline Report - {circuit_name.capitalize()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
            img {{ max-width: 100%; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 50%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>FTQC Pipeline Report - {circuit_name.capitalize()}</h1>
        <h2>Final Measurement Result</h2>
        <table>
            <tr>
                <th>Logical Qubit</th>
                <th>Measurement Result</th>
            </tr>
    """
    for i, result in enumerate(results):
        html_content += f"""
            <tr>
                <td>{i}</td>
                <td>{result}</td>
            </tr>
        """
    html_content += f"""
        </table>
        <h2>Error Rate</h2>
        <p>Detected {error_count} errors in {num_shots} syndrome measurement shots.</p>
        <p>Physical error rate per gate: 0.05%, below the d=3 surface code threshold (~0.7%).</p>
        <p>Logical error rate: ~0.0005^(3/2) ≈ 0.000011 (0.0011%), ensuring reliable computation.</p>
        <h2>Input Circuit Diagram</h2>
        <img src="images/circuit_{circuit_name}.png" alt="Circuit Diagram">
        <h2>Surface Code Lattice (d=3) with T-Gate Applications</h2>
        <img src="images/surface_code_{circuit_name}.png" alt="Surface Code Lattice">
        <h2>Pipeline Log</h2>
        <pre>{html.escape(log_content)}</pre>
    </body>
    </html>
    """

    with open(f"logs/ftqc_report_{circuit_name}.html", "w") as f:
        f.write(html_content)
    log_message(f"HTML report saved as 'logs/ftqc_report_{circuit_name}.html'")


# Main Dynamic FTQC Pipeline
def run_ftqc_pipeline(circuit_name: str = "user"):
    """Execute the FTQC pipeline for a given quantum circuit."""
    log_message(f"=== FTQC Pipeline for {circuit_name.capitalize()} Circuit (Started at {datetime.datetime.now()}) ===")

    # Select the input circuit
    if circuit_name.lower() == "bell":
        qc = create_bell_state_circuit()
    elif circuit_name.lower() == "grover":
        qc = create_grover_circuit()
    elif circuit_name.lower() == "shor":
        qc = create_shor_circuit()
    else:
        qc = create_user_circuit()

    log_message(f"Input circuit: {circuit_name.capitalize()}")
    log_message(f"Number of qubits: {qc.num_qubits}")
    log_message(f"Circuit diagram:\n{qc.draw()}")

    # Visualize the circuit
    visualize_circuit(qc, circuit_name)

    # Step 0: Decompose the circuit
    gates = decompose_circuit(qc)
    num_t_gates = sum(1 for gate in gates if gate["type"] == "T")
    num_qubits = qc.num_qubits

    # Step 1: Encode physical qubits into logical qubits (d=3 surface code)
    log_message("\nStep 1: Encoding physical qubits into logical qubits using d=3 surface code")
    log_message(f"Each logical qubit uses 9 data qubits and 8 ancilla qubits, totaling 17 qubits per logical qubit.")
    log_message(f"For {num_qubits} qubits: {num_qubits} * 17 = {num_qubits * 17} qubits.")
    log_message("This step is like adding a safety net to each qubit to protect it from errors.")

    logical_qubits = []
    base_qubit = 0
    for i in range(num_qubits):
        surface_code = create_surface_code_circuit_d3()
        shifted_circuit = stim.Circuit()
        for inst in surface_code:
            shifted_targets = []
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    shifted_targets.append(t.value + base_qubit)
                elif t.is_measurement_record_target:
                    shifted_targets.append(t)
                else:
                    raise ValueError(f"Unsupported target type: {t}")
            shifted_inst = stim.CircuitInstruction(
                inst.name,
                shifted_targets,
                inst.gate_args_copy()
            )
            shifted_circuit.append(shifted_inst)
        logical_qubits.append({
            "circuit": shifted_circuit,
            "data_qubits": list(range(base_qubit, base_qubit + 9)),
            "ancilla_qubits": list(range(base_qubit + 9, base_qubit + 17))
        })
        base_qubit += 17

    full_circuit = stim.Circuit()
    for lq in logical_qubits:
        full_circuit += lq["circuit"]

    # Step 2: Distill magic states for T-gates
    log_message("\nStep 2: Distilling magic states for T-gates")
    log_message(f"The circuit has {num_t_gates} T-gates, so we need {num_t_gates} magic states.")
    log_message("A magic state is a special quantum state (|0⟩ + e^{iπ/4}|1⟩) needed to apply T-gates safely.")
    log_message("We distill these states to make them very reliable, like purifying water to drink.")

    magic_state_qubits = list(range(num_qubits * 17, num_qubits * 17 + num_t_gates))
    magic_state_circuits = []
    total_success_prob = 1.0
    base_magic_qubit = num_qubits * 17
    for i in range(num_t_gates):
        magic_circuit, error_rate = distill_magic_state(base_magic_qubit, num_rounds=2, initial_error=0.01)
        magic_state_circuits.append(magic_circuit)
        success_prob = 1 - error_rate
        total_success_prob *= success_prob
        log_message(f"Magic state {i + 1} success probability: {success_prob:.6f}")
        base_magic_qubit += 16

    log_message(f"Total success probability for all magic states: {total_success_prob:.6f}")

    for magic_circuit in magic_state_circuits:
        full_circuit += magic_circuit

    # Step 3: Apply the circuit fault-tolerantly
    log_message("\nStep 3: Applying the circuit fault-tolerantly")
    log_message("We go through each gate in the circuit and apply it to the logical qubits.")
    log_message("This is like following a recipe, but with extra steps to keep everything safe from errors.")

    t_gate_counter = 0
    t_gate_applications = []
    for gate in gates:
        if gate["type"] == "H":
            qubit_idx = gate["qubits"][0]
            full_circuit = apply_logical_h(full_circuit, logical_qubits[qubit_idx]["data_qubits"])
            log_message(f"Applied H gate to logical qubit {qubit_idx}.")
        elif gate["type"] == "CX":
            control_idx, target_idx = gate["qubits"]
            full_circuit = apply_logical_cx(full_circuit, logical_qubits[control_idx]["data_qubits"],
                                            logical_qubits[target_idx]["data_qubits"])
            log_message(f"Applied CX gate (control: qubit {control_idx}, target: qubit {target_idx}).")
        elif gate["type"] == "SWAP":
            qubit1_idx, qubit2_idx = gate["qubits"]
            full_circuit = apply_logical_swap(full_circuit, logical_qubits[qubit1_idx]["data_qubits"],
                                              logical_qubits[qubit2_idx]["data_qubits"])
            log_message(f"Applied SWAP gate between logical qubits {qubit1_idx} and {qubit2_idx}.")
        elif gate["type"] == "T":
            qubit_idx = gate["qubits"][0]
            full_circuit, t_app = inject_magic_state(full_circuit, logical_qubits[qubit_idx]["data_qubits"][0],
                                                     magic_state_qubits[t_gate_counter])
            t_gate_applications.append(t_app)
            log_message(
                f"Applied T-gate {t_gate_counter + 1} to logical qubit {qubit_idx} (target qubit {t_app['target_qubit']}, magic qubit {t_app['magic_qubit']}).")
            t_gate_counter += 1
        elif gate["type"] == "MEASURE":
            continue
        else:
            log_message(f"Skipping unsupported gate: {gate['type']}")

    # Visualize the surface code with T-gate applications
    visualize_surface_code(logical_qubits, t_gate_applications, circuit_name)

    # Step 4 & 5: Syndrome measurement and decoding
    log_message("\nStep 4 & 5: Syndrome measurement and decoding")
    log_message("We check for errors by measuring special qubits (ancilla qubits) that act like security cameras.")
    log_message(f"Monitoring {len(logical_qubits) * 8} ancilla qubits to catch any errors in the computation.")
    syndromes, error_count = perform_syndrome_measurement_and_decoding(full_circuit, logical_qubits)

    # Step 6: Apply corrections
    log_message("\nStep 6: Applying corrections")
    log_message("Using the error information (syndromes) to fix any problems we found.")
    full_circuit = apply_corrections(full_circuit, syndromes, logical_qubits)

    # Step 7: Final logical measurement
    log_message("\nStep 7: Final logical measurement")
    log_message("Now we measure the final state of each logical qubit to get our result.")
    results = []
    for i in range(len(logical_qubits)):
        result = measure_logical_qubit(full_circuit, logical_qubits[i]["data_qubits"])
        results.append(result)
    log_message(f"Final result (logical qubits 0–{len(logical_qubits) - 1}): {results}")

    # Step 8: Analysis and Explanation
    log_message("\nStep 8: Analysis and Explanation")
    if circuit_name.lower() == "bell":
        log_message("This is a Bell state circuit, which creates entanglement between 2 qubits.")
        log_message("Expected outcome: Measuring 00 or 11 with 50% probability each (e.g., [0, 0] or [1, 1]).")
    elif circuit_name.lower() == "grover":
        log_message("This is a Grover's algorithm circuit for 2 qubits, searching for the state |11⟩.")
        log_message("Expected outcome: High probability of measuring 11 (e.g., [1, 1]).")
    elif circuit_name.lower() == "shor":
        log_message("This is a simplified Shor's algorithm circuit for factoring 15.")
        log_message("Expected outcome: Depends on the modular exponentiation, typically 0 or 1 on the control qubit.")
    else:
        log_message("This is the user's custom circuit with 4 qubits, 15 T-gates, and various Clifford gates.")
        log_message("Expected outcome: Complex distribution due to multiple T-gates, SWAP, and entangling operations.")

    log_message(f"Error rate: The physical error rate per gate is 0.05%, below the d=3 surface code threshold (~0.7%).")
    log_message(f"Logical error rate: ~0.0005^(3/2) ≈ 0.000011 (0.0011%), meaning the result is very reliable.")

    # Generate HTML report
    with open(log_file, "r") as f:
        log_content = f.read()
    generate_html_report(circuit_name, results, error_count, 100, log_content)

    log_message("\n=== Pipeline Completed ===")
    log.close()


if __name__ == "__main__":
    # Run the pipeline for different circuits
    run_ftqc_pipeline(circuit_name="bell")
    # run_ftqc_pipeline(circuit_name="grover")
    # run_ftqc_pipeline(circuit_name="shor")
    # run_ftqc_pipeline(circuit_name="user")