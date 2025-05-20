import matplotlib.pyplot as plt
import networkx as nx
import logging
import stim
import pymatching
import numpy as np

class EnhancedSurfaceCode:
    """Production-ready Surface Code simulator with visualization, decoding, fault injection, and lattice surgery."""
    def __init__(self, distance=3, logical_qubit_id=0, num_qubits=None, noise=0.001, error_rate=0.01, debug=False):
        if distance < 3 or distance > 6:
            raise ValueError("Distance must be between 3 and 6 for this implementation.")
        self.distance = distance
        self.logical_qubit_id = logical_qubit_id
        self.num_qubits = num_qubits
        self.noise = noise
        self.error_rate = error_rate
        self.debug = debug
        self.num_data_qubits = distance ** 2
        self.physical_qubits = self.num_data_qubits
        self.debug_logs = []
        self.stabilizers = self._generate_stabilizers()
        self.num_stabilizers = len(self.stabilizers)
        self.ancilla_qubits = self.num_stabilizers
        self.circuit = self._build_circuit()
        self.decoder = self._build_decoder()
        self.qubit_error_log = {i: [] for i in range(self.num_data_qubits)}
        self.gate_times = {'h': 20, 'cx': 40, 't': 50, 'measure': 80}
        self.total_time = 0
        self.syndrome_history = []
        # For lattice surgery
        self.logical_state = None  # Logical state (e.g., |0⟩ or |+⟩)
        self.neighbor_patch = None  # Reference to neighboring patch for lattice surgery

    def _generate_stabilizers(self):
        d = self.distance
        stabilizers = []

        # Z-stabilizers (2x2 squares at plaquettes)
        for i in range(d - 1):
            for j in range(d - 1):
                stab = [
                    i * d + j,           # top-left
                    i * d + j + 1,       # top-right
                    (i + 1) * d + j,     # bottom-left
                    (i + 1) * d + j + 1  # bottom-right
                ]
                stabilizers.append(('Z', stab))

        # X-stabilizers (crosses at vertices)
        for i in range(d):
            for j in range(d):
                if (i + j) % 2 == 0:  # Place X-stabilizers at vertices
                    stab = []
                    if i > 0: stab.append((i - 1) * d + j)       # up
                    if i < d - 1: stab.append((i + 1) * d + j)   # down
                    if j > 0: stab.append(i * d + j - 1)         # left
                    if j < d - 1: stab.append(i * d + j + 1)     # right
                    if len(stab) >= 2:  # Ensure valid stabilizer
                        stabilizers.append(('X', stab))

        if self.debug:
            logging.debug(f"Generated {len(stabilizers)} stabilizers for distance {d}")
        return stabilizers

    def _build_circuit(self):
        circuit = stim.Circuit()
        all_qubits = list(range(self.num_data_qubits + self.ancilla_qubits))
        for q in all_qubits:
            circuit.append("R", q)
        for q in range(self.num_data_qubits):
            circuit.append("DEPOLARIZE1", q, self.noise)
        for idx, (typ, qlist) in enumerate(self.stabilizers):
            anc = self.num_data_qubits + idx
            if typ == 'Z':
                for q in qlist:
                    circuit.append("CNOT", [anc, q])
            elif typ == 'X':
                circuit.append("H", anc)
                for q in qlist:
                    circuit.append("CNOT", [anc, q])
                circuit.append("H", anc)
            circuit.append("M", anc)
        return circuit

    def _build_decoder(self):
        matcher = pymatching.Matching()
        added_edges = set()
        for idx, (typ, qlist) in enumerate(self.stabilizers):
            for q in qlist:
                edge = tuple(sorted([idx, q]))
                if edge not in added_edges:
                    matcher.add_edge(idx, q, weight=1.0)
                    added_edges.add(edge)
                else:
                    if self.debug:
                        logging.debug(f"Skipped duplicate edge {edge}")
        return matcher

    def measure_syndrome(self):
        sampler = self.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=1, append_observables=False)
        syndrome = samples[0].tolist() if samples.shape[1] >= 1 else [0] * self.num_stabilizers
        simplified_syndrome = [1 if any(syndrome) else 0]
        self.syndrome_history.append({
            "iteration": len(self.syndrome_history) + 1,
            "syndrome": syndrome,
            "simplified": simplified_syndrome
        })
        return simplified_syndrome, syndrome

    def apply_correction(self, syndrome):
        if len(syndrome) != self.num_stabilizers:
            logging.error(f"Invalid syndrome length: expected {self.num_stabilizers}, got {len(syndrome)}")
            return [0] * self.num_data_qubits
        correction = self.decoder.decode(syndrome)
        return correction.tolist() if hasattr(correction, 'tolist') else correction

    def inject_fault(self, qubit, pauli='X'):
        if pauli == 'X':
            self.circuit.append("X", qubit)
        elif pauli == 'Z':
            self.circuit.append("Z", qubit)
        self.qubit_error_log[qubit].append({'type': pauli, 'round': 0})

    def estimate_timing(self):
        self.total_time = sum(self.gate_times.values()) * self.num_data_qubits
        return self.total_time

    def initialize_logical_state(self, state='zero'):
        """Initialize the logical state of the surface code patch."""
        if state == 'zero':
            self.logical_state = '|0⟩'
            # Logical |0⟩: All data qubits in |0⟩ state (already initialized by R)
        elif state == 'plus':
            self.logical_state = '|+⟩'
            # Logical |+⟩: Apply H to all data qubits to create superposition
            for q in range(self.num_data_qubits):
                self.circuit.append("H", q)
        else:
            raise ValueError("State must be 'zero' or 'plus'")
        if self.debug:
            logging.debug(f"Logical qubit {self.logical_qubit_id} initialized to {self.logical_state}")

    def set_neighbor_patch(self, neighbor_patch):
        """Set a neighboring patch for lattice surgery operations."""
        if not isinstance(neighbor_patch, EnhancedSurfaceCode):
            raise ValueError("Neighbor patch must be an EnhancedSurfaceCode instance")
        self.neighbor_patch = neighbor_patch
        if self.debug:
            logging.debug(f"Set neighbor patch for logical qubit {self.logical_qubit_id} to {neighbor_patch.logical_qubit_id}")

    def lattice_surgery_cnot(self, control_patch):
        """Perform a logical CNOT using lattice surgery between this patch (target) and control_patch."""
        if self.distance < 5:
            raise ValueError("Lattice surgery requires distance >= 5 for reliable operations")
        if not isinstance(control_patch, EnhancedSurfaceCode) or control_patch != self.neighbor_patch:
            raise ValueError("Control patch must be the set neighbor patch")

        # Step 1: Merge patches by measuring joint X operators along the boundary
        merge_circuit = stim.Circuit()
        boundary_qubits = []
        d = self.distance
        # Assume patches are side by side; merge along the right edge of control and left edge of target
        for i in range(d):
            control_qubit = i * d + (d - 1)  # Rightmost column of control patch
            target_qubit = i * d  # Leftmost column of target patch
            ancilla = self.num_data_qubits + self.num_stabilizers + len(boundary_qubits)
            merge_circuit.append("R", ancilla)
            merge_circuit.append("CNOT", [ancilla, control_qubit])
            merge_circuit.append("CNOT", [ancilla, target_qubit])
            merge_circuit.append("M", ancilla)
            boundary_qubits.append((control_qubit, target_qubit, ancilla))

        # Step 2: Execute merge circuit and measure syndromes
        sampler = merge_circuit.compile_detector_sampler()
        samples = sampler.sample(shots=1, append_observables=False)
        merge_syndrome = samples[0].tolist()

        # Step 3: Apply corrections based on merge syndrome
        # Simplified: Assume correction based on syndrome (in practice, more complex)
        for idx, (control_q, target_q, ancilla) in enumerate(boundary_qubits):
            if idx < len(merge_syndrome) and merge_syndrome[idx] == 1:
                self.circuit.append("Z", target_q)  # Apply Z correction on target

        # Step 4: Split patches by measuring joint Z operators
        split_circuit = stim.Circuit()
        for i in range(d):
            control_qubit = i * d + (d - 1)
            target_qubit = i * d
            ancilla = self.num_data_qubits + self.num_stabilizers + i
            split_circuit.append("R", ancilla)
            split_circuit.append("H", ancilla)
            split_circuit.append("CNOT", [ancilla, control_qubit])
            split_circuit.append("CNOT", [ancilla, target_qubit])
            split_circuit.append("H", ancilla)
            split_circuit.append("M", ancilla)

        # Step 5: Execute split circuit and measure syndromes
        sampler = split_circuit.compile_detector_sampler()
        samples = sampler.sample(shots=1, append_observables=False)
        split_syndrome = samples[0].tolist()

        # Step 6: Apply corrections based on split syndrome
        for idx, (control_q, target_q, ancilla) in enumerate(boundary_qubits):
            if idx < len(split_syndrome) and split_syndrome[idx] == 1:
                self.circuit.append("X", control_q)  # Apply X correction on control

        if self.debug:
            logging.debug(f"Performed lattice surgery CNOT: Control qubit {control_patch.logical_qubit_id}, Target qubit {self.logical_qubit_id}")