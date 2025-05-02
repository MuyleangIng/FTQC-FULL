
import matplotlib.pyplot as plt
import networkx as nx
import logging
import stim
import pymatching
import numpy as np

class EnhancedSurfaceCode:
    """Production-ready Surface Code simulator with visualization, decoding, and fault injection."""
    def __init__(self, distance=3, logical_qubit_id=0, num_qubits=None, noise=0.001, error_rate=0.01, debug=False):
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

    def _generate_stabilizers(self):
        d = self.distance
        stabilizers = []
        for i in range(d - 1):
            for j in range(d - 1):
                stab = [i * d + j, i * d + j + 1, (i + 1) * d + j, (i + 1) * d + j + 1]
                stabilizers.append(('Z', stab))
        for i in range(d):
            for j in range(d):
                if (i + j) % 2 == 1:
                    stab = []
                    if i > 0: stab.append((i - 1) * d + j)
                    if i < d - 1: stab.append((i + 1) * d + j)
                    if j > 0: stab.append(i * d + j - 1)
                    if j < d - 1: stab.append(i * d + j + 1)
                    if len(stab) >= 2:
                        stabilizers.append(('X', stab))
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
        for i in range(self.num_stabilizers):
            matcher.add_edge(i, i)
            matcher.add_boundary_edge(i)
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

    def visualize(self):
        d = self.distance
        G = nx.Graph()
        pos = {}
        for i in range(d):
            for j in range(d):
                idx = i * d + j
                G.add_node(idx)
                pos[idx] = (j, -i)
        for typ, qlist in self.stabilizers:
            for q1 in qlist:
                for q2 in qlist:
                    if q1 != q2:
                        G.add_edge(q1, q2)
        # plt.figure(figsize=(6, 6))
        # nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=600, font_size=10)
        # plt.title(f"Surface Code (Logical Qubit {self.logical_qubit_id}, Distance={d})")
        # plt.axis('off')
        # plt.show()
