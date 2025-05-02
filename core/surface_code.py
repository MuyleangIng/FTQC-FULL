import logging
import stim
import numpy as np
import pymatching


class SurfaceCode:
    def __init__(self, distance, logical_qubit_id, num_qubits, noise=0.05, error_rate=0.01, debug=False):
        self.distance = distance
        self.logical_qubit_id = logical_qubit_id
        self.num_qubits = num_qubits
        self.noise = noise
        self.error_rate = error_rate
        self.debug = debug
        self.debug_logs = []
        self.physical_qubits = (self.distance ** 2)  # Data qubits only
        self.stabilizers = self._generate_stabilizers()
        self.num_stabilizers = len(self.stabilizers)
        self.ancilla_qubits = self.num_stabilizers  # One ancilla per stabilizer
        self.circuit = self._build_surface_code_circuit()
        self.decoder = self._build_decoder()
        self.syndrome_history = []

    def _log_debug(self, message):
        logging.debug(message)
        self.debug_logs.append(message)

    def _generate_stabilizers(self):
        stabilizers = []
        d = self.distance
        # For distance-3: 3x3 grid = 9 data qubits
        # Z-stabilizers (plaquettes): 4 in a 2x2 grid
        for i in range(d - 1):
            for j in range(d - 1):
                z_stab = [
                    i * d + j,  # Top-left
                    i * d + (j + 1),  # Top-right
                    (i + 1) * d + j,  # Bottom-left
                    (i + 1) * d + (j + 1)  # Bottom-right
                ]
                z_stab = [q for q in z_stab if q < d * d]
                if len(z_stab) >= 2:  # Ensure at least 2 qubits for a stabilizer
                    stabilizers.append(('Z', z_stab))

        # X-stabilizers (vertices): 4 at the centers of plaquettes
        for i in range(d):
            for j in range(d):
                if (i + j) % 2 == 1:  # Place X-stabilizers at vertices (alternate positions)
                    x_stab = []
                    if i > 0:  # Up
                        x_stab.append((i - 1) * d + j)
                    if i < d - 1:  # Down
                        x_stab.append((i + 1) * d + j)
                    if j > 0:  # Left
                        x_stab.append(i * d + (j - 1))
                    if j < d - 1:  # Right
                        x_stab.append(i * d + (j + 1))
                    x_stab = [q for q in x_stab if q < d * d]
                    if len(x_stab) >= 2:
                        stabilizers.append(('X', x_stab))

        if self.debug:
            self._log_debug(
                f"DEBUG: Logical Qubit {self.logical_qubit_id} - Generated {len(stabilizers)} stabilizers: {stabilizers}")
        return stabilizers

    def _build_surface_code_circuit(self):
        circuit = stim.Circuit()
        data_qubits = list(range(self.physical_qubits))
        ancilla_qubits = list(range(self.physical_qubits, self.physical_qubits + self.ancilla_qubits))

        # Initialize all qubits
        for q in data_qubits + ancilla_qubits:
            circuit.append("R", q)

        rounds = 2
        measurement_indices = [[] for _ in range(self.num_stabilizers)]  # Track measurement indices per stabilizer
        measurement_count = 0

        for r in range(rounds):
            if self.debug:
                self._log_debug(
                    f"DEBUG: Logical Qubit {self.logical_qubit_id} - Building round {r + 1} of surface code circuit")
            # Apply noise to data qubits
            for q in data_qubits:
                circuit.append("DEPOLARIZE1", q, self.noise)
            if self.debug:
                self._log_debug(f"DEBUG: Applied DEPOLARIZE1 noise with probability {self.noise} to data qubits")

            # Measure stabilizers
            for idx, (stab_type, qubits) in enumerate(self.stabilizers):
                ancilla = ancilla_qubits[idx]
                if stab_type == 'Z':
                    # Z-stabilizer: CNOT-based measurement
                    for data_qubit in qubits:
                        circuit.append("CNOT", [ancilla, data_qubit])
                elif stab_type == 'X':
                    # X-stabilizer: Hadamard + CNOT + Hadamard
                    circuit.append("H", ancilla)
                    for data_qubit in qubits:
                        circuit.append("CNOT", [ancilla, data_qubit])
                    circuit.append("H", ancilla)
                # Measure ancilla
                circuit.append("M", ancilla)
                measurement_indices[idx].append(measurement_count)
                measurement_count += 1
                circuit.append("R", ancilla)

        # Define detectors (compare consecutive rounds for each stabilizer)
        for idx in range(self.num_stabilizers):
            if len(measurement_indices[idx]) >= 2:
                # Detector compares measurements from round 0 and round 1
                rec0 = measurement_indices[idx][0]
                rec1 = measurement_indices[idx][1]
                # Relative to the last measurement
                total_measurements = measurement_count
                circuit.append("DETECTOR", [
                    stim.target_rec(rec0 - total_measurements),
                    stim.target_rec(rec1 - total_measurements)
                ])

        if self.debug:
            self._log_debug(f"DEBUG: Logical Qubit {self.logical_qubit_id} - Surface code circuit:\n{circuit}")
        return circuit

    def _build_decoder(self):
        matcher = pymatching.Matching()
        # Connect stabilizers across rounds (temporal edges)
        for i in range(self.num_stabilizers):
            matcher.add_edge(i, i, weight=1.0)
        # Add spatial edges between neighboring stabilizers
        for i in range(self.num_stabilizers):
            for j in range(i + 1, self.num_stabilizers):
                if set(self.stabilizers[i][1]) & set(self.stabilizers[j][1]):  # Share a qubit
                    matcher.add_edge(i, j, weight=1.0)
        if self.num_stabilizers > 0:
            matcher.add_boundary_edge(0, weight=1.0)
        if self.num_stabilizers > 1:
            matcher.add_boundary_edge(self.num_stabilizers - 1, weight=1.0)
        if self.debug:
            self._log_debug(
                f"DEBUG: Logical Qubit {self.logical_qubit_id} - Decoder graph set up with {self.num_stabilizers} detectors")
        return matcher

    def measure_syndrome(self):
        if self.debug:
            self._log_debug(f"DEBUG: Logical Qubit {self.logical_qubit_id} - Measuring syndrome")
        sampler = self.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=1, append_observables=False)
        if samples.shape[1] != self.num_stabilizers:
            logging.warning(
                f"Logical Qubit {self.logical_qubit_id} - Unexpected number of detectors: {samples.shape[1]}, expected {self.num_stabilizers}")
            syndrome = [0] * self.num_stabilizers
        else:
            syndrome = samples[0].tolist()
        if self.debug:
            self._log_debug(f"DEBUG: Logical Qubit {self.logical_qubit_id} - Raw syndrome samples: {samples}")
        simplified_syndrome = [1 if any(syndrome) else 0]
        if self.debug:
            self._log_debug(
                f"DEBUG: Logical Qubit {self.logical_qubit_id} - Raw syndrome: {syndrome}, Simplified: {simplified_syndrome}")
        self.syndrome_history.append(
            {"iteration": len(self.syndrome_history) + 1, "syndrome": syndrome, "simplified": simplified_syndrome})
        return simplified_syndrome, syndrome

    def apply_correction(self, syndrome):
        if self.debug:
            self._log_debug(
                f"DEBUG: Logical Qubit {self.logical_qubit_id} - Applying correction for syndrome: {syndrome}")
        if len(syndrome) != self.num_stabilizers:
            logging.error(
                f"Logical Qubit {self.logical_qubit_id} - Invalid syndrome length: {len(syndrome)}, expected {self.num_stabilizers}")
            return [0] * self.physical_qubits
        correction = self.decoder.decode(syndrome)
        if self.debug:
            self._log_debug(f"DEBUG: Logical Qubit {self.logical_qubit_id} - Correction: {correction}")
        return correction.tolist() if isinstance(correction, np.ndarray) else correction