import logging
import stim
import numpy as np
import pymatching


class SurfaceCode:
    def __init__(self, distance, logical_qubit_id, num_qubits, noise=0.001, error_rate=0.01, debug=False):
        self.distance = distance
        self.logical_qubit_id = logical_qubit_id
        self.num_qubits = num_qubits
        self.noise = noise
        self.error_rate = error_rate
        self.debug = debug
        self.debug_logs = []
        self.physical_qubits = (self.distance ** 2) + ((self.distance - 1) ** 2)
        self.stabilizers = self._generate_stabilizers()
        self.num_stabilizers = len(self.stabilizers)
        self.circuit = self._build_surface_code_circuit()
        self.decoder = self._build_decoder()
        self.syndrome_history = []

    def _log_debug(self, message):
        logging.debug(message)
        self.debug_logs.append(message)

    def _generate_stabilizers(self):
        stabilizers = []
        data_qubits = list(range(self.distance ** 2))
        for i in range(self.distance - 1):
            for j in range(self.distance - 1):
                z_stab = [
                    i * self.distance + j,
                    i * self.distance + (j + 1),
                    (i + 1) * self.distance + j,
                    (i + 1) * self.distance + (j + 1)
                ]
                z_stab = [q for q in z_stab if q < self.distance ** 2]
                if len(z_stab) >= 2:
                    stabilizers.append(z_stab)
                x_stab = [
                    i * self.distance + j,
                    i * self.distance + (j + 1) if (j + 1) < self.distance else None,
                    (i + 1) * self.distance + j if (i + 1) < self.distance else None
                ]
                x_stab = [q for q in x_stab if q is not None and q < self.distance ** 2]
                if len(x_stab) >= 2:
                    stabilizers.append(x_stab)
        if len(stabilizers) < 2 and self.distance == 2:
            stabilizers = [[0, 1], [2, 3]]
        if self.debug:
            self._log_debug(
                f"DEBUG: Logical Qubit {self.logical_qubit_id} - Generated {len(stabilizers)} stabilizers: {stabilizers}")
        return stabilizers

    def _build_surface_code_circuit(self):
        circuit = stim.Circuit()
        data_qubits = list(range(self.physical_qubits))
        ancilla_qubits = list(range(self.physical_qubits, self.physical_qubits + self.num_stabilizers))

        for q in data_qubits + ancilla_qubits:
            circuit.append("R", q)

        rounds = 2
        measurement_records = []

        for r in range(rounds):
            if self.debug:
                self._log_debug(
                    f"DEBUG: Logical Qubit {self.logical_qubit_id} - Building round {r + 1} of surface code circuit")
            for q in data_qubits:
                circuit.append("DEPOLARIZE1", q, self.noise)

            for idx, stabilizer in enumerate(self.stabilizers):
                ancilla = ancilla_qubits[idx]
                circuit.append("H", ancilla)
                for data_qubit in stabilizer:
                    circuit.append("CNOT", [ancilla, data_qubit])
                circuit.append("H", ancilla)
                circuit.append("M", ancilla)
                measurement_records.append((r, idx, ancilla))
                circuit.append("R", ancilla)

        for idx in range(self.num_stabilizers):
            round0_m = measurement_records[idx][2]
            round1_m = measurement_records[idx + self.num_stabilizers][2]
            circuit.append("DETECTOR", [stim.target_rec(round0_m - measurement_records[-1][2] - 1),
                                        stim.target_rec(round1_m - measurement_records[-1][2] - 1)])

        if self.debug:
            self._log_debug(f"DEBUG: Logical Qubit {self.logical_qubit_id} - Surface code circuit:\n{circuit}")
        return circuit

    def _build_decoder(self):
        matcher = pymatching.Matching()
        for i in range(self.num_stabilizers - 1):
            matcher.add_edge(i, i + 1, weight=1.0)
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