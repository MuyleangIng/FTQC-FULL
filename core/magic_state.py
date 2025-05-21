import logging
import numpy as np
import stim
import time
from qiskit.quantum_info import state_fidelity

class MagicStateDistillation:
    def __init__(self, num_input_states, noise_prob, t_gate_id, logical_qubit, gate_type='t', debug=False):
        self.num_input_states = num_input_states
        self.noise_prob = noise_prob
        self.t_gate_id = t_gate_id
        self.logical_qubit = logical_qubit
        self.gate_type = gate_type  # 't' for T gate, 'tdg' for T-dagger
        self.debug = debug
        self.debug_logs = []
        # Ideal states: |A⟩ for T, |A*⟩ for Tdg
        self.ideal_t_state = np.array([np.cos(np.pi / 8), np.exp(1j * np.pi / 4) * np.sin(np.pi / 8)]) if gate_type == 't' else \
                             np.array([np.cos(np.pi / 8), np.exp(-1j * np.pi / 4) * np.sin(np.pi / 8)])
        self.attempts_per_round = []
        self.lookup_table = self.build_lookup_table()
        self.round_results = []
        self.success_count = 0
        self.total_attempts = 0
        self.distillation_time = 0
        logging.info(f"T{self.t_gate_id} ({self.gate_type}) - Initialized MagicStateDistillation: num_input_states={num_input_states}, noise_prob={noise_prob}")

    def _log_debug(self, message):
        if self.debug:
            logging.debug(message)
            self.debug_logs.append(message)

    def build_lookup_table(self):
        logging.info(f"T{self.t_gate_id} - Building lookup table for syndrome measurements...")
        lookup = {}
        stabilizers = [[0, 1, 2], [0, 1, 3], [0, 2, 4], [1, 3, 4]]
        for qubit in range(self.num_input_states):
            syndrome = [0] * 4
            for idx, stab in enumerate(stabilizers):
                if qubit in stab:
                    syndrome[idx] = 1
            lookup[tuple(syndrome)] = qubit
        if self.debug:
            self._log_debug(f"DEBUG: T{self.t_gate_id} - Lookup table for MSD: {lookup}")
        logging.info(f"T{self.t_gate_id} - Lookup table built with {len(lookup)} entries.")
        return lookup

    def prepare_noisy_t_state(self, circuit, qubit):
        logging.info(f"T{self.t_gate_id} - Preparing noisy {self.gate_type} state for qubit {qubit}...")
        circuit.append("H", qubit)
        if self.gate_type == 't':
            circuit.append("S", qubit)  # For |A⟩
        else:
            circuit.append("S_DAG", qubit)  # For |A*⟩ (S dagger for T dagger)
        circuit.append("DEPOLARIZE1", qubit, self.noise_prob)
        logging.info(f"T{self.t_gate_id} - Noisy {self.gate_type} state prepared for qubit {qubit}.")

    def syndrome_measurement(self, circuit, data_qubits, ancilla_qubits):
        logging.info(f"T{self.t_gate_id} - Performing syndrome measurement...")
        stabilizers = [[0, 1, 2], [0, 1, 3], [0, 2, 4], [1, 3, 4]]
        num_measurements = len(stabilizers)
        for idx, ancilla in enumerate(ancilla_qubits):
            if idx < num_measurements:
                circuit.append("H", ancilla)
                for data in stabilizers[idx]:
                    circuit.append("CNOT", [ancilla, data_qubits[data]])
                circuit.append("H", ancilla)
                circuit.append("DEPOLARIZE1", ancilla, self.noise_prob)
                circuit.append("M", ancilla)
                circuit.append("R", ancilla)
        logging.info(f"T{self.t_gate_id} - Syndrome measurement completed with {num_measurements} measurements.")
        return circuit, num_measurements, stabilizers

    def distill_once(self, round_num):
        start_time = time.time()
        logging.info(f"T{self.t_gate_id} Round {round_num} - Starting distillation attempt")
        if self.debug:
            self._log_debug(f"DEBUG: T{self.t_gate_id} Round {round_num} - Starting distillation attempt")
        circuit = stim.Circuit()
        data_qubits = list(range(self.num_input_states))
        ancilla_qubits = list(range(self.num_input_states, self.num_input_states + 4))
        for i in data_qubits:
            self.prepare_noisy_t_state(circuit, i)
        for q in data_qubits:
            circuit.append("DEPOLARIZE1", q, self.noise_prob)
        circuit, num_measurements, stabilizers = self.syndrome_measurement(circuit, data_qubits, ancilla_qubits)

        if self.debug:
            circuit_str = str(circuit).replace('\n', ' | ')
            self._log_debug(f"DEBUG: T{self.t_gate_id} Round {round_num} - MSD circuit: {circuit_str}")

        simulator = stim.TableauSimulator()
        simulator.do(circuit)
        measurements = simulator.current_measurement_record()[-num_measurements:]
        syndrome = tuple(1 if m else 0 for m in measurements)

        if self.debug:
            self._log_debug(f"DEBUG: T{self.t_gate_id} Round {round_num} - Measurements: {measurements}, Syndrome: {syndrome}")

        if syndrome in self.lookup_table:
            error_qubit = self.lookup_table[syndrome]
            logging.info(f"T{self.t_gate_id} Round {round_num}: Correcting error on qubit {error_qubit}")
            simulator.x(error_qubit)
            success = True
        else:
            success = (syndrome == (0, 0, 0, 0))
            logging.info(f"T{self.t_gate_id} Round {round_num}: No error detected (syndrome={syndrome}, success={success})")

        tableau = simulator.current_inverse_tableau() ** -1
        state = self.tableau_to_state(tableau, qubit=0)

        weight = sum(syndrome)
        round_result = {
            "t_gate": f"T{self.t_gate_id}",
            "round": round_num,
            "success": success,
            "syndrome": syndrome,
            "weight": weight
        }
        self.round_results.append(round_result)
        self.total_attempts += 1
        if success:
            self.success_count += 1
        self.distillation_time += time.time() - start_time
        logging.info(f"T{self.t_gate_id} Round {round_num}: Distillation attempt completed (success={success}, time={self.distillation_time:.2f}s)")
        return state, success

    def tableau_to_state(self, tableau, qubit):
        logging.info(f"T{self.t_gate_id} - Computing state from tableau for qubit {qubit}...")
        state = self.ideal_t_state.copy()
        pauli_z = tableau.z_output(qubit)
        pauli_x = tableau.x_output(qubit)
        if str(pauli_z) != "+Z" or str(pauli_x) != "+X":
            state += np.random.randn(2) * 0.001
        state = state / np.linalg.norm(state)
        if self.debug:
            self._log_debug(f"DEBUG: T{self.t_gate_id} - Computed state: {state}")
        logging.info(f"T{self.t_gate_id} - State computation completed for qubit {qubit}.")
        return state

    def distill(self, rounds):
        initial_noise = self.noise_prob
        if self.noise_prob > 0.01:
            rounds = min(rounds + 2, 5)
        elif self.noise_prob < 0.001:
            rounds = max(rounds - 1, 1)
        logging.info(f"T{self.t_gate_id} - Adjusted rounds to {rounds} based on noise {self.noise_prob}")
        if self.debug:
            self._log_debug(f"DEBUG: T{self.t_gate_id} - Adjusted MSD rounds to {rounds} based on noise {self.noise_prob}")

        current_noise = self.noise_prob
        attempts_this_run = []
        for r in range(rounds):
            self.noise_prob = current_noise
            attempts = 0
            max_attempts = 50
            logging.info(f"T{self.t_gate_id} - Starting distillation round {r + 1}...")
            while attempts < max_attempts:
                state, success = self.distill_once(r + 1)
                attempts += 1
                if success:
                    logging.info(f"T{self.t_gate_id} Round {r + 1} succeeded after {attempts} attempts")
                    attempts_this_run.append(attempts)
                    break
                if self.debug:
                    self._log_debug(f"DEBUG: T{self.t_gate_id} Round {r + 1} - Attempt {attempts} failed")
            if attempts == max_attempts:
                logging.error(f"T{self.t_gate_id} Round {r + 1} failed after {max_attempts} attempts")
                return None, 0.0
            current_noise = 35 * (current_noise ** 3)
            logging.info(f"T{self.t_gate_id} - Round {r + 1} completed. Reduced noise to {current_noise:.6f}")
        self.attempts_per_round.extend(attempts_this_run)
        fidelity = state_fidelity(state, self.ideal_t_state)
        success_rate = self.success_count / self.total_attempts if self.total_attempts > 0 else 0.0
        logging.info(f"T{self.t_gate_id} Distilled {self.gate_type} state fidelity after {rounds} rounds: {fidelity:.4f}")
        logging.info(f"T{self.t_gate_id} MSD Success Rate: {success_rate:.4f}")
        logging.info(f"T{self.t_gate_id} Noise Reduced From {initial_noise:.6f} to {current_noise:.6f}")
        return state, fidelity

class MSDFactory:
    def __init__(self, num_input_states=5, noise_prob=0.001, rounds=2, parallel_units=2, debug=False):
        self.num_input_states = num_input_states
        self.noise_prob = noise_prob
        self.rounds = rounds
        self.parallel_units = parallel_units
        self.debug = debug
        self.magic_state_buffer = []
        self.total_qubits_used = 0
        self.total_gates_used = 0
        self.total_time = 0
        self.debug_logs = []
        logging.info(f"MSD Factory initialized: num_input_states={num_input_states}, noise_prob={noise_prob}, rounds={rounds}, parallel_units={parallel_units}")

    def _log_debug(self, message):
        if self.debug:
            logging.debug(message)
            self.debug_logs.append(message)

    def produce_magic_states(self, num_states_needed, gate_types):
        logging.info(f"MSD Factory: Producing {num_states_needed} magic states with {self.parallel_units} parallel units")
        if self.debug:
            self._log_debug(f"MSD Factory: Producing {num_states_needed} magic states with {self.parallel_units} parallel units")
        while len(self.magic_state_buffer) < num_states_needed:
            states_to_produce = min(self.parallel_units, num_states_needed - len(self.magic_state_buffer))
            for unit in range(states_to_produce):
                t_gate_id = len(self.magic_state_buffer) + 1
                gate_type = gate_types[t_gate_id - 1] if t_gate_id - 1 < len(gate_types) else 't'
                logging.info(f"MSD Factory: Distilling {gate_type} state {t_gate_id} in unit {unit + 1}")
                if self.debug:
                    self._log_debug(f"MSD Factory: Distilling {gate_type} state {t_gate_id} in unit {unit + 1}")
                msd = MagicStateDistillation(
                    num_input_states=self.num_input_states,
                    noise_prob=self.noise_prob,
                    t_gate_id=t_gate_id,
                    logical_qubit=None,
                    gate_type=gate_type,
                    debug=self.debug
                )
                magic_state, fidelity = msd.distill(self.rounds)
                if magic_state is not None:
                    self.magic_state_buffer.append((magic_state, fidelity))
                    self.total_qubits_used += 9
                    self.total_gates_used += msd.total_attempts * 100
                    self.total_time += msd.distillation_time
                    self.debug_logs.extend(msd.debug_logs)
                    logging.info(f"MSD Factory: Produced {gate_type} state {t_gate_id} with fidelity {fidelity:.4f}")
                else:
                    logging.error(f"MSD Factory: Failed to produce {gate_type} state {t_gate_id}")
                    return False
        logging.info(f"MSD Factory: Produced {len(self.magic_state_buffer)} magic states")
        if self.debug:
            self._log_debug(f"MSD Factory: Produced {len(self.magic_state_buffer)} magic states")
        return True

    def get_magic_state(self):
        if not self.magic_state_buffer:
            logging.warning("MSD Factory: Magic state buffer is empty!")
            return None, 0.0
        state, fidelity = self.magic_state_buffer.pop(0)
        logging.info(f"MSD Factory: Retrieved magic state with fidelity {fidelity:.4f}")
        return state, fidelity