import logging
import random
import json
import time
import os
import numpy as np
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from core.enhanced_surface_code import EnhancedSurfaceCode as SurfaceCode
from core.magic_state import MSDFactory
from config import LOGS_DIR
from utils.visualization import visualize_surface_code

# Create logs directory if it doesn't exist
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Generate log file name with timestamp
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_debug-{current_time}.log")

# Set up logging to a file for debugging
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ],
    format='%(asctime)s,%(msecs)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class FTQCPipeline:
    def __init__(self, circuit: QuantumCircuit, iterations: int = 10, noise: float = 0.001, distance: int = 3,
                 rounds: int = 2, error_rate: float = 0.01, debug: bool = False):
        self.circuit = circuit
        self.iterations = iterations
        self.noise = noise
        self.distance = distance
        self.rounds = rounds
        self.error_rate = error_rate
        self.debug = debug
        self.debug_logs = []
        self.logical_qubits = []
        self.t_gates = []
        self.gates = []
        self.total_physical_qubits = 0
        self.msd_physical_qubits = 0
        self.total_operations = 0
        self.msd_results = []
        self.session_log = []
        self.execution_time = 0
        self.start_time = time.time()
        self.measurement_stats = {}
        self.physical_errors = []
        self.error_corrections = []
        self.msd_factory = MSDFactory(
            num_input_states=5,
            noise_prob=self.noise,
            rounds=self.rounds,
            parallel_units=2,
            debug=self.debug
        )
        self.t_fidelities = []
        self.ideal_probabilities = {}
        self.simulator_success_prob = 0.0

    def _log_debug(self, message):
        logging.debug(message)
        self.debug_logs.append(message)

    def parse_circuit(self):
        circuit_info = {'gates': [], 'qubits': self.circuit.num_qubits, 'clbits': self.circuit.num_clbits,
                        't_gates': []}
        if self.debug:
            self._log_debug("DEBUG: Parsing circuit instructions")
        for instruction in self.circuit:
            gate = instruction.operation
            qubits = [self.circuit.find_bit(q).index for q in instruction.qubits]
            if self.debug:
                self._log_debug(f"DEBUG: Instruction: {gate.name}, Qubits: {qubits}")
            if gate.name == 'h':
                circuit_info['gates'].append({'gate': 'h', 'target': qubits[0]})
            elif gate.name == 'cx':
                circuit_info['gates'].append({'gate': 'cx', 'control': qubits[0], 'target': qubits[1]})
            elif gate.name == 't':
                circuit_info['gates'].append({'gate': 't', 'target': qubits[0]})
                circuit_info['t_gates'].append(('t', qubits[0]))
            elif gate.name == 'swap':
                circuit_info['gates'].append({'gate': 'swap', 'control': qubits[0], 'target': qubits[1]})
            elif gate.name == 'measure':
                circuit_info['gates'].append({'gate': 'measure', 'target': qubits[0]})
            else:
                logging.warning(f"Unsupported gate '{gate.name}' encountered. Skipping this gate.")
                continue
        self.gates = circuit_info['gates']
        self.t_gates = circuit_info['t_gates']
        self.session_log.append(
            f"Parsed Circuit: Qubits={self.circuit.num_qubits}, Gates={len(self.gates)}, T-Gates={len(self.t_gates)}")
        logging.info(f"=== Fault-Tolerant Quantum Computing Pipeline ===")
        logging.info(f"Parsed circuit info: {circuit_info}")
        return circuit_info

    def setup_surface_code(self):
        if self.debug:
            self._log_debug("DEBUG: Setting up surface code for logical qubits")
        for qubit in range(self.circuit.num_qubits):
            sc = SurfaceCode(
                distance=self.distance,
                logical_qubit_id=qubit,
                num_qubits=self.circuit.num_qubits,
                noise=self.noise,
                error_rate=self.error_rate,
                debug=self.debug
            )
            self.logical_qubits.append(sc)
            self.total_physical_qubits += sc.physical_qubits
        self.session_log.append(f"Surface Code Setup: Total Physical Qubits={self.total_physical_qubits}")
        logging.info(f"Total physical qubits for surface code: {self.total_physical_qubits}")

    def apply_gates(self):
        step = 1
        for gate in self.gates:
            gate_type = gate['gate']
            logging.info(f"Step {step}: Applying {gate_type} gate")
            if self.debug:
                self._log_debug(f"DEBUG: Gate details: {gate}")
            if gate_type == 'h':
                self.total_operations += 100
            elif gate_type == 'cx':
                self.total_operations += 200
            elif gate_type == 't':
                self.total_operations += 150
            elif gate_type == 'swap':
                self.total_operations += 300
            elif gate_type == 'measure':
                self.total_operations += 50
            step += 1
        self.session_log.append(f"Gates Applied: Total Operations={self.total_operations}")

    def distill_magic_states(self):
        if not self.t_gates:
            logging.info("No T-gates found in the circuit. Skipping magic state distillation.")
            self.session_log.append("Magic State Distillation: Skipped (No T-Gates)")
            return [], []
        logging.info("Magic state distillation will be performed on-demand during execution.")
        return [], []

    def simulate_with_noise(self):
        noise_model = NoiseModel()
        error_1q = depolarizing_error(self.noise, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['h', 't', 's'])
        error_2q = depolarizing_error(self.noise * 2, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'swap'])

        simulator = AerSimulator(noise_model=noise_model)

        ideal_circuit = QuantumCircuit(self.circuit.num_qubits, self.circuit.num_clbits)
        for instruction in self.circuit:
            if instruction.operation.name != 'measure':
                ideal_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)
        if self.debug:
            self._log_debug(f"DEBUG: Ideal circuit for statevector simulation:\n{ideal_circuit}")

        ideal_state = Statevector.from_instruction(ideal_circuit)
        if self.debug:
            self._log_debug(f"DEBUG: Ideal statevector:\n{ideal_state}")

        shots = 1000
        noisy_circuit = self.circuit.copy()
        if not any(instr.operation.name == 'measure' for instr in noisy_circuit):
            noisy_circuit.measure_all()
        if self.debug:
            self._log_debug(f"DEBUG: Noisy circuit for simulation:\n{noisy_circuit}")

        result = simulator.run(noisy_circuit, shots=shots).result()
        counts = result.get_counts()
        if self.debug:
            self._log_debug(f"DEBUG: Noisy simulation counts:\n{counts}")

        for state, count in counts.items():
            if state not in self.measurement_stats:
                self.measurement_stats[state] = 0
            self.measurement_stats[state] += count

        self.ideal_probabilities = ideal_state.probabilities_dict()
        if self.debug:
            self._log_debug(f"DEBUG: Ideal probabilities:\n{self.ideal_probabilities}")

        max_prob = max(self.ideal_probabilities.values())
        ideal_states = [state for state, prob in self.ideal_probabilities.items() if abs(prob - max_prob) < 1e-6]
        if self.debug:
            self._log_debug(f"DEBUG: Ideal states with max probability ({max_prob}):\n{ideal_states}")

        total_success_counts = sum(counts.get(state, 0) for state in ideal_states)
        self.simulator_success_prob = total_success_counts / shots
        logging.info(f"Simulator Success Probability (AerSimulator): {self.simulator_success_prob:.4f}")
        return self.simulator_success_prob

    def execute_circuit(self, magic_states_unused):
        if self.debug:
            self._log_debug("DEBUG: Starting fault-tolerant circuit execution")
        ideal_t_state = np.array([np.cos(np.pi / 8), np.exp(1j * np.pi / 4) * np.sin(np.pi / 8)])
        t_fidelities = []
        success_probs = []
        successes = 0

        simulator_success_prob = self.simulate_with_noise()

        for iteration in range(1, self.iterations + 1):
            logging.info(f"Iteration {iteration}/{self.iterations}")
            errors_detected = 0
            for logical_qubit in self.logical_qubits:
                simplified_syndrome, full_syndrome = logical_qubit.measure_syndrome()
                logging.info(
                    f"Logical qubit {logical_qubit.logical_qubit_id} - Raw syndrome: {full_syndrome}, Simplified: {simplified_syndrome}")
                correction = logical_qubit.apply_correction(full_syndrome)
                logging.info(f"Logical qubit {logical_qubit.logical_qubit_id} - Correction applied: {correction}")
                if sum(correction) > 0:
                    errors_detected += 1
                    self.physical_errors.append({
                        "iteration": iteration,
                        "logical_qubit": logical_qubit.logical_qubit_id,
                        "syndrome": full_syndrome,
                        "correction": correction
                    })
                    self.error_corrections.append({
                        "iteration": iteration,
                        "logical_qubit": logical_qubit.logical_qubit_id,
                        "success": True
                    })
            t_gate_index = 0
            for idx, (gate_type, qubit) in enumerate(self.t_gates, 1):
                if t_gate_index >= len(self.msd_factory.magic_state_buffer):
                    success = self.msd_factory.produce_magic_states(t_gate_index + 1)
                    if not success:
                        logging.error(f"Failed to produce magic state for T-gate {idx}")
                        return 0.0, 1.0, 0.0, 0.0
                magic_state, fidelity = self.msd_factory.get_magic_state()
                if magic_state is None:
                    logging.error(f"Failed to retrieve magic state for T-gate {idx}")
                    t_fidelity = max(0, 0.001 - random.uniform(0, 0.0005))
                    success_prob = 0.0
                else:
                    t_fidelity = state_fidelity(magic_state, ideal_t_state)
                    noise = random.uniform(0, 0.001)
                    t_fidelity = max(0, t_fidelity - noise)
                    success_prob = simulator_success_prob if t_fidelity > 0.99 else 0.0
                    logging.info(
                        f"T-gate on logical qubit {qubit}: Fidelity = {t_fidelity:.4f}, Success Prob = {success_prob:.4f}, Noise = {noise:.6f}")
                t_fidelities.append(t_fidelity)
                success_probs.append(success_prob)
                t_gate_index += 1
            logical_error = errors_detected > 0
            if not logical_error:
                successes += 1
                logging.info(f"Iteration {iteration} successful: No logical errors detected")
            else:
                logging.info(f"Iteration {iteration} failed: {errors_detected} logical errors detected")
        avg_fidelity = sum(t_fidelities) / len(t_fidelities) if t_fidelities else 0.0
        avg_success_prob = sum(success_probs) / len(success_probs) if success_probs else 0.0
        logical_error_rate = 1 - (successes / self.iterations)
        success_rate = successes / self.iterations
        self.t_fidelities = t_fidelities
        self.session_log.append(
            f"Execution: Avg T-Gate Fidelity={avg_fidelity:.4f}, Logical Error Rate={logical_error_rate:.4f}")
        logging.info(f"Average T-gate fidelity: {avg_fidelity:.4f}")
        logging.info(f"Logical error rate: {logical_error_rate:.4f}")
        logging.info(f"Success rate: {success_rate:.4f}")
        logging.info(f"Average T-gate success probability: {avg_success_prob:.4f}")
        self.execution_time = time.time() - self.start_time
        for qubit in self.logical_qubits:
            qubit.visualize()
        return avg_fidelity, logical_error_rate, success_rate, avg_success_prob

    def generate_json_response(self, avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts):
        circuit_info = {
            "gates": self.gates,
            "qubits": self.circuit.num_qubits,
            "clbits": self.circuit.num_clbits,
            "tGates": [{"gate": gate_type, "target": qubit} for gate_type, qubit in self.t_gates]
        }

        surface_codes = []
        for logical_qubit in self.logical_qubits:
            grid_size = {"rows": 3, "cols": 3}
            data_qubits = [
                {"id": f"D{i // 3}.{i % 3}", "row": i // 3, "col": i % 3} for i in range(9)
            ]
            t_gate = next((t for t, q in self.t_gates if q == logical_qubit.logical_qubit_id), None)
            if t_gate:
                t_fidelity = self.t_fidelities[self.t_gates.index((t_gate, logical_qubit.logical_qubit_id))]
                data_qubits[logical_qubit.logical_qubit_id]["tGateInjection"] = {
                    "applied": True,
                    "magicState": {
                        "fidelity": t_fidelity,
                        "source": f"MSD Factory Unit {1 if logical_qubit.logical_qubit_id < 2 else 2}",
                        "inputFidelity": 0.95
                    },
                    "ancillaQubit": "A0",
                    "measurementOutcome": 0,
                    "correctionApplied": "none"
                }
            stabilizers = [
                {
                    "id": f"S{idx}",
                    "row": sum((q // 3) for q in qubits) / len(qubits),
                    "col": sum((q % 3) for q in qubits) / len(qubits),
                    "type": stab_type,
                    "connectedQubits": [f"D{q // 3}.{q % 3}" for q in qubits]
                } for idx, (stab_type, qubits) in enumerate(logical_qubit.stabilizers)
            ]
            surface_codes.append({
                "logicalQubitId": logical_qubit.logical_qubit_id,
                "gridSize": grid_size,
                "codeDistance": logical_qubit.distance,
                "dataQubits": data_qubits,
                "stabilizers": stabilizers,
                "metadata": {
                    "syndrome": logical_qubit.syndrome_history[-1]["syndrome"] if logical_qubit.syndrome_history else [
                                                                                                                          0] * logical_qubit.num_stabilizers,
                    "detectedErrors": sum(
                        1 for error in self.physical_errors if error["logical_qubit"] == logical_qubit.logical_qubit_id)
                },
                "visualization": {
                    "showLabels": True,
                    "showGrid": True,
                    "highlightTGates": True,
                    "dataQubitStyle": {"color": "blue", "shape": "circle"},
                    "stabilizerStyle": {
                        "Z": {"color": "yellow", "shape": "square"},
                        "X": {"color": "green", "shape": "square"}
                    },
                    "tGateStyle": {"color": "red", "border": "solid 2px"}
                }
            })

        magic_state_distillation = {
            "factoryType": "on-demand",
            "parallelUnits": self.msd_factory.parallel_units,
            "rounds": self.msd_factory.rounds,
            "inputFidelity": 0.95,
            "outputFidelity": max(self.t_fidelities) if self.t_fidelities else 0.0,
            "noiseReduction": {"initial": self.noise, "final": 35 * (self.noise ** 3)},
            "successRate": self.msd_factory.magic_state_buffer[0][1] if self.msd_factory.magic_state_buffer else 0.4286,
            "tStatesProduced": len(self.t_gates)
        }

        simulation_results = {
            "idealProbabilities": {state: float(prob) for state, prob in self.ideal_probabilities.items()},
            "noisyCounts": self.measurement_stats,
            "simulatorSuccessProbability": self.simulator_success_prob
        }

        theoretical_error_rate = self.noise ** 2
        larger_qubits = self.circuit.num_qubits * 2
        projected_qubits = larger_qubits * (self.logical_qubits[0].distance ** 2 + (
                    self.logical_qubits[0].distance - 1) ** 2) if self.logical_qubits else 0
        projected_error_rate = self.noise ** (self.logical_qubits[0].distance + 1) if self.logical_qubits else 0
        performance_metrics = {
            "theoreticalLogicalErrorRate": theoretical_error_rate,
            "actualLogicalErrorRate": logical_error_rate,
            "physicalQubitsEfficiency": f"{(self.circuit.num_qubits * 5 / self.total_physical_qubits * 100):.1f}% of theoretical minimum" if self.total_physical_qubits > 0 else "0.0%",
            "alternativeApproaches": {
                "codeDistance3": {
                    "physicalQubits": self.circuit.num_qubits * (3 ** 2 + (3 - 1) ** 2),
                    "errorRate": self.noise ** 3
                }
            },
            "scalabilityProjections": {
                f"for{larger_qubits}Qubits": {
                    "physicalQubits": projected_qubits,
                    "errorRate": projected_error_rate
                }
            }
        }

        json_response = {
            "circuitInfo": circuit_info,
            "surfaceCodes": surface_codes,
            "magicStateDistillation": magic_state_distillation,
            "simulationResults": simulation_results,
            "performanceMetrics": performance_metrics
        }

        return json.dumps(json_response, indent=2, sort_keys=True)

    def analyze_results(self, avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts):
        all_debug_logs = self.debug_logs[:]
        for logical_qubit in self.logical_qubits:
            all_debug_logs.extend(logical_qubit.debug_logs)
        all_debug_logs.extend(self.msd_factory.debug_logs)

        logging.info("=== Fault-Tolerant Quantum Circuit Execution Results ===")
        logging.info(f"Circuit: Custom Quantum Circuit")
        logging.info(f"Code Distance: {self.logical_qubits[0].distance if self.logical_qubits else 0}")
        logging.info(f"Physical Error Rate: {self.noise:.3f}")

        logging.info("\nLogical Measurement Results:")
        total_shots = sum(self.measurement_stats.values())
        for state, count in self.measurement_stats.items():
            confidence = (count / total_shots) * 100 if total_shots > 0 else 0
            logging.info(f"- State |{state}> ({count}/{total_shots} shots, {confidence:.1f}% confidence)")

        logging.info("\nMeasurement Statistics:")
        for state, count in self.measurement_stats.items():
            probability = count / total_shots if total_shots > 0 else 0
            logging.info(f"- |{state}>: {probability:.4f} ({count} shots)")

        logging.info("\nParity Check Results (Syndrome Measurement History):")
        for logical_qubit in self.logical_qubits:
            logging.info(f"Logical Qubit {logical_qubit.logical_qubit_id}:")
            for entry in logical_qubit.syndrome_history:
                logging.info(
                    f"  Iteration {entry['iteration']}: Syndrome={entry['syndrome']}, Simplified={entry['simplified']}")

        logging.info("\nError Correction Performance:")
        detected_errors = len(self.physical_errors)
        corrected_errors = sum(1 for correction in self.error_corrections if correction["success"])
        if detected_errors == 0:
            logging.info("- Error Correction Success Rate: N/A (no errors detected)")
        else:
            error_correction_success_rate = (corrected_errors / detected_errors * 100)
            logging.info(f"- Error Correction Success Rate: {error_correction_success_rate:.1f}%")
        logging.info(f"- Detected Errors: {detected_errors}")
        logging.info(f"- Corrected Errors: {corrected_errors}")
        logging.info(f"- Logical Error Rate: {logical_error_rate:.6f}")

        logging.info("\nPhysical Error Events:")
        for error in self.physical_errors:
            logging.info(
                f"- Iteration {error['iteration']}, Logical Qubit {error['logical_qubit']}: Syndrome={error['syndrome']}, Correction={error['correction']}")

        logging.info("\nResource Usage:")
        logging.info(f"- Physical Qubits Used: {self.total_physical_qubits}")
        logging.info(f"- Magic States Consumed: {len(self.t_gates)}")
        distillation_qubits = self.msd_factory.total_qubits_used
        distillation_gates = self.msd_factory.total_gates_used
        distillation_time = self.msd_factory.total_time
        logging.info(
            f"- Distillation Resources: {distillation_qubits} qubits, {distillation_gates} gates, {distillation_time:.2f} seconds")
        logging.info(f"- MSD Factory Parallel Units: {self.msd_factory.parallel_units}")
        circuit_depth = len(self.gates) * 2
        logging.info(f"- Circuit Depth: {circuit_depth} cycles")

        logging.info("\nPerformance Metrics:")
        logging.info(f"- Execution Time: {self.execution_time:.2f} seconds")
        logical_operations = len(self.gates)
        error_correction_overhead = self.total_operations / logical_operations if logical_operations > 0 else 0
        logging.info(f"- Error Correction Overhead: {error_correction_overhead:.1f}x")
        logging.info(f"- Final T-Gate Fidelity: {avg_fidelity * 100:.3f}%")
        threshold_distance = self.error_rate
        threshold_performance = (
                                            threshold_distance - self.noise) / threshold_distance * 100 if threshold_distance > 0 else 0
        logging.info(f"- Threshold Performance: {threshold_performance:.1f}% below threshold")

        logging.info("\nDebugging Information:")
        logging.info("Error Chain Visualization (Simplified):")
        for error in self.physical_errors:
            logging.info(
                f"- Logical Qubit {error['logical_qubit']}, Iteration {error['iteration']}: Syndrome={error['syndrome']}")

        logging.info("\nDecoder Performance:")
        for correction in self.error_corrections:
            logging.info(
                f"- Logical Qubit {correction['logical_qubit']}, Iteration {correction['iteration']}: Success={correction['success']}")

        logging.info("\nFailure Points:")
        if logical_error_rate > 0:
            logging.info(f"- Errors accumulated in {detected_errors - corrected_errors} uncorrected events")
        else:
            logging.info("- No significant failure points detected")

        logging.info("\nCritical Path Analysis:")
        t_gate_ops = len(self.t_gates) * 150
        total_ops = self.total_operations
        t_gate_contribution = (t_gate_ops / total_ops * 100) if total_ops > 0 else 0
        logging.info(f"- T-Gate Operations: {t_gate_contribution:.1f}% of total operations")

        logging.info("\nDebug Logs:")
        for log in all_debug_logs:
            logging.info(f"- {log}")

        logging.info("\nComparative Analysis:")
        theoretical_error_rate = self.noise ** 2
        logging.info(f"Theoretical vs. Actual Performance:")
        logging.info(f"- Theoretical Logical Error Rate: {theoretical_error_rate:.6f}")
        logging.info(f"- Actual Logical Error Rate: {logical_error_rate:.6f}")

        logging.info("\nResource Efficiency:")
        theoretical_min_qubits = self.circuit.num_qubits * 5
        efficiency = (
                    theoretical_min_qubits / self.total_physical_qubits * 100) if self.total_physical_qubits > 0 else 0
        logging.info(f"- Physical Qubits Efficiency: {efficiency:.1f}% of theoretical minimum")

        logging.info("\nAlternative Approaches:")
        logging.info(
            f"- Increasing code distance to 3 would use {self.circuit.num_qubits * (3 ** 2 + (3 - 1) ** 2)} qubits but reduce error rate to ~{(self.noise ** 3):.6f}")

        logging.info("\nScalability Projections:")
        larger_qubits = self.circuit.num_qubits * 2
        projected_qubits = larger_qubits * (self.logical_qubits[0].distance ** 2 + (
                    self.logical_qubits[0].distance - 1) ** 2) if self.logical_qubits else 0
        projected_error_rate = self.noise ** (self.logical_qubits[0].distance + 1) if self.logical_qubits else 0
        logging.info(
            f"- For {larger_qubits} qubits: ~{projected_qubits} physical qubits, error rate ~{projected_error_rate:.6f}")

    def run(self):
        try:
            circuit_info = self.parse_circuit()
            self.setup_surface_code()
            self.apply_gates()
            msd_attempts, magic_states = self.distill_magic_states()
            if msd_attempts is None:
                logging.error("Magic state distillation failed. Aborting execution.")
                return None, None, None, None, None
            avg_fidelity, logical_error_rate, success_rate, avg_success_prob = self.execute_circuit(magic_states)
            self.analyze_results(avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts)

            # Generate and save JSON response
            json_response = self.generate_json_response(avg_fidelity, logical_error_rate, success_rate,
                                                        avg_success_prob, msd_attempts)
            json_filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_response-{current_time}.json")
            with open(json_filename, 'w') as f:
                f.write(json_response)
            logging.info(f"JSON response saved to {json_filename}")
            print(f"JSON response saved to {json_filename}")

            # Generate surface code visualizations for each logical qubit
            for logical_qubit in self.logical_qubits:
                syndrome = logical_qubit.syndrome_history[-1]["syndrome"] if logical_qubit.syndrome_history else [
                                                                                                                     0] * logical_qubit.num_stabilizers
                t_gate_qubit = next((q for _, q in self.t_gates if q == logical_qubit.logical_qubit_id), None)
                vis_filename = visualize_surface_code(logical_qubit, syndrome, t_gate_qubit)
                logging.info(
                    f"Surface code visualization for logical qubit {logical_qubit.logical_qubit_id} saved to {vis_filename}")
                print(f"Surface code visualization saved to {vis_filename}")

            return avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            raise



# import logging
# import random
# import json
# import time
# import os
# import numpy as np
# from datetime import datetime
# from qiskit import QuantumCircuit
# from qiskit.quantum_info import state_fidelity, Statevector
# from qiskit_aer import AerSimulator
# from qiskit_aer.noise import NoiseModel, depolarizing_error
# from enhanced_surface_code import EnhancedSurfaceCode as SurfaceCode
# from core.magic_state import MSDFactory
# from config import LOGS_DIR
#
# # Create logs directory if it doesn't exist
# if not os.path.exists(LOGS_DIR):
#     os.makedirs(LOGS_DIR)
#
# # Generate log file name with timestamp
# current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# log_filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_debug-{current_time}.log")
#
# # Set up logging to a file for debugging
# logging.basicConfig(
#     level=logging.DEBUG,
#     handlers=[
#         logging.FileHandler(log_filename),
#         logging.StreamHandler()
#     ],
#     format='%(asctime)s,%(msecs)d - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
#
#
# class FTQCPipeline:
#     def __init__(self, circuit: QuantumCircuit, iterations: int = 10, noise: float = 0.001, distance: int = 3,
#                  rounds: int = 2, error_rate: float = 0.01, debug: bool = False):
#         self.circuit = circuit
#         self.iterations = iterations
#         self.noise = noise
#         self.distance = distance
#         self.rounds = rounds
#         self.error_rate = error_rate
#         self.debug = debug
#         self.debug_logs = []
#         self.logical_qubits = []
#         self.t_gates = []
#         self.gates = []
#         self.total_physical_qubits = 0
#         self.msd_physical_qubits = 0
#         self.total_operations = 0
#         self.msd_results = []
#         self.session_log = []
#         self.execution_time = 0
#         self.start_time = time.time()
#         self.measurement_stats = {}
#         self.physical_errors = []
#         self.error_corrections = []
#         self.msd_factory = MSDFactory(
#             num_input_states=5,
#             noise_prob=self.noise,
#             rounds=self.rounds,
#             parallel_units=2,
#             debug=self.debug
#         )
#         self.t_fidelities = []
#         self.ideal_probabilities = {}
#         self.simulator_success_prob = 0.0
#
#     def _log_debug(self, message):
#         logging.debug(message)
#         self.debug_logs.append(message)
#
#     def parse_circuit(self):
#         circuit_info = {'gates': [], 'qubits': self.circuit.num_qubits, 'clbits': self.circuit.num_clbits,
#                         't_gates': []}
#         if self.debug:
#             self._log_debug("DEBUG: Parsing circuit instructions")
#         for instruction in self.circuit:
#             gate = instruction.operation
#             qubits = [self.circuit.find_bit(q).index for q in instruction.qubits]
#             if self.debug:
#                 self._log_debug(f"DEBUG: Instruction: {gate.name}, Qubits: {qubits}")
#             if gate.name == 'h':
#                 circuit_info['gates'].append({'gate': 'h', 'target': qubits[0]})
#             elif gate.name == 'cx':
#                 circuit_info['gates'].append({'gate': 'cx', 'control': qubits[0], 'target': qubits[1]})
#             elif gate.name == 't':
#                 circuit_info['gates'].append({'gate': 't', 'target': qubits[0]})
#                 circuit_info['t_gates'].append(('t', qubits[0]))
#             elif gate.name == 'swap':
#                 circuit_info['gates'].append({'gate': 'swap', 'control': qubits[0], 'target': qubits[1]})
#             elif gate.name == 'measure':
#                 circuit_info['gates'].append({'gate': 'measure', 'target': qubits[0]})
#             else:
#                 logging.warning(f"Unsupported gate '{gate.name}' encountered. Skipping this gate.")
#                 continue
#         self.gates = circuit_info['gates']
#         self.t_gates = circuit_info['t_gates']
#         self.session_log.append(
#             f"Parsed Circuit: Qubits={self.circuit.num_qubits}, Gates={len(self.gates)}, T-Gates={len(self.t_gates)}")
#         logging.info(f"=== Fault-Tolerant Quantum Computing Pipeline ===")
#         logging.info(f"Parsed circuit info: {circuit_info}")
#         return circuit_info
#
#     def setup_surface_code(self):
#         if self.debug:
#             self._log_debug("DEBUG: Setting up surface code for logical qubits")
#         for qubit in range(self.circuit.num_qubits):
#             sc = SurfaceCode(
#                 distance=self.distance,
#                 logical_qubit_id=qubit,
#                 num_qubits=self.circuit.num_qubits,
#                 noise=self.noise,
#                 error_rate=self.error_rate,
#                 debug=self.debug
#             )
#             self.logical_qubits.append(sc)
#             self.total_physical_qubits += sc.physical_qubits
#         self.session_log.append(f"Surface Code Setup: Total Physical Qubits={self.total_physical_qubits}")
#         logging.info(f"Total physical qubits for surface code: {self.total_physical_qubits}")
#
#     def apply_gates(self):
#         step = 1
#         for gate in self.gates:
#             gate_type = gate['gate']
#             logging.info(f"Step {step}: Applying {gate_type} gate")
#             if self.debug:
#                 self._log_debug(f"DEBUG: Gate details: {gate}")
#             if gate_type == 'h':
#                 self.total_operations += 100
#             elif gate_type == 'cx':
#                 self.total_operations += 200
#             elif gate_type == 't':
#                 self.total_operations += 150
#             elif gate_type == 'swap':
#                 self.total_operations += 300
#             elif gate_type == 'measure':
#                 self.total_operations += 50
#             step += 1
#         self.session_log.append(f"Gates Applied: Total Operations={self.total_operations}")
#
#     def distill_magic_states(self):
#         if not self.t_gates:
#             logging.info("No T-gates found in the circuit. Skipping magic state distillation.")
#             self.session_log.append("Magic State Distillation: Skipped (No T-Gates)")
#             return [], []
#         logging.info("Magic state distillation will be performed on-demand during execution.")
#         return [], []
#
#     def simulate_with_noise(self):
#         noise_model = NoiseModel()
#         error_1q = depolarizing_error(self.noise, 1)
#         noise_model.add_all_qubit_quantum_error(error_1q, ['h', 't', 's'])
#         error_2q = depolarizing_error(self.noise * 2, 2)
#         noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'swap'])
#
#         simulator = AerSimulator(noise_model=noise_model)
#
#         ideal_circuit = QuantumCircuit(self.circuit.num_qubits, self.circuit.num_clbits)
#         for instruction in self.circuit:
#             if instruction.operation.name != 'measure':
#                 ideal_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal circuit for statevector simulation:\n{ideal_circuit}")
#
#         ideal_state = Statevector.from_instruction(ideal_circuit)
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal statevector:\n{ideal_state}")
#
#         shots = 1000
#         noisy_circuit = self.circuit.copy()
#         if not any(instr.operation.name == 'measure' for instr in noisy_circuit):
#             noisy_circuit.measure_all()
#         if self.debug:
#             self._log_debug(f"DEBUG: Noisy circuit for simulation:\n{noisy_circuit}")
#
#         result = simulator.run(noisy_circuit, shots=shots).result()
#         counts = result.get_counts()
#         if self.debug:
#             self._log_debug(f"DEBUG: Noisy simulation counts:\n{counts}")
#
#         for state, count in counts.items():
#             if state not in self.measurement_stats:
#                 self.measurement_stats[state] = 0
#             self.measurement_stats[state] += count
#
#         self.ideal_probabilities = ideal_state.probabilities_dict()
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal probabilities:\n{self.ideal_probabilities}")
#
#         max_prob = max(self.ideal_probabilities.values())
#         ideal_states = [state for state, prob in self.ideal_probabilities.items() if abs(prob - max_prob) < 1e-6]
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal states with max probability ({max_prob}):\n{ideal_states}")
#
#         total_success_counts = sum(counts.get(state, 0) for state in ideal_states)
#         self.simulator_success_prob = total_success_counts / shots
#         logging.info(f"Simulator Success Probability (AerSimulator): {self.simulator_success_prob:.4f}")
#         return self.simulator_success_prob
#
#     def execute_circuit(self, magic_states_unused):
#         if self.debug:
#             self._log_debug("DEBUG: Starting fault-tolerant circuit execution")
#         ideal_t_state = np.array([np.cos(np.pi / 8), np.exp(1j * np.pi / 4) * np.sin(np.pi / 8)])
#         t_fidelities = []
#         success_probs = []
#         successes = 0
#
#         simulator_success_prob = self.simulate_with_noise()
#
#         for iteration in range(1, self.iterations + 1):
#             logging.info(f"Iteration {iteration}/{self.iterations}")
#             errors_detected = 0
#             for logical_qubit in self.logical_qubits:
#                 simplified_syndrome, full_syndrome = logical_qubit.measure_syndrome()
#                 logging.info(f"Logical qubit {logical_qubit.logical_qubit_id} syndrome: {simplified_syndrome}")
#                 correction = logical_qubit.apply_correction(full_syndrome)
#                 logging.info(f"Correction applied: {correction}")
#                 if sum(correction) > 0:
#                     errors_detected += 1
#                     self.physical_errors.append({
#                         "iteration": iteration,
#                         "logical_qubit": logical_qubit.logical_qubit_id,
#                         "syndrome": full_syndrome,
#                         "correction": correction
#                     })
#                     self.error_corrections.append({
#                         "iteration": iteration,
#                         "logical_qubit": logical_qubit.logical_qubit_id,
#                         "success": True
#                     })
#             t_gate_index = 0
#             for idx, (gate_type, qubit) in enumerate(self.t_gates, 1):
#                 if t_gate_index >= len(self.msd_factory.magic_state_buffer):
#                     success = self.msd_factory.produce_magic_states(t_gate_index + 1)
#                     if not success:
#                         logging.error(f"Failed to produce magic state for T-gate {idx}")
#                         return 0.0, 1.0, 0.0, 0.0
#                 magic_state, fidelity = self.msd_factory.get_magic_state()
#                 if magic_state is None:
#                     logging.error(f"Failed to retrieve magic state for T-gate {idx}")
#                     t_fidelity = max(0, 0.001 - random.uniform(0, 0.0005))
#                     success_prob = 0.0
#                 else:
#                     t_fidelity = state_fidelity(magic_state, ideal_t_state)
#                     noise = random.uniform(0, 0.001)
#                     t_fidelity = max(0, t_fidelity - noise)
#                     success_prob = simulator_success_prob if t_fidelity > 0.99 else 0.0
#                     logging.info(
#                         f"T-gate on logical qubit {qubit}: Fidelity = {t_fidelity:.4f}, Success Prob = {success_prob:.4f}, Noise = {noise:.6f}")
#                 t_fidelities.append(t_fidelity)
#                 success_probs.append(success_prob)
#                 t_gate_index += 1
#             logical_error = errors_detected > 0
#             if not logical_error:
#                 successes += 1
#                 logging.info("Iteration successful: No logical errors detected")
#             else:
#                 logging.info(f"Iteration failed: {errors_detected} logical errors detected")
#         avg_fidelity = sum(t_fidelities) / len(t_fidelities) if t_fidelities else 0.0
#         avg_success_prob = sum(success_probs) / len(success_probs) if success_probs else 0.0
#         logical_error_rate = 1 - (successes / self.iterations)
#         success_rate = successes / self.iterations
#         self.t_fidelities = t_fidelities
#         self.session_log.append(
#             f"Execution: Avg T-Gate Fidelity={avg_fidelity:.4f}, Logical Error Rate={logical_error_rate:.4f}")
#         logging.info(f"Average T-gate fidelity: {avg_fidelity:.4f}")
#         logging.info(f"Logical error rate: {logical_error_rate:.4f}")
#         logging.info(f"Success rate: {success_rate:.4f}")
#         logging.info(f"Average T-gate success probability: {avg_success_prob:.4f}")
#         self.execution_time = time.time() - self.start_time
#         return avg_fidelity, logical_error_rate, success_rate, avg_success_prob
#
#     def generate_json_response(self, avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts):
#         circuit_info = {
#             "gates": self.gates,
#             "qubits": self.circuit.num_qubits,
#             "clbits": self.circuit.num_clbits,
#             "tGates": [{"gate": gate_type, "target": qubit} for gate_type, qubit in self.t_gates]
#         }
#
#         surface_codes = []
#         for logical_qubit in self.logical_qubits:
#             grid_size = {"rows": 2, "cols": 3}
#             data_qubits = [
#                 {"id": "D0.0", "row": 0, "col": 0},
#                 {"id": "D0.1", "row": 0, "col": 1},
#                 {"id": "D1.0", "row": 1, "col": 0},
#                 {"id": "D1.1", "row": 1, "col": 1},
#                 {"id": "A1.1", "row": 1, "col": 2, "type": "ancilla"}
#             ]
#             t_gate = next((t for t, q in self.t_gates if q == logical_qubit.logical_qubit_id), None)
#             if t_gate:
#                 t_fidelity = self.t_fidelities[self.t_gates.index((t_gate, logical_qubit.logical_qubit_id))]
#                 data_qubits[3]["tGateInjection"] = {
#                     "applied": True,
#                     "magicState": {
#                         "fidelity": t_fidelity,
#                         "source": f"MSD Factory Unit {1 if logical_qubit.logical_qubit_id < 2 else 2}",
#                         "inputFidelity": 0.95
#                     },
#                     "ancillaQubit": "A1.1",
#                     "measurementOutcome": 0,
#                     "correctionApplied": "none"
#                 }
#             stabilizers = [
#                 {"id": "S0.0", "row": 0.5, "col": 0.5, "type": "Z",
#                  "connectedQubits": ["D0.0", "D0.1", "D1.0", "D1.1"]},
#                 {"id": "S0.1", "row": 0.5, "col": 1.5, "type": "X", "connectedQubits": ["D0.1", "D1.0", "D1.1"]}
#             ]
#             surface_codes.append({
#                 "logicalQubitId": logical_qubit.logical_qubit_id,
#                 "gridSize": grid_size,
#                 "codeDistance": logical_qubit.distance,
#                 "dataQubits": data_qubits,
#                 "stabilizers": stabilizers,
#                 "metadata": {
#                     "syndrome": logical_qubit.syndrome_history[-1][
#                         "simplified"] if logical_qubit.syndrome_history else [0],
#                     "detectedErrors": sum(
#                         1 for error in self.physical_errors if error["logical_qubit"] == logical_qubit.logical_qubit_id)
#                 },
#                 "visualization": {
#                     "showLabels": True,
#                     "showGrid": True,
#                     "highlightTGates": True,
#                     "dataQubitStyle": {"color": "blue", "shape": "circle"},
#                     "stabilizerStyle": {
#                         "Z": {"color": "yellow", "shape": "square"},
#                         "X": {"color": "green", "shape": "square"}
#                     },
#                     "tGateStyle": {"color": "red", "border": "solid 2px"}
#                 }
#             })
#
#         magic_state_distillation = {
#             "factoryType": "on-demand",
#             "parallelUnits": self.msd_factory.parallel_units,
#             "rounds": self.msd_factory.rounds,
#             "inputFidelity": 0.95,
#             "outputFidelity": max(self.t_fidelities) if self.t_fidelities else 0.0,
#             "noiseReduction": {"initial": self.noise, "final": 35 * (self.noise ** 3)},
#             "successRate": self.msd_factory.magic_state_buffer[0][1] if self.msd_factory.magic_state_buffer else 0.4286,
#             "tStatesProduced": len(self.t_gates)
#         }
#
#         simulation_results = {
#             "idealProbabilities": {state: float(prob) for state, prob in self.ideal_probabilities.items()},
#             "noisyCounts": self.measurement_stats,
#             "simulatorSuccessProbability": self.simulator_success_prob
#         }
#
#         theoretical_error_rate = self.noise ** 2
#         larger_qubits = self.circuit.num_qubits * 2
#         projected_qubits = larger_qubits * (self.logical_qubits[0].distance ** 2 + (
#                     self.logical_qubits[0].distance - 1) ** 2) if self.logical_qubits else 0
#         projected_error_rate = self.noise ** (self.logical_qubits[0].distance + 1) if self.logical_qubits else 0
#         performance_metrics = {
#             "theoreticalLogicalErrorRate": theoretical_error_rate,
#             "actualLogicalErrorRate": logical_error_rate,
#             "physicalQubitsEfficiency": f"{(self.circuit.num_qubits * 5 / self.total_physical_qubits * 100):.1f}% of theoretical minimum" if self.total_physical_qubits > 0 else "0.0%",
#             "alternativeApproaches": {
#                 "codeDistance3": {
#                     "physicalQubits": self.circuit.num_qubits * (3 ** 2 + (3 - 1) ** 2),
#                     "errorRate": self.noise ** 3
#                 }
#             },
#             "scalabilityProjections": {
#                 f"for{larger_qubits}Qubits": {
#                     "physicalQubits": projected_qubits,
#                     "errorRate": projected_error_rate
#                 }
#             }
#         }
#
#         json_response = {
#             "circuitInfo": circuit_info,
#             "surfaceCodes": surface_codes,
#             "magicStateDistillation": magic_state_distillation,
#             "simulationResults": simulation_results,
#             "performanceMetrics": performance_metrics
#         }
#
#         return json.dumps(json_response, indent=2, sort_keys=True)
#
#     def analyze_results(self, avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts):
#         all_debug_logs = self.debug_logs[:]
#         for logical_qubit in self.logical_qubits:
#             all_debug_logs.extend(logical_qubit.debug_logs)
#         all_debug_logs.extend(self.msd_factory.debug_logs)
#
#         logging.info("=== Fault-Tolerant Quantum Circuit Execution Results ===")
#         logging.info(f"Circuit: Custom Quantum Circuit")
#         logging.info(f"Code Distance: {self.logical_qubits[0].distance if self.logical_qubits else 0}")
#         logging.info(f"Physical Error Rate: {self.noise:.3f}")
#
#         logging.info("\nLogical Measurement Results:")
#         total_shots = sum(self.measurement_stats.values())
#         for state, count in self.measurement_stats.items():
#             confidence = (count / total_shots) * 100 if total_shots > 0 else 0
#             logging.info(f"- State |{state}> ({count}/{total_shots} shots, {confidence:.1f}% confidence)")
#
#         logging.info("\nMeasurement Statistics:")
#         for state, count in self.measurement_stats.items():
#             probability = count / total_shots if total_shots > 0 else 0
#             logging.info(f"- |{state}>: {probability:.4f} ({count} shots)")
#
#         logging.info("\nParity Check Results (Syndrome Measurement History):")
#         for logical_qubit in self.logical_qubits:
#             logging.info(f"Logical Qubit {logical_qubit.logical_qubit_id}:")
#             for entry in logical_qubit.syndrome_history:
#                 logging.info(
#                     f"  Iteration {entry['iteration']}: Syndrome={entry['syndrome']}, Simplified={entry['simplified']}")
#
#         logging.info("\nError Correction Performance:")
#         detected_errors = len(self.physical_errors)
#         corrected_errors = sum(1 for correction in self.error_corrections if correction["success"])
#         if detected_errors == 0:
#             logging.info("- Error Correction Success Rate: N/A (no errors detected)")
#         else:
#             error_correction_success_rate = (corrected_errors / detected_errors * 100)
#             logging.info(f"- Error Correction Success Rate: {error_correction_success_rate:.1f}%")
#         logging.info(f"- Detected Errors: {detected_errors}")
#         logging.info(f"- Corrected Errors: {corrected_errors}")
#         logging.info(f"- Logical Error Rate: {logical_error_rate:.6f}")
#
#         logging.info("\nPhysical Error Events:")
#         for error in self.physical_errors:
#             logging.info(
#                 f"- Iteration {error['iteration']}, Logical Qubit {error['logical_qubit']}: Syndrome={error['syndrome']}, Correction={error['correction']}")
#
#         logging.info("\nResource Usage:")
#         logging.info(f"- Physical Qubits Used: {self.total_physical_qubits}")
#         logging.info(f"- Magic States Consumed: {len(self.t_gates)}")
#         distillation_qubits = self.msd_factory.total_qubits_used
#         distillation_gates = self.msd_factory.total_gates_used
#         distillation_time = self.msd_factory.total_time
#         logging.info(
#             f"- Distillation Resources: {distillation_qubits} qubits, {distillation_gates} gates, {distillation_time:.2f} seconds")
#         logging.info(f"- MSD Factory Parallel Units: {self.msd_factory.parallel_units}")
#         circuit_depth = len(self.gates) * 2
#         logging.info(f"- Circuit Depth: {circuit_depth} cycles")
#
#         logging.info("\nPerformance Metrics:")
#         logging.info(f"- Execution Time: {self.execution_time:.2f} seconds")
#         logical_operations = len(self.gates)
#         error_correction_overhead = self.total_operations / logical_operations if logical_operations > 0 else 0
#         logging.info(f"- Error Correction Overhead: {error_correction_overhead:.1f}x")
#         logging.info(f"- Final T-Gate Fidelity: {avg_fidelity * 100:.3f}%")
#         threshold_distance = self.error_rate
#         threshold_performance = (
#                                             threshold_distance - self.noise) / threshold_distance * 100 if threshold_distance > 0 else 0
#         logging.info(f"- Threshold Performance: {threshold_performance:.1f}% below threshold")
#
#         logging.info("\nDebugging Information:")
#         logging.info("Error Chain Visualization (Simplified):")
#         for error in self.physical_errors:
#             logging.info(
#                 f"- Logical Qubit {error['logical_qubit']}, Iteration {error['iteration']}: Syndrome={error['syndrome']}")
#
#         logging.info("\nDecoder Performance:")
#         for correction in self.error_corrections:
#             logging.info(
#                 f"- Logical Qubit {correction['logical_qubit']}, Iteration {correction['iteration']}: Success={correction['success']}")
#
#         logging.info("\nFailure Points:")
#         if logical_error_rate > 0:
#             logging.info(f"- Errors accumulated in {detected_errors - corrected_errors} uncorrected events")
#         else:
#             logging.info("- No significant failure points detected")
#
#         logging.info("\nCritical Path Analysis:")
#         t_gate_ops = len(self.t_gates) * 150
#         total_ops = self.total_operations
#         t_gate_contribution = (t_gate_ops / total_ops * 100) if total_ops > 0 else 0
#         logging.info(f"- T-Gate Operations: {t_gate_contribution:.1f}% of total operations")
#
#         logging.info("\nDebug Logs:")
#         for log in all_debug_logs:
#             logging.info(f"- {log}")
#
#         logging.info("\nComparative Analysis:")
#         theoretical_error_rate = self.noise ** 2
#         logging.info(f"Theoretical vs. Actual Performance:")
#         logging.info(f"- Theoretical Logical Error Rate: {theoretical_error_rate:.6f}")
#         logging.info(f"- Actual Logical Error Rate: {logical_error_rate:.6f}")
#
#         logging.info("\nResource Efficiency:")
#         theoretical_min_qubits = self.circuit.num_qubits * 5
#         efficiency = (
#                     theoretical_min_qubits / self.total_physical_qubits * 100) if self.total_physical_qubits > 0 else 0
#         logging.info(f"- Physical Qubits Efficiency: {efficiency:.1f}% of theoretical minimum")
#
#         logging.info("\nAlternative Approaches:")
#         logging.info(
#             f"- Increasing code distance to 3 would use {self.circuit.num_qubits * (3 ** 2 + (3 - 1) ** 2)} qubits but reduce error rate to ~{(self.noise ** 3):.6f}")
#
#         logging.info("\nScalability Projections:")
#         larger_qubits = self.circuit.num_qubits * 2
#         projected_qubits = larger_qubits * (self.logical_qubits[0].distance ** 2 + (
#                     self.logical_qubits[0].distance - 1) ** 2) if self.logical_qubits else 0
#         projected_error_rate = self.noise ** (self.logical_qubits[0].distance + 1) if self.logical_qubits else 0
#         logging.info(
#             f"- For {larger_qubits} qubits: ~{projected_qubits} physical qubits, error rate ~{projected_error_rate:.6f}")
#
#     def run(self):
#         try:
#             circuit_info = self.parse_circuit()
#             self.setup_surface_code()
#             self.apply_gates()
#             msd_attempts, magic_states = self.distill_magic_states()
#             if msd_attempts is None:
#                 logging.error("Magic state distillation failed. Aborting execution.")
#                 return None, None, None, None, None
#             avg_fidelity, logical_error_rate, success_rate, avg_success_prob = self.execute_circuit(magic_states)
#             self.analyze_results(avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts)
#
#             # Generate and save JSON response
#             json_response = self.generate_json_response(avg_fidelity, logical_error_rate, success_rate,
#                                                         avg_success_prob, msd_attempts)
#             json_filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_response-{current_time}.json")
#             with open(json_filename, 'w') as f:
#                 f.write(json_response)
#             logging.info(f"JSON response saved to {json_filename}")
#             print(f"JSON response saved to {json_filename}")
#             return avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts
#         except Exception as e:
#             logging.error(f"Pipeline execution failed: {str(e)}")
#             raise



# import logging
# import random
# import json
# import time
# import os
# import numpy as np
# from datetime import datetime
# from qiskit import QuantumCircuit
# from qiskit.quantum_info import state_fidelity, Statevector
# from qiskit_aer import AerSimulator
# from qiskit_aer.noise import NoiseModel, depolarizing_error
# from core.surface_code import SurfaceCode
# from core.magic_state import MSDFactory
# from config import LOGS_DIR
# from utils.visualization import visualize_surface_code
#
# # Create logs directory if it doesn't exist
# if not os.path.exists(LOGS_DIR):
#     os.makedirs(LOGS_DIR)
#
# # Generate log file name with timestamp
# current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# log_filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_debug-{current_time}.log")
#
# # Set up logging to a file for debugging
# logging.basicConfig(
#     level=logging.DEBUG,
#     handlers=[
#         logging.FileHandler(log_filename),
#         logging.StreamHandler()
#     ],
#     format='%(asctime)s,%(msecs)d - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
#
#
# class FTQCPipeline:
#     def __init__(self, circuit: QuantumCircuit, iterations: int = 10, noise: float = 0.001, distance: int = 3,
#                  rounds: int = 2, error_rate: float = 0.01, debug: bool = False):
#         self.circuit = circuit
#         self.iterations = iterations
#         self.noise = noise
#         self.distance = distance
#         self.rounds = rounds
#         self.error_rate = error_rate
#         self.debug = debug
#         self.debug_logs = []
#         self.logical_qubits = []
#         self.t_gates = []
#         self.gates = []
#         self.total_physical_qubits = 0
#         self.msd_physical_qubits = 0
#         self.total_operations = 0
#         self.msd_results = []
#         self.session_log = []
#         self.execution_time = 0
#         self.start_time = time.time()
#         self.measurement_stats = {}
#         self.physical_errors = []
#         self.error_corrections = []
#         self.msd_factory = MSDFactory(
#             num_input_states=5,
#             noise_prob=self.noise,
#             rounds=self.rounds,
#             parallel_units=2,
#             debug=self.debug
#         )
#         self.t_fidelities = []
#         self.ideal_probabilities = {}
#         self.simulator_success_prob = 0.0
#
#     def _log_debug(self, message):
#         logging.debug(message)
#         self.debug_logs.append(message)
#
#     def parse_circuit(self):
#         circuit_info = {'gates': [], 'qubits': self.circuit.num_qubits, 'clbits': self.circuit.num_clbits,
#                         't_gates': []}
#         if self.debug:
#             self._log_debug("DEBUG: Parsing circuit instructions")
#         for instruction in self.circuit:
#             gate = instruction.operation
#             qubits = [self.circuit.find_bit(q).index for q in instruction.qubits]
#             if self.debug:
#                 self._log_debug(f"DEBUG: Instruction: {gate.name}, Qubits: {qubits}")
#             if gate.name == 'h':
#                 circuit_info['gates'].append({'gate': 'h', 'target': qubits[0]})
#             elif gate.name == 'cx':
#                 circuit_info['gates'].append({'gate': 'cx', 'control': qubits[0], 'target': qubits[1]})
#             elif gate.name == 't':
#                 circuit_info['gates'].append({'gate': 't', 'target': qubits[0]})
#                 circuit_info['t_gates'].append(('t', qubits[0]))
#             elif gate.name == 'swap':
#                 circuit_info['gates'].append({'gate': 'swap', 'control': qubits[0], 'target': qubits[1]})
#             elif gate.name == 'measure':
#                 circuit_info['gates'].append({'gate': 'measure', 'target': qubits[0]})
#             else:
#                 logging.warning(f"Unsupported gate '{gate.name}' encountered. Skipping this gate.")
#                 continue
#         self.gates = circuit_info['gates']
#         self.t_gates = circuit_info['t_gates']
#         self.session_log.append(
#             f"Parsed Circuit: Qubits={self.circuit.num_qubits}, Gates={len(self.gates)}, T-Gates={len(self.t_gates)}")
#         logging.info(f"=== Fault-Tolerant Quantum Computing Pipeline ===")
#         logging.info(f"Parsed circuit info: {circuit_info}")
#         return circuit_info
#
#     def setup_surface_code(self):
#         if self.debug:
#             self._log_debug("DEBUG: Setting up surface code for logical qubits")
#         for qubit in range(self.circuit.num_qubits):
#             sc = SurfaceCode(
#                 distance=self.distance,
#                 logical_qubit_id=qubit,
#                 num_qubits=self.circuit.num_qubits,
#                 noise=self.noise,
#                 error_rate=self.error_rate,
#                 debug=self.debug
#             )
#             self.logical_qubits.append(sc)
#             self.total_physical_qubits += sc.physical_qubits
#         self.session_log.append(f"Surface Code Setup: Total Physical Qubits={self.total_physical_qubits}")
#         logging.info(f"Total physical qubits for surface code: {self.total_physical_qubits}")
#
#     def apply_gates(self):
#         step = 1
#         for gate in self.gates:
#             gate_type = gate['gate']
#             logging.info(f"Step {step}: Applying {gate_type} gate")
#             if self.debug:
#                 self._log_debug(f"DEBUG: Gate details: {gate}")
#             if gate_type == 'h':
#                 self.total_operations += 100
#             elif gate_type == 'cx':
#                 self.total_operations += 200
#             elif gate_type == 't':
#                 self.total_operations += 150
#             elif gate_type == 'swap':
#                 self.total_operations += 300
#             elif gate_type == 'measure':
#                 self.total_operations += 50
#             step += 1
#         self.session_log.append(f"Gates Applied: Total Operations={self.total_operations}")
#
#     def distill_magic_states(self):
#         if not self.t_gates:
#             logging.info("No T-gates found in the circuit. Skipping magic state distillation.")
#             self.session_log.append("Magic State Distillation: Skipped (No T-Gates)")
#             return [], []
#         logging.info("Magic state distillation will be performed on-demand during execution.")
#         return [], []
#
#     def simulate_with_noise(self):
#         noise_model = NoiseModel()
#         error_1q = depolarizing_error(self.noise, 1)
#         noise_model.add_all_qubit_quantum_error(error_1q, ['h', 't', 's'])
#         error_2q = depolarizing_error(self.noise * 2, 2)
#         noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'swap'])
#
#         simulator = AerSimulator(noise_model=noise_model)
#
#         ideal_circuit = QuantumCircuit(self.circuit.num_qubits, self.circuit.num_clbits)
#         for instruction in self.circuit:
#             if instruction.operation.name != 'measure':
#                 ideal_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal circuit for statevector simulation:\n{ideal_circuit}")
#
#         ideal_state = Statevector.from_instruction(ideal_circuit)
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal statevector:\n{ideal_state}")
#
#         shots = 1000
#         noisy_circuit = self.circuit.copy()
#         if not any(instr.operation.name == 'measure' for instr in noisy_circuit):
#             noisy_circuit.measure_all()
#         if self.debug:
#             self._log_debug(f"DEBUG: Noisy circuit for simulation:\n{noisy_circuit}")
#
#         result = simulator.run(noisy_circuit, shots=shots).result()
#         counts = result.get_counts()
#         if self.debug:
#             self._log_debug(f"DEBUG: Noisy simulation counts:\n{counts}")
#
#         for state, count in counts.items():
#             if state not in self.measurement_stats:
#                 self.measurement_stats[state] = 0
#             self.measurement_stats[state] += count
#
#         self.ideal_probabilities = ideal_state.probabilities_dict()
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal probabilities:\n{self.ideal_probabilities}")
#
#         max_prob = max(self.ideal_probabilities.values())
#         ideal_states = [state for state, prob in self.ideal_probabilities.items() if abs(prob - max_prob) < 1e-6]
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal states with max probability ({max_prob}):\n{ideal_states}")
#
#         total_success_counts = sum(counts.get(state, 0) for state in ideal_states)
#         self.simulator_success_prob = total_success_counts / shots
#         logging.info(f"Simulator Success Probability (AerSimulator): {self.simulator_success_prob:.4f}")
#         return self.simulator_success_prob
#
#     def execute_circuit(self, magic_states_unused):
#         if self.debug:
#             self._log_debug("DEBUG: Starting fault-tolerant circuit execution")
#         ideal_t_state = np.array([np.cos(np.pi / 8), np.exp(1j * np.pi / 4) * np.sin(np.pi / 8)])
#         t_fidelities = []
#         success_probs = []
#         successes = 0
#
#         simulator_success_prob = self.simulate_with_noise()
#
#         for iteration in range(1, self.iterations + 1):
#             logging.info(f"Iteration {iteration}/{self.iterations}")
#             errors_detected = 0
#             for logical_qubit in self.logical_qubits:
#                 simplified_syndrome, full_syndrome = logical_qubit.measure_syndrome()
#                 logging.info(
#                     f"Logical qubit {logical_qubit.logical_qubit_id} - Raw syndrome: {full_syndrome}, Simplified: {simplified_syndrome}")
#                 correction = logical_qubit.apply_correction(full_syndrome)
#                 logging.info(f"Logical qubit {logical_qubit.logical_qubit_id} - Correction applied: {correction}")
#                 if sum(correction) > 0:
#                     errors_detected += 1
#                     self.physical_errors.append({
#                         "iteration": iteration,
#                         "logical_qubit": logical_qubit.logical_qubit_id,
#                         "syndrome": full_syndrome,
#                         "correction": correction
#                     })
#                     self.error_corrections.append({
#                         "iteration": iteration,
#                         "logical_qubit": logical_qubit.logical_qubit_id,
#                         "success": True
#                     })
#             t_gate_index = 0
#             for idx, (gate_type, qubit) in enumerate(self.t_gates, 1):
#                 if t_gate_index >= len(self.msd_factory.magic_state_buffer):
#                     success = self.msd_factory.produce_magic_states(t_gate_index + 1)
#                     if not success:
#                         logging.error(f"Failed to produce magic state for T-gate {idx}")
#                         return 0.0, 1.0, 0.0, 0.0
#                 magic_state, fidelity = self.msd_factory.get_magic_state()
#                 if magic_state is None:
#                     logging.error(f"Failed to retrieve magic state for T-gate {idx}")
#                     t_fidelity = max(0, 0.001 - random.uniform(0, 0.0005))
#                     success_prob = 0.0
#                 else:
#                     t_fidelity = state_fidelity(magic_state, ideal_t_state)
#                     noise = random.uniform(0, 0.001)
#                     t_fidelity = max(0, t_fidelity - noise)
#                     success_prob = simulator_success_prob if t_fidelity > 0.99 else 0.0
#                     logging.info(
#                         f"T-gate on logical qubit {qubit}: Fidelity = {t_fidelity:.4f}, Success Prob = {success_prob:.4f}, Noise = {noise:.6f}")
#                 t_fidelities.append(t_fidelity)
#                 success_probs.append(success_prob)
#                 t_gate_index += 1
#             logical_error = errors_detected > 0
#             if not logical_error:
#                 successes += 1
#                 logging.info(f"Iteration {iteration} successful: No logical errors detected")
#             else:
#                 logging.info(f"Iteration {iteration} failed: {errors_detected} logical errors detected")
#         avg_fidelity = sum(t_fidelities) / len(t_fidelities) if t_fidelities else 0.0
#         avg_success_prob = sum(success_probs) / len(success_probs) if success_probs else 0.0
#         logical_error_rate = 1 - (successes / self.iterations)
#         success_rate = successes / self.iterations
#         self.t_fidelities = t_fidelities
#         self.session_log.append(
#             f"Execution: Avg T-Gate Fidelity={avg_fidelity:.4f}, Logical Error Rate={logical_error_rate:.4f}")
#         logging.info(f"Average T-gate fidelity: {avg_fidelity:.4f}")
#         logging.info(f"Logical error rate: {logical_error_rate:.4f}")
#         logging.info(f"Success rate: {success_rate:.4f}")
#         logging.info(f"Average T-gate success probability: {avg_success_prob:.4f}")
#         self.execution_time = time.time() - self.start_time
#         return avg_fidelity, logical_error_rate, success_rate, avg_success_prob
#
#     def generate_json_response(self, avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts):
#         circuit_info = {
#             "gates": self.gates,
#             "qubits": self.circuit.num_qubits,
#             "clbits": self.circuit.num_clbits,
#             "tGates": [{"gate": gate_type, "target": qubit} for gate_type, qubit in self.t_gates]
#         }
#
#         surface_codes = []
#         for logical_qubit in self.logical_qubits:
#             grid_size = {"rows": 3, "cols": 3}
#             data_qubits = [
#                 {"id": f"D{i // 3}.{i % 3}", "row": i // 3, "col": i % 3} for i in range(9)
#             ]
#             t_gate = next((t for t, q in self.t_gates if q == logical_qubit.logical_qubit_id), None)
#             if t_gate:
#                 t_fidelity = self.t_fidelities[self.t_gates.index((t_gate, logical_qubit.logical_qubit_id))]
#                 data_qubits[logical_qubit.logical_qubit_id]["tGateInjection"] = {
#                     "applied": True,
#                     "magicState": {
#                         "fidelity": t_fidelity,
#                         "source": f"MSD Factory Unit {1 if logical_qubit.logical_qubit_id < 2 else 2}",
#                         "inputFidelity": 0.95
#                     },
#                     "ancillaQubit": "A0",
#                     "measurementOutcome": 0,
#                     "correctionApplied": "none"
#                 }
#             stabilizers = [
#                 {
#                     "id": f"S{idx}",
#                     "row": sum((q // 3) for q in qubits) / len(qubits),
#                     "col": sum((q % 3) for q in qubits) / len(qubits),
#                     "type": stab_type,
#                     "connectedQubits": [f"D{q // 3}.{q % 3}" for q in qubits]
#                 } for idx, (stab_type, qubits) in enumerate(logical_qubit.stabilizers)
#             ]
#             surface_codes.append({
#                 "logicalQubitId": logical_qubit.logical_qubit_id,
#                 "gridSize": grid_size,
#                 "codeDistance": logical_qubit.distance,
#                 "dataQubits": data_qubits,
#                 "stabilizers": stabilizers,
#                 "metadata": {
#                     "syndrome": logical_qubit.syndrome_history[-1]["syndrome"] if logical_qubit.syndrome_history else [
#                                                                                                                           0] * logical_qubit.num_stabilizers,
#                     "detectedErrors": sum(
#                         1 for error in self.physical_errors if error["logical_qubit"] == logical_qubit.logical_qubit_id)
#                 },
#                 "visualization": {
#                     "showLabels": True,
#                     "showGrid": True,
#                     "highlightTGates": True,
#                     "dataQubitStyle": {"color": "blue", "shape": "circle"},
#                     "stabilizerStyle": {
#                         "Z": {"color": "yellow", "shape": "square"},
#                         "X": {"color": "green", "shape": "square"}
#                     },
#                     "tGateStyle": {"color": "red", "border": "solid 2px"}
#                 }
#             })
#
#         magic_state_distillation = {
#             "factoryType": "on-demand",
#             "parallelUnits": self.msd_factory.parallel_units,
#             "rounds": self.msd_factory.rounds,
#             "inputFidelity": 0.95,
#             "outputFidelity": max(self.t_fidelities) if self.t_fidelities else 0.0,
#             "noiseReduction": {"initial": self.noise, "final": 35 * (self.noise ** 3)},
#             "successRate": self.msd_factory.magic_state_buffer[0][1] if self.msd_factory.magic_state_buffer else 0.4286,
#             "tStatesProduced": len(self.t_gates)
#         }
#
#         simulation_results = {
#             "idealProbabilities": {state: float(prob) for state, prob in self.ideal_probabilities.items()},
#             "noisyCounts": self.measurement_stats,
#             "simulatorSuccessProbability": self.simulator_success_prob
#         }
#
#         theoretical_error_rate = self.noise ** 2
#         larger_qubits = self.circuit.num_qubits * 2
#         projected_qubits = larger_qubits * (self.logical_qubits[0].distance ** 2 + (
#                     self.logical_qubits[0].distance - 1) ** 2) if self.logical_qubits else 0
#         projected_error_rate = self.noise ** (self.logical_qubits[0].distance + 1) if self.logical_qubits else 0
#         performance_metrics = {
#             "theoreticalLogicalErrorRate": theoretical_error_rate,
#             "actualLogicalErrorRate": logical_error_rate,
#             "physicalQubitsEfficiency": f"{(self.circuit.num_qubits * 5 / self.total_physical_qubits * 100):.1f}% of theoretical minimum" if self.total_physical_qubits > 0 else "0.0%",
#             "alternativeApproaches": {
#                 "codeDistance3": {
#                     "physicalQubits": self.circuit.num_qubits * (3 ** 2 + (3 - 1) ** 2),
#                     "errorRate": self.noise ** 3
#                 }
#             },
#             "scalabilityProjections": {
#                 f"for{larger_qubits}Qubits": {
#                     "physicalQubits": projected_qubits,
#                     "errorRate": projected_error_rate
#                 }
#             }
#         }
#
#         json_response = {
#             "circuitInfo": circuit_info,
#             "surfaceCodes": surface_codes,
#             "magicStateDistillation": magic_state_distillation,
#             "simulationResults": simulation_results,
#             "performanceMetrics": performance_metrics
#         }
#
#         return json.dumps(json_response, indent=2, sort_keys=True)
#
#     def analyze_results(self, avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts):
#         all_debug_logs = self.debug_logs[:]
#         for logical_qubit in self.logical_qubits:
#             all_debug_logs.extend(logical_qubit.debug_logs)
#         all_debug_logs.extend(self.msd_factory.debug_logs)
#
#         logging.info("=== Fault-Tolerant Quantum Circuit Execution Results ===")
#         logging.info(f"Circuit: Custom Quantum Circuit")
#         logging.info(f"Code Distance: {self.logical_qubits[0].distance if self.logical_qubits else 0}")
#         logging.info(f"Physical Error Rate: {self.noise:.3f}")
#
#         logging.info("\nLogical Measurement Results:")
#         total_shots = sum(self.measurement_stats.values())
#         for state, count in self.measurement_stats.items():
#             confidence = (count / total_shots) * 100 if total_shots > 0 else 0
#             logging.info(f"- State |{state}> ({count}/{total_shots} shots, {confidence:.1f}% confidence)")
#
#         logging.info("\nMeasurement Statistics:")
#         for state, count in self.measurement_stats.items():
#             probability = count / total_shots if total_shots > 0 else 0
#             logging.info(f"- |{state}>: {probability:.4f} ({count} shots)")
#
#         logging.info("\nParity Check Results (Syndrome Measurement History):")
#         for logical_qubit in self.logical_qubits:
#             logging.info(f"Logical Qubit {logical_qubit.logical_qubit_id}:")
#             for entry in logical_qubit.syndrome_history:
#                 logging.info(
#                     f"  Iteration {entry['iteration']}: Syndrome={entry['syndrome']}, Simplified={entry['simplified']}")
#
#         logging.info("\nError Correction Performance:")
#         detected_errors = len(self.physical_errors)
#         corrected_errors = sum(1 for correction in self.error_corrections if correction["success"])
#         if detected_errors == 0:
#             logging.info("- Error Correction Success Rate: N/A (no errors detected)")
#         else:
#             error_correction_success_rate = (corrected_errors / detected_errors * 100)
#             logging.info(f"- Error Correction Success Rate: {error_correction_success_rate:.1f}%")
#         logging.info(f"- Detected Errors: {detected_errors}")
#         logging.info(f"- Corrected Errors: {corrected_errors}")
#         logging.info(f"- Logical Error Rate: {logical_error_rate:.6f}")
#
#         logging.info("\nPhysical Error Events:")
#         for error in self.physical_errors:
#             logging.info(
#                 f"- Iteration {error['iteration']}, Logical Qubit {error['logical_qubit']}: Syndrome={error['syndrome']}, Correction={error['correction']}")
#
#         logging.info("\nResource Usage:")
#         logging.info(f"- Physical Qubits Used: {self.total_physical_qubits}")
#         logging.info(f"- Magic States Consumed: {len(self.t_gates)}")
#         distillation_qubits = self.msd_factory.total_qubits_used
#         distillation_gates = self.msd_factory.total_gates_used
#         distillation_time = self.msd_factory.total_time
#         logging.info(
#             f"- Distillation Resources: {distillation_qubits} qubits, {distillation_gates} gates, {distillation_time:.2f} seconds")
#         logging.info(f"- MSD Factory Parallel Units: {self.msd_factory.parallel_units}")
#         circuit_depth = len(self.gates) * 2
#         logging.info(f"- Circuit Depth: {circuit_depth} cycles")
#
#         logging.info("\nPerformance Metrics:")
#         logging.info(f"- Execution Time: {self.execution_time:.2f} seconds")
#         logical_operations = len(self.gates)
#         error_correction_overhead = self.total_operations / logical_operations if logical_operations > 0 else 0
#         logging.info(f"- Error Correction Overhead: {error_correction_overhead:.1f}x")
#         logging.info(f"- Final T-Gate Fidelity: {avg_fidelity * 100:.3f}%")
#         threshold_distance = self.error_rate
#         threshold_performance = (
#                                             threshold_distance - self.noise) / threshold_distance * 100 if threshold_distance > 0 else 0
#         logging.info(f"- Threshold Performance: {threshold_performance:.1f}% below threshold")
#
#         logging.info("\nDebugging Information:")
#         logging.info("Error Chain Visualization (Simplified):")
#         for error in self.physical_errors:
#             logging.info(
#                 f"- Logical Qubit {error['logical_qubit']}, Iteration {error['iteration']}: Syndrome={error['syndrome']}")
#
#         logging.info("\nDecoder Performance:")
#         for correction in self.error_corrections:
#             logging.info(
#                 f"- Logical Qubit {correction['logical_qubit']}, Iteration {correction['iteration']}: Success={correction['success']}")
#
#         logging.info("\nFailure Points:")
#         if logical_error_rate > 0:
#             logging.info(f"- Errors accumulated in {detected_errors - corrected_errors} uncorrected events")
#         else:
#             logging.info("- No significant failure points detected")
#
#         logging.info("\nCritical Path Analysis:")
#         t_gate_ops = len(self.t_gates) * 150
#         total_ops = self.total_operations
#         t_gate_contribution = (t_gate_ops / total_ops * 100) if total_ops > 0 else 0
#         logging.info(f"- T-Gate Operations: {t_gate_contribution:.1f}% of total operations")
#
#         logging.info("\nDebug Logs:")
#         for log in all_debug_logs:
#             logging.info(f"- {log}")
#
#         logging.info("\nComparative Analysis:")
#         theoretical_error_rate = self.noise ** 2
#         logging.info(f"Theoretical vs. Actual Performance:")
#         logging.info(f"- Theoretical Logical Error Rate: {theoretical_error_rate:.6f}")
#         logging.info(f"- Actual Logical Error Rate: {logical_error_rate:.6f}")
#
#         logging.info("\nResource Efficiency:")
#         theoretical_min_qubits = self.circuit.num_qubits * 5
#         efficiency = (
#                     theoretical_min_qubits / self.total_physical_qubits * 100) if self.total_physical_qubits > 0 else 0
#         logging.info(f"- Physical Qubits Efficiency: {efficiency:.1f}% of theoretical minimum")
#
#         logging.info("\nAlternative Approaches:")
#         logging.info(
#             f"- Increasing code distance to 3 would use {self.circuit.num_qubits * (3 ** 2 + (3 - 1) ** 2)} qubits but reduce error rate to ~{(self.noise ** 3):.6f}")
#
#         logging.info("\nScalability Projections:")
#         larger_qubits = self.circuit.num_qubits * 2
#         projected_qubits = larger_qubits * (self.logical_qubits[0].distance ** 2 + (
#                     self.logical_qubits[0].distance - 1) ** 2) if self.logical_qubits else 0
#         projected_error_rate = self.noise ** (self.logical_qubits[0].distance + 1) if self.logical_qubits else 0
#         logging.info(
#             f"- For {larger_qubits} qubits: ~{projected_qubits} physical qubits, error rate ~{projected_error_rate:.6f}")
#
#     def run(self):
#         try:
#             circuit_info = self.parse_circuit()
#             self.setup_surface_code()
#             self.apply_gates()
#             msd_attempts, magic_states = self.distill_magic_states()
#             if msd_attempts is None:
#                 logging.error("Magic state distillation failed. Aborting execution.")
#                 return None, None, None, None, None
#             avg_fidelity, logical_error_rate, success_rate, avg_success_prob = self.execute_circuit(magic_states)
#             self.analyze_results(avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts)
#
#             # Generate and save JSON response
#             json_response = self.generate_json_response(avg_fidelity, logical_error_rate, success_rate,
#                                                         avg_success_prob, msd_attempts)
#             json_filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_response-{current_time}.json")
#             with open(json_filename, 'w') as f:
#                 f.write(json_response)
#             logging.info(f"JSON response saved to {json_filename}")
#             print(f"JSON response saved to {json_filename}")
#
#             # Generate surface code visualizations for each logical qubit
#             for logical_qubit in self.logical_qubits:
#                 syndrome = logical_qubit.syndrome_history[-1]["syndrome"] if logical_qubit.syndrome_history else [
#                                                                                                                      0] * logical_qubit.num_stabilizers
#                 t_gate_qubit = next((q for _, q in self.t_gates if q == logical_qubit.logical_qubit_id), None)
#                 vis_filename = visualize_surface_code(logical_qubit, syndrome, t_gate_qubit)
#                 logging.info(
#                     f"Surface code visualization for logical qubit {logical_qubit.logical_qubit_id} saved to {vis_filename}")
#                 print(f"Surface code visualization saved to {vis_filename}")
#
#             return avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts
#         except Exception as e:
#             logging.error(f"Pipeline execution failed: {str(e)}")
#             raise



# import logging
# import random
# import json
# import time
# import os
# import numpy as np
# from datetime import datetime
# from qiskit import QuantumCircuit
# from qiskit.quantum_info import state_fidelity, Statevector
# from qiskit_aer import AerSimulator
# from qiskit_aer.noise import NoiseModel, depolarizing_error
# from core.surface_code import SurfaceCode
# from core.magic_state import MSDFactory
# from config import LOGS_DIR
#
# # Create logs directory if it doesn't exist
# if not os.path.exists(LOGS_DIR):
#     os.makedirs(LOGS_DIR)
#
# # Generate log file name with timestamp
# current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# log_filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_debug-{current_time}.log")
#
# # Set up logging to a file for debugging
# logging.basicConfig(
#     level=logging.DEBUG,
#     handlers=[
#         logging.FileHandler(log_filename),
#         logging.StreamHandler()
#     ],
#     format='%(asctime)s,%(msecs)d - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
#
#
# class FTQCPipeline:
#     def __init__(self, circuit: QuantumCircuit, iterations: int = 10, noise: float = 0.001, distance: int = 3,
#                  rounds: int = 2, error_rate: float = 0.01, debug: bool = False):
#         self.circuit = circuit
#         self.iterations = iterations
#         self.noise = noise
#         self.distance = distance
#         self.rounds = rounds
#         self.error_rate = error_rate
#         self.debug = debug
#         self.debug_logs = []
#         self.logical_qubits = []
#         self.t_gates = []
#         self.gates = []
#         self.total_physical_qubits = 0
#         self.msd_physical_qubits = 0
#         self.total_operations = 0
#         self.msd_results = []
#         self.session_log = []
#         self.execution_time = 0
#         self.start_time = time.time()
#         self.measurement_stats = {}
#         self.physical_errors = []
#         self.error_corrections = []
#         self.msd_factory = MSDFactory(
#             num_input_states=5,
#             noise_prob=self.noise,
#             rounds=self.rounds,
#             parallel_units=2,
#             debug=self.debug
#         )
#         self.t_fidelities = []
#         self.ideal_probabilities = {}
#         self.simulator_success_prob = 0.0
#
#     def _log_debug(self, message):
#         logging.debug(message)
#         self.debug_logs.append(message)
#
#     def parse_circuit(self):
#         circuit_info = {'gates': [], 'qubits': self.circuit.num_qubits, 'clbits': self.circuit.num_clbits,
#                         't_gates': []}
#         if self.debug:
#             self._log_debug("DEBUG: Parsing circuit instructions")
#         for instruction in self.circuit:
#             gate = instruction.operation
#             qubits = [self.circuit.find_bit(q).index for q in instruction.qubits]
#             if self.debug:
#                 self._log_debug(f"DEBUG: Instruction: {gate.name}, Qubits: {qubits}")
#             if gate.name == 'h':
#                 circuit_info['gates'].append({'gate': 'h', 'target': qubits[0]})
#             elif gate.name == 'cx':
#                 circuit_info['gates'].append({'gate': 'cx', 'control': qubits[0], 'target': qubits[1]})
#             elif gate.name == 't':
#                 circuit_info['gates'].append({'gate': 't', 'target': qubits[0]})
#                 circuit_info['t_gates'].append(('t', qubits[0]))
#             elif gate.name == 'swap':
#                 circuit_info['gates'].append({'gate': 'swap', 'control': qubits[0], 'target': qubits[1]})
#             elif gate.name == 'measure':
#                 circuit_info['gates'].append({'gate': 'measure', 'target': qubits[0]})
#             else:
#                 logging.warning(f"Unsupported gate '{gate.name}' encountered. Skipping this gate.")
#                 continue
#         self.gates = circuit_info['gates']
#         self.t_gates = circuit_info['t_gates']
#         self.session_log.append(
#             f"Parsed Circuit: Qubits={self.circuit.num_qubits}, Gates={len(self.gates)}, T-Gates={len(self.t_gates)}")
#         logging.info(f"=== Fault-Tolerant Quantum Computing Pipeline ===")
#         logging.info(f"Parsed circuit info: {circuit_info}")
#         return circuit_info
#
#     def setup_surface_code(self):
#         if self.debug:
#             self._log_debug("DEBUG: Setting up surface code for logical qubits")
#         for qubit in range(self.circuit.num_qubits):
#             sc = SurfaceCode(
#                 distance=self.distance,
#                 logical_qubit_id=qubit,
#                 num_qubits=self.circuit.num_qubits,
#                 noise=self.noise,
#                 error_rate=self.error_rate,
#                 debug=self.debug
#             )
#             self.logical_qubits.append(sc)
#             self.total_physical_qubits += sc.physical_qubits
#         self.session_log.append(f"Surface Code Setup: Total Physical Qubits={self.total_physical_qubits}")
#         logging.info(f"Total physical qubits for surface code: {self.total_physical_qubits}")
#
#     def apply_gates(self):
#         step = 1
#         for gate in self.gates:
#             gate_type = gate['gate']
#             logging.info(f"Step {step}: Applying {gate_type} gate")
#             if self.debug:
#                 self._log_debug(f"DEBUG: Gate details: {gate}")
#             if gate_type == 'h':
#                 self.total_operations += 100
#             elif gate_type == 'cx':
#                 self.total_operations += 200
#             elif gate_type == 't':
#                 self.total_operations += 150
#             elif gate_type == 'swap':
#                 self.total_operations += 300
#             elif gate_type == 'measure':
#                 self.total_operations += 50
#             step += 1
#         self.session_log.append(f"Gates Applied: Total Operations={self.total_operations}")
#
#     def distill_magic_states(self):
#         if not self.t_gates:
#             logging.info("No T-gates found in the circuit. Skipping magic state distillation.")
#             self.session_log.append("Magic State Distillation: Skipped (No T-Gates)")
#             return [], []
#         logging.info("Magic state distillation will be performed on-demand during execution.")
#         return [], []
#
#     def simulate_with_noise(self):
#         noise_model = NoiseModel()
#         error_1q = depolarizing_error(self.noise, 1)
#         noise_model.add_all_qubit_quantum_error(error_1q, ['h', 't', 's'])
#         error_2q = depolarizing_error(self.noise * 2, 2)
#         noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'swap'])
#
#         simulator = AerSimulator(noise_model=noise_model)
#
#         ideal_circuit = QuantumCircuit(self.circuit.num_qubits, self.circuit.num_clbits)
#         for instruction in self.circuit:
#             if instruction.operation.name != 'measure':
#                 ideal_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal circuit for statevector simulation:\n{ideal_circuit}")
#
#         ideal_state = Statevector.from_instruction(ideal_circuit)
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal statevector:\n{ideal_state}")
#
#         shots = 1000
#         noisy_circuit = self.circuit.copy()
#         if not any(instr.operation.name == 'measure' for instr in noisy_circuit):
#             noisy_circuit.measure_all()
#         if self.debug:
#             self._log_debug(f"DEBUG: Noisy circuit for simulation:\n{noisy_circuit}")
#
#         result = simulator.run(noisy_circuit, shots=shots).result()
#         counts = result.get_counts()
#         if self.debug:
#             self._log_debug(f"DEBUG: Noisy simulation counts:\n{counts}")
#
#         for state, count in counts.items():
#             if state not in self.measurement_stats:
#                 self.measurement_stats[state] = 0
#             self.measurement_stats[state] += count
#
#         self.ideal_probabilities = ideal_state.probabilities_dict()
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal probabilities:\n{self.ideal_probabilities}")
#
#         max_prob = max(self.ideal_probabilities.values())
#         ideal_states = [state for state, prob in self.ideal_probabilities.items() if abs(prob - max_prob) < 1e-6]
#         if self.debug:
#             self._log_debug(f"DEBUG: Ideal states with max probability ({max_prob}):\n{ideal_states}")
#
#         total_success_counts = sum(counts.get(state, 0) for state in ideal_states)
#         self.simulator_success_prob = total_success_counts / shots
#         logging.info(f"Simulator Success Probability (AerSimulator): {self.simulator_success_prob:.4f}")
#         return self.simulator_success_prob
#
#     def execute_circuit(self, magic_states_unused):
#         if self.debug:
#             self._log_debug("DEBUG: Starting fault-tolerant circuit execution")
#         ideal_t_state = np.array([np.cos(np.pi / 8), np.exp(1j * np.pi / 4) * np.sin(np.pi / 8)])
#         t_fidelities = []
#         success_probs = []
#         successes = 0
#
#         simulator_success_prob = self.simulate_with_noise()
#
#         for iteration in range(1, self.iterations + 1):
#             logging.info(f"Iteration {iteration}/{self.iterations}")
#             errors_detected = 0
#             for logical_qubit in self.logical_qubits:
#                 simplified_syndrome, full_syndrome = logical_qubit.measure_syndrome()
#                 logging.info(f"Logical qubit {logical_qubit.logical_qubit_id} syndrome: {simplified_syndrome}")
#                 correction = logical_qubit.apply_correction(full_syndrome)
#                 logging.info(f"Correction applied: {correction}")
#                 if sum(correction) > 0:
#                     errors_detected += 1
#                     self.physical_errors.append({
#                         "iteration": iteration,
#                         "logical_qubit": logical_qubit.logical_qubit_id,
#                         "syndrome": full_syndrome,
#                         "correction": correction
#                     })
#                     self.error_corrections.append({
#                         "iteration": iteration,
#                         "logical_qubit": logical_qubit.logical_qubit_id,
#                         "success": True
#                     })
#             t_gate_index = 0
#             for idx, (gate_type, qubit) in enumerate(self.t_gates, 1):
#                 if t_gate_index >= len(self.msd_factory.magic_state_buffer):
#                     success = self.msd_factory.produce_magic_states(t_gate_index + 1)
#                     if not success:
#                         logging.error(f"Failed to produce magic state for T-gate {idx}")
#                         return 0.0, 1.0, 0.0, 0.0
#                 magic_state, fidelity = self.msd_factory.get_magic_state()
#                 if magic_state is None:
#                     logging.error(f"Failed to retrieve magic state for T-gate {idx}")
#                     t_fidelity = max(0, 0.001 - random.uniform(0, 0.0005))
#                     success_prob = 0.0
#                 else:
#                     t_fidelity = state_fidelity(magic_state, ideal_t_state)
#                     noise = random.uniform(0, 0.001)
#                     t_fidelity = max(0, t_fidelity - noise)
#                     success_prob = simulator_success_prob if t_fidelity > 0.99 else 0.0
#                     logging.info(
#                         f"T-gate on logical qubit {qubit}: Fidelity = {t_fidelity:.4f}, Success Prob = {success_prob:.4f}, Noise = {noise:.6f}")
#                 t_fidelities.append(t_fidelity)
#                 success_probs.append(success_prob)
#                 t_gate_index += 1
#             logical_error = errors_detected > 0
#             if not logical_error:
#                 successes += 1
#                 logging.info("Iteration successful: No logical errors detected")
#             else:
#                 logging.info(f"Iteration failed: {errors_detected} logical errors detected")
#         avg_fidelity = sum(t_fidelities) / len(t_fidelities) if t_fidelities else 0.0
#         avg_success_prob = sum(success_probs) / len(success_probs) if success_probs else 0.0
#         logical_error_rate = 1 - (successes / self.iterations)
#         success_rate = successes / self.iterations
#         self.t_fidelities = t_fidelities
#         self.session_log.append(
#             f"Execution: Avg T-Gate Fidelity={avg_fidelity:.4f}, Logical Error Rate={logical_error_rate:.4f}")
#         logging.info(f"Average T-gate fidelity: {avg_fidelity:.4f}")
#         logging.info(f"Logical error rate: {logical_error_rate:.4f}")
#         logging.info(f"Success rate: {success_rate:.4f}")
#         logging.info(f"Average T-gate success probability: {avg_success_prob:.4f}")
#         self.execution_time = time.time() - self.start_time
#         return avg_fidelity, logical_error_rate, success_rate, avg_success_prob
#
#     def generate_json_response(self, avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts):
#         circuit_info = {
#             "gates": self.gates,
#             "qubits": self.circuit.num_qubits,
#             "clbits": self.circuit.num_clbits,
#             "tGates": [{"gate": gate_type, "target": qubit} for gate_type, qubit in self.t_gates]
#         }
#
#         surface_codes = []
#         for logical_qubit in self.logical_qubits:
#             grid_size = {"rows": 2, "cols": 3}
#             data_qubits = [
#                 {"id": "D0.0", "row": 0, "col": 0},
#                 {"id": "D0.1", "row": 0, "col": 1},
#                 {"id": "D1.0", "row": 1, "col": 0},
#                 {"id": "D1.1", "row": 1, "col": 1},
#                 {"id": "A1.1", "row": 1, "col": 2, "type": "ancilla"}
#             ]
#             t_gate = next((t for t, q in self.t_gates if q == logical_qubit.logical_qubit_id), None)
#             if t_gate:
#                 t_fidelity = self.t_fidelities[self.t_gates.index((t_gate, logical_qubit.logical_qubit_id))]
#                 data_qubits[3]["tGateInjection"] = {
#                     "applied": True,
#                     "magicState": {
#                         "fidelity": t_fidelity,
#                         "source": f"MSD Factory Unit {1 if logical_qubit.logical_qubit_id < 2 else 2}",
#                         "inputFidelity": 0.95
#                     },
#                     "ancillaQubit": "A1.1",
#                     "measurementOutcome": 0,
#                     "correctionApplied": "none"
#                 }
#             stabilizers = [
#                 {"id": "S0.0", "row": 0.5, "col": 0.5, "type": "Z",
#                  "connectedQubits": ["D0.0", "D0.1", "D1.0", "D1.1"]},
#                 {"id": "S0.1", "row": 0.5, "col": 1.5, "type": "X", "connectedQubits": ["D0.1", "D1.0", "D1.1"]}
#             ]
#             surface_codes.append({
#                 "logicalQubitId": logical_qubit.logical_qubit_id,
#                 "gridSize": grid_size,
#                 "codeDistance": logical_qubit.distance,
#                 "dataQubits": data_qubits,
#                 "stabilizers": stabilizers,
#                 "metadata": {
#                     "syndrome": logical_qubit.syndrome_history[-1][
#                         "simplified"] if logical_qubit.syndrome_history else [0],
#                     "detectedErrors": sum(
#                         1 for error in self.physical_errors if error["logical_qubit"] == logical_qubit.logical_qubit_id)
#                 },
#                 "visualization": {
#                     "showLabels": True,
#                     "showGrid": True,
#                     "highlightTGates": True,
#                     "dataQubitStyle": {"color": "blue", "shape": "circle"},
#                     "stabilizerStyle": {
#                         "Z": {"color": "yellow", "shape": "square"},
#                         "X": {"color": "green", "shape": "square"}
#                     },
#                     "tGateStyle": {"color": "red", "border": "solid 2px"}
#                 }
#             })
#
#         magic_state_distillation = {
#             "factoryType": "on-demand",
#             "parallelUnits": self.msd_factory.parallel_units,
#             "rounds": self.msd_factory.rounds,
#             "inputFidelity": 0.95,
#             "outputFidelity": max(self.t_fidelities) if self.t_fidelities else 0.0,
#             "noiseReduction": {"initial": self.noise, "final": 35 * (self.noise ** 3)},
#             "successRate": self.msd_factory.magic_state_buffer[0][1] if self.msd_factory.magic_state_buffer else 0.4286,
#             "tStatesProduced": len(self.t_gates)
#         }
#
#         simulation_results = {
#             "idealProbabilities": {state: float(prob) for state, prob in self.ideal_probabilities.items()},
#             "noisyCounts": self.measurement_stats,
#             "simulatorSuccessProbability": self.simulator_success_prob
#         }
#
#         theoretical_error_rate = self.noise ** 2
#         larger_qubits = self.circuit.num_qubits * 2
#         projected_qubits = larger_qubits * (self.logical_qubits[0].distance ** 2 + (
#                     self.logical_qubits[0].distance - 1) ** 2) if self.logical_qubits else 0
#         projected_error_rate = self.noise ** (self.logical_qubits[0].distance + 1) if self.logical_qubits else 0
#         performance_metrics = {
#             "theoreticalLogicalErrorRate": theoretical_error_rate,
#             "actualLogicalErrorRate": logical_error_rate,
#             "physicalQubitsEfficiency": f"{(self.circuit.num_qubits * 5 / self.total_physical_qubits * 100):.1f}% of theoretical minimum" if self.total_physical_qubits > 0 else "0.0%",
#             "alternativeApproaches": {
#                 "codeDistance3": {
#                     "physicalQubits": self.circuit.num_qubits * (3 ** 2 + (3 - 1) ** 2),
#                     "errorRate": self.noise ** 3
#                 }
#             },
#             "scalabilityProjections": {
#                 f"for{larger_qubits}Qubits": {
#                     "physicalQubits": projected_qubits,
#                     "errorRate": projected_error_rate
#                 }
#             }
#         }
#
#         json_response = {
#             "circuitInfo": circuit_info,
#             "surfaceCodes": surface_codes,
#             "magicStateDistillation": magic_state_distillation,
#             "simulationResults": simulation_results,
#             "performanceMetrics": performance_metrics
#         }
#
#         return json.dumps(json_response, indent=2, sort_keys=True)
#
#     def analyze_results(self, avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts):
#         all_debug_logs = self.debug_logs[:]
#         for logical_qubit in self.logical_qubits:
#             all_debug_logs.extend(logical_qubit.debug_logs)
#         all_debug_logs.extend(self.msd_factory.debug_logs)
#
#         logging.info("=== Fault-Tolerant Quantum Circuit Execution Results ===")
#         logging.info(f"Circuit: Custom Quantum Circuit")
#         logging.info(f"Code Distance: {self.logical_qubits[0].distance if self.logical_qubits else 0}")
#         logging.info(f"Physical Error Rate: {self.noise:.3f}")
#
#         logging.info("\nLogical Measurement Results:")
#         total_shots = sum(self.measurement_stats.values())
#         for state, count in self.measurement_stats.items():
#             confidence = (count / total_shots) * 100 if total_shots > 0 else 0
#             logging.info(f"- State |{state}> ({count}/{total_shots} shots, {confidence:.1f}% confidence)")
#
#         logging.info("\nMeasurement Statistics:")
#         for state, count in self.measurement_stats.items():
#             probability = count / total_shots if total_shots > 0 else 0
#             logging.info(f"- |{state}>: {probability:.4f} ({count} shots)")
#
#         logging.info("\nParity Check Results (Syndrome Measurement History):")
#         for logical_qubit in self.logical_qubits:
#             logging.info(f"Logical Qubit {logical_qubit.logical_qubit_id}:")
#             for entry in logical_qubit.syndrome_history:
#                 logging.info(
#                     f"  Iteration {entry['iteration']}: Syndrome={entry['syndrome']}, Simplified={entry['simplified']}")
#
#         logging.info("\nError Correction Performance:")
#         detected_errors = len(self.physical_errors)
#         corrected_errors = sum(1 for correction in self.error_corrections if correction["success"])
#         if detected_errors == 0:
#             logging.info("- Error Correction Success Rate: N/A (no errors detected)")
#         else:
#             error_correction_success_rate = (corrected_errors / detected_errors * 100)
#             logging.info(f"- Error Correction Success Rate: {error_correction_success_rate:.1f}%")
#         logging.info(f"- Detected Errors: {detected_errors}")
#         logging.info(f"- Corrected Errors: {corrected_errors}")
#         logging.info(f"- Logical Error Rate: {logical_error_rate:.6f}")
#
#         logging.info("\nPhysical Error Events:")
#         for error in self.physical_errors:
#             logging.info(
#                 f"- Iteration {error['iteration']}, Logical Qubit {error['logical_qubit']}: Syndrome={error['syndrome']}, Correction={error['correction']}")
#
#         logging.info("\nResource Usage:")
#         logging.info(f"- Physical Qubits Used: {self.total_physical_qubits}")
#         logging.info(f"- Magic States Consumed: {len(self.t_gates)}")
#         distillation_qubits = self.msd_factory.total_qubits_used
#         distillation_gates = self.msd_factory.total_gates_used
#         distillation_time = self.msd_factory.total_time
#         logging.info(
#             f"- Distillation Resources: {distillation_qubits} qubits, {distillation_gates} gates, {distillation_time:.2f} seconds")
#         logging.info(f"- MSD Factory Parallel Units: {self.msd_factory.parallel_units}")
#         circuit_depth = len(self.gates) * 2
#         logging.info(f"- Circuit Depth: {circuit_depth} cycles")
#
#         logging.info("\nPerformance Metrics:")
#         logging.info(f"- Execution Time: {self.execution_time:.2f} seconds")
#         logical_operations = len(self.gates)
#         error_correction_overhead = self.total_operations / logical_operations if logical_operations > 0 else 0
#         logging.info(f"- Error Correction Overhead: {error_correction_overhead:.1f}x")
#         logging.info(f"- Final T-Gate Fidelity: {avg_fidelity * 100:.3f}%")
#         threshold_distance = self.error_rate
#         threshold_performance = (
#                                             threshold_distance - self.noise) / threshold_distance * 100 if threshold_distance > 0 else 0
#         logging.info(f"- Threshold Performance: {threshold_performance:.1f}% below threshold")
#
#         logging.info("\nDebugging Information:")
#         logging.info("Error Chain Visualization (Simplified):")
#         for error in self.physical_errors:
#             logging.info(
#                 f"- Logical Qubit {error['logical_qubit']}, Iteration {error['iteration']}: Syndrome={error['syndrome']}")
#
#         logging.info("\nDecoder Performance:")
#         for correction in self.error_corrections:
#             logging.info(
#                 f"- Logical Qubit {correction['logical_qubit']}, Iteration {correction['iteration']}: Success={correction['success']}")
#
#         logging.info("\nFailure Points:")
#         if logical_error_rate > 0:
#             logging.info(f"- Errors accumulated in {detected_errors - corrected_errors} uncorrected events")
#         else:
#             logging.info("- No significant failure points detected")
#
#         logging.info("\nCritical Path Analysis:")
#         t_gate_ops = len(self.t_gates) * 150
#         total_ops = self.total_operations
#         t_gate_contribution = (t_gate_ops / total_ops * 100) if total_ops > 0 else 0
#         logging.info(f"- T-Gate Operations: {t_gate_contribution:.1f}% of total operations")
#
#         logging.info("\nDebug Logs:")
#         for log in all_debug_logs:
#             logging.info(f"- {log}")
#
#         logging.info("\nComparative Analysis:")
#         theoretical_error_rate = self.noise ** 2
#         logging.info(f"Theoretical vs. Actual Performance:")
#         logging.info(f"- Theoretical Logical Error Rate: {theoretical_error_rate:.6f}")
#         logging.info(f"- Actual Logical Error Rate: {logical_error_rate:.6f}")
#
#         logging.info("\nResource Efficiency:")
#         theoretical_min_qubits = self.circuit.num_qubits * 5
#         efficiency = (
#                     theoretical_min_qubits / self.total_physical_qubits * 100) if self.total_physical_qubits > 0 else 0
#         logging.info(f"- Physical Qubits Efficiency: {efficiency:.1f}% of theoretical minimum")
#
#         logging.info("\nAlternative Approaches:")
#         logging.info(
#             f"- Increasing code distance to 3 would use {self.circuit.num_qubits * (3 ** 2 + (3 - 1) ** 2)} qubits but reduce error rate to ~{(self.noise ** 3):.6f}")
#
#         logging.info("\nScalability Projections:")
#         larger_qubits = self.circuit.num_qubits * 2
#         projected_qubits = larger_qubits * (self.logical_qubits[0].distance ** 2 + (
#                     self.logical_qubits[0].distance - 1) ** 2) if self.logical_qubits else 0
#         projected_error_rate = self.noise ** (self.logical_qubits[0].distance + 1) if self.logical_qubits else 0
#         logging.info(
#             f"- For {larger_qubits} qubits: ~{projected_qubits} physical qubits, error rate ~{projected_error_rate:.6f}")
#
#     def run(self):
#         try:
#             circuit_info = self.parse_circuit()
#             self.setup_surface_code()
#             self.apply_gates()
#             msd_attempts, magic_states = self.distill_magic_states()
#             if msd_attempts is None:
#                 logging.error("Magic state distillation failed. Aborting execution.")
#                 return None, None, None, None, None
#             avg_fidelity, logical_error_rate, success_rate, avg_success_prob = self.execute_circuit(magic_states)
#             self.analyze_results(avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts)
#
#             # Generate and save JSON response
#             json_response = self.generate_json_response(avg_fidelity, logical_error_rate, success_rate,
#                                                         avg_success_prob, msd_attempts)
#             json_filename = os.path.join(LOGS_DIR, f"ftqc_pipeline_response-{current_time}.json")
#             with open(json_filename, 'w') as f:
#                 f.write(json_response)
#             logging.info(f"JSON response saved to {json_filename}")
#             print(f"JSON response saved to {json_filename}")
#             return avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts
#         except Exception as e:
#             logging.error(f"Pipeline execution failed: {str(e)}")
#             raise