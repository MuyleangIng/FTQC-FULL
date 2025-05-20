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
from utils.visualization import visualize_surface_code
from config import LOGS_DIR, FIGURES_DIR

class FTQCPipeline:
    def __init__(self, circuit: QuantumCircuit, iterations: int = 10, noise: float = 0.001, distance: int = 3,
                 rounds: int = 2, error_rate: float = 0.01, debug: bool = True, job_id: str = None):
        self.circuit = circuit
        self.iterations = iterations
        self.noise = noise
        self.distance = distance
        self.rounds = rounds
        self.error_rate = error_rate
        self.debug = debug
        self.job_id = job_id
        self.logger = logging.getLogger(f"job_{self.job_id}")
        self.debug_logs = []
        self.logical_qubits = []
        self.t_gates = []
        self.gates = []
        self.total_physical_qubits = 0
        self.msd_physical_qubits = 0
        self.total_operations = 0
        self.operation_counts = {"H": 0, "CX": 0, "T": 0, "Measure": 0, "Swap": 0}
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
        self.logger.info(f"Initialized FTQC Pipeline for job {self.job_id}")
        self.logger.info(f"Simulation Parameters: iterations={iterations}, noise={noise}, distance={distance}, rounds={rounds}, error_rate={error_rate}, debug={debug}")

    def _log_debug(self, message):
        if self.debug:
            self.logger.debug(message)
            self.debug_logs.append(message)

    def parse_circuit(self):
        self.logger.info("Starting circuit parsing...")
        if not self.circuit:
            self.logger.error("Circuit is None. Cannot proceed with parsing.")
            raise ValueError("Circuit is None")
        circuit_info = {'gates': [], 'qubits': self.circuit.num_qubits, 'clbits': self.circuit.num_clbits, 't_gates': []}
        self._log_debug("DEBUG: Parsing circuit instructions")
        for instruction in self.circuit:
            gate = instruction.operation
            qubits = [self.circuit.find_bit(q).index for q in instruction.qubits]
            self._log_debug(f"DEBUG: Instruction: {gate.name}, Qubits: {qubits}")
            if gate.name == 'h':
                circuit_info['gates'].append({'gate': 'h', 'target': qubits[0]})
                self.operation_counts["H"] += 1
            elif gate.name == 'cx':
                circuit_info['gates'].append({'gate': 'cx', 'control': qubits[0], 'target': qubits[1]})
                self.operation_counts["CX"] += 1
            elif gate.name == 't':
                circuit_info['gates'].append({'gate': 't', 'target': qubits[0]})
                circuit_info['t_gates'].append(('t', qubits[0]))
                self.operation_counts["T"] += 1
            elif gate.name == 'swap':
                circuit_info['gates'].append({'gate': 'swap', 'control': qubits[0], 'target': qubits[1]})
                self.operation_counts["Swap"] += 1
            elif gate.name == 'measure':
                circuit_info['gates'].append({'gate': 'measure', 'target': qubits[0]})
                self.operation_counts["Measure"] += 1
            else:
                self.logger.warning(f"Unsupported gate '{gate.name}' encountered. Skipping this gate.")
                continue
        self.gates = circuit_info['gates']
        self.t_gates = circuit_info['t_gates']
        self.session_log.append(f"Parsed Circuit: Qubits={self.circuit.num_qubits}, Gates={len(self.gates)}, T-Gates={len(self.t_gates)}")
        self.logger.info(f"Parsed circuit: Qubits={self.circuit.num_qubits}, Gates={len(self.gates)}, T-Gates={len(self.t_gates)}")
        self._log_debug(f"DEBUG: Parsed circuit info: {circuit_info}")
        self.logger.info("Circuit parsing completed.")
        return circuit_info

    def setup_surface_code(self):
        self.logger.info("Setting up surface code for logical qubits...")
        self._log_debug("DEBUG: Setting up surface code for logical qubits")
        if self.circuit.num_qubits == 0:
            self.logger.warning("No qubits in circuit. Skipping surface code setup.")
            return
        for qubit in range(self.circuit.num_qubits):
            self.logger.info(f"Configuring logical qubit {qubit}...")
            sc = SurfaceCode(
                distance=self.distance,
                logical_qubit_id=qubit,
                num_qubits=self.circuit.num_qubits,
                noise=self.noise,
                error_rate=self.error_rate,
                debug=self.debug
            )
            sc.initialize_logical_state(state='zero')
            self.logical_qubits.append(sc)
            self.total_physical_qubits += sc.physical_qubits
            self.logger.info(f"Logical qubit {qubit} configured with {sc.physical_qubits} physical qubits (distance={self.distance}).")

        for i in range(len(self.logical_qubits) - 1):
            self.logical_qubits[i].set_neighbor_patch(self.logical_qubits[i + 1])
            self.logical_qubits[i + 1].set_neighbor_patch(self.logical_qubits[i])
            self.logger.info(f"Set logical qubit {i} neighbor to logical qubit {i + 1}")
            self.logger.info(f"Set logical qubit {i + 1} neighbor to logical qubit {i}")

        self.session_log.append(f"Surface Code Setup: Total Physical Qubits={self.total_physical_qubits}")
        self.logger.info(f"Total physical qubits for surface code: {self.total_physical_qubits}")
        self.logger.info("Surface code setup completed.")

    def apply_gates(self):
        self.logger.info("Applying gates to the circuit...")
        if not self.gates:
            self.logger.warning("No gates to apply. Skipping gate application.")
            return
        step = 1
        for gate in self.gates:
            gate_type = gate['gate']
            self.logger.info(f"Step {step}: Applying {gate_type} gate")
            self._log_debug(f"DEBUG: Gate details: {gate}")
            if gate_type == 'h':
                self.total_operations += 100
            elif gate_type == 'cx':
                if self.distance >= 5:
                    control_qubit = gate['control']
                    target_qubit = gate['target']
                    if control_qubit < len(self.logical_qubits) and target_qubit < len(self.logical_qubits):
                        target_patch = self.logical_qubits[target_qubit]
                        control_patch = self.logical_qubits[control_qubit]
                        if target_patch.neighbor_patch == control_patch:
                            self.logger.info(f"Applying logical CNOT via lattice surgery: control qubit {control_qubit}, target qubit {target_qubit}")
                            target_patch.lattice_surgery_cnot(control_patch)
                            self.total_operations += 500
                        else:
                            self.logger.warning(f"Cannot apply CNOT via lattice surgery: Control qubit {control_qubit} is not the neighbor of target qubit {target_qubit}")
                            self.total_operations += 200
                    else:
                        self.logger.warning(f"Cannot apply CNOT: Invalid qubit indices {control_qubit}, {target_qubit}")
                        self.total_operations += 200
                else:
                    self.total_operations += 200
            elif gate_type == 't':
                self.total_operations += 150
            elif gate_type == 'swap':
                self.total_operations += 300
            elif gate_type == 'measure':
                self.total_operations += 50
            step += 1
        self.session_log.append(f"Gates Applied: Total Operations={self.total_operations}")
        self.logger.info(f"Gate application completed. Total operations: {self.total_operations}")

    def distill_magic_states(self):
        self.logger.info("Initiating magic state distillation...")
        if not self.t_gates:
            self.logger.info("No T-gates found in the circuit. Skipping magic state distillation.")
            self.session_log.append("Magic State Distillation: Skipped (No T-Gates)")
            return [], []
        self.logger.info(f"Found {len(self.t_gates)} T-gates. Proceeding with magic state distillation.")
        num_states_needed = len(self.t_gates)
        self.logger.info(f"Producing {num_states_needed} magic states using MSDFactory...")
        success = self.msd_factory.produce_magic_states(num_states_needed)
        if not success:
            self.logger.error("Failed to produce required magic states.")
            return None, []
        self.logger.info(f"Successfully produced {len(self.msd_factory.magic_state_buffer)} magic states.")
        self.msd_physical_qubits = self.msd_factory.total_qubits_used
        self.logger.info(f"Magic State Distillation completed: Used {self.msd_physical_qubits} physical qubits.")
        return [], []

    def simulate_with_noise(self):
        self.logger.info("Simulating circuit with noise model...")
        if self.circuit.num_qubits == 0:
            self.logger.warning("No qubits in circuit. Skipping noise simulation.")
            return 0.0
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
        self._log_debug(f"DEBUG: Ideal circuit for statevector simulation:\n{ideal_circuit}")

        ideal_state = Statevector.from_instruction(ideal_circuit)
        self._log_debug(f"DEBUG: Ideal statevector:\n{ideal_state}")

        shots = 1000
        noisy_circuit = self.circuit.copy()
        if not any(instr.operation.name == 'measure' for instr in noisy_circuit):
            noisy_circuit.measure_all()
        self._log_debug(f"DEBUG: Noisy circuit for simulation:\n{noisy_circuit}")

        result = simulator.run(noisy_circuit, shots=shots).result()
        counts = result.get_counts()
        self._log_debug(f"DEBUG: Noisy simulation counts:\n{counts}")

        for state, count in counts.items():
            if state not in self.measurement_stats:
                self.measurement_stats[state] = 0
            self.measurement_stats[state] += count

        self.ideal_probabilities = ideal_state.probabilities_dict()
        self._log_debug(f"DEBUG: Ideal probabilities:\n{self.ideal_probabilities}")

        max_prob = max(self.ideal_probabilities.values())
        ideal_states = [state for state, prob in self.ideal_probabilities.items() if abs(prob - max_prob) < 1e-6]
        self._log_debug(f"DEBUG: Ideal states with max probability ({max_prob}):\n{ideal_states}")

        total_success_counts = sum(counts.get(state, 0) for state in ideal_states)
        self.simulator_success_prob = total_success_counts / shots
        self.logger.info(f"Simulator Success Probability (AerSimulator): {self.simulator_success_prob:.4f}")
        self.logger.info("Noise simulation completed.")
        return self.simulator_success_prob

    def execute_circuit(self, magic_states_unused):
        self.logger.info("Starting fault-tolerant circuit execution...")
        self._log_debug("DEBUG: Starting fault-tolerant circuit execution")
        if not self.logical_qubits:
            self.logger.warning("No logical qubits configured. Skipping circuit execution.")
            return 0.0, 0.0, 0.0, 0.0
        ideal_t_state = np.array([np.cos(np.pi / 8), np.exp(1j * np.pi / 4) * np.sin(np.pi / 8)])
        t_fidelities = []
        success_probs = []
        successes = 0

        simulator_success_prob = self.simulate_with_noise()

        logical_errors = 0
        for iteration in range(1, self.iterations + 1):
            self.logger.info(f"Iteration {iteration}/{self.iterations}")
            errors_detected = 0
            for logical_qubit in self.logical_qubits:
                simplified_syndrome, full_syndrome = logical_qubit.measure_syndrome()
                self._log_debug(f"Logical qubit {logical_qubit.logical_qubit_id} - Syndrome: {full_syndrome}")
                correction = logical_qubit.apply_correction(full_syndrome)
                self._log_debug(f"Logical qubit {logical_qubit.logical_qubit_id} - Correction: {correction}")
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
            self._log_debug(f"Iteration {iteration} - Errors detected: {errors_detected}")
            self.logger.info(f"Iteration {iteration} - Errors detected: {errors_detected}")
            t_gate_index = 0
            for idx, (gate_type, qubit) in enumerate(self.t_gates, 1):
                if t_gate_index >= len(self.msd_factory.magic_state_buffer):
                    self.logger.info(f"Producing magic states for T-gate {idx}...")
                    success = self.msd_factory.produce_magic_states(t_gate_index + 1)
                    if not success:
                        self.logger.error(f"Failed to produce magic state for T-gate {idx}")
                        t_fidelity = 0.0
                        success_prob = 0.0
                    else:
                        magic_state, fidelity = self.msd_factory.get_magic_state()
                        if magic_state is None:
                            self.logger.error(f"Failed to retrieve magic state for T-gate {idx}")
                            t_fidelity = 0.0
                            success_prob = 0.0
                        else:
                            t_fidelity = state_fidelity(magic_state, ideal_t_state)
                            noise = random.uniform(0, 0.001)
                            t_fidelity = max(0, t_fidelity - noise)
                            success_prob = simulator_success_prob if t_fidelity > 0.99 else 0.0
                            self.logger.info(f"T-gate on logical qubit {qubit}: Fidelity = {t_fidelity:.4f}, Success Prob = {success_prob:.4f}, Noise = {noise:.6f}")
                    t_fidelities.append(t_fidelity)
                    success_probs.append(success_prob)
                else:
                    magic_state, fidelity = self.msd_factory.get_magic_state()
                    if magic_state is None:
                        self.logger.error(f"Failed to retrieve magic state for T-gate {idx}")
                        t_fidelity = 0.0
                        success_prob = 0.0
                    else:
                        t_fidelity = state_fidelity(magic_state, ideal_t_state)
                        noise = random.uniform(0, 0.001)
                        t_fidelity = max(0, t_fidelity - noise)
                        success_prob = simulator_success_prob if t_fidelity > 0.99 else 0.0
                        self.logger.info(f"T-gate on logical qubit {qubit}: Fidelity = {t_fidelity:.4f}, Success Prob = {success_prob:.4f}, Noise = {noise:.6f}")
                    t_fidelities.append(t_fidelity)
                    success_probs.append(success_prob)
                t_gate_index += 1

            if random.random() > simulator_success_prob:
                logical_errors += 1
                self.logger.info(f"Iteration {iteration} failed: Logical error detected via noisy simulation")
            else:
                successes += 1
                self.logger.info(f"Iteration {iteration} successful: No logical errors detected")

        avg_fidelity = sum(t_fidelities) / len(t_fidelities) if t_fidelities else 0.0
        avg_success_prob = sum(success_probs) / len(success_probs) if success_probs else 0.0
        logical_error_rate = logical_errors / self.iterations
        success_rate = successes / self.iterations
        self.t_fidelities = t_fidelities
        self.session_log.append(f"Execution: Avg T-Gate Fidelity={avg_fidelity:.4f}, Logical Error Rate={logical_error_rate:.4f}")
        self.logger.info(f"Average T-gate fidelity: {avg_fidelity:.4f}")
        self.logger.info(f"Logical error rate: {logical_error_rate:.4f}")
        self.logger.info(f"Success rate: {success_rate:.4f}")
        self.logger.info(f"Average T-gate success probability: {avg_success_prob:.4f}")
        self.execution_time = time.time() - self.start_time
        self.logger.info(f"Circuit execution completed in {self.execution_time:.2f} seconds.")
        return avg_fidelity, logical_error_rate, success_rate, avg_success_prob

    def generate_surface_code_visualizations(self):
        self.logger.info("Generating surface code visualizations...")
        visualizations = []
        for logical_qubit in self.logical_qubits:
            before_filename = visualize_surface_code(
                surface_code=logical_qubit,
                syndrome=None,
                t_gate_qubit=None,
                filename=os.path.join(FIGURES_DIR, f"surface_code_{logical_qubit.logical_qubit_id}_before_t_{self.job_id}.png")
            )
            t_gate = next((t for t, q in self.t_gates if q == logical_qubit.logical_qubit_id), None)
            after_filename = None
            if t_gate:
                after_filename = visualize_surface_code(
                    surface_code=logical_qubit,
                    syndrome=logical_qubit.syndrome_history[-1]["syndrome"] if logical_qubit.syndrome_history else None,
                    t_gate_qubit=logical_qubit.logical_qubit_id,
                    filename=os.path.join(FIGURES_DIR, f"surface_code_{logical_qubit.logical_qubit_id}_after_t_{self.job_id}.png")
                )
            visualizations.append({
                "logicalQubitId": logical_qubit.logical_qubit_id,
                "beforeTGate": before_filename,
                "afterTGate": after_filename
            })
        self.logger.info(f"Generated visualizations: {len(visualizations)} entries.")
        return visualizations

    def generate_json_response(self, avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts):
        self.logger.info("Generating JSON response...")
        surface_codes = []
        for logical_qubit in self.logical_qubits:
            d = logical_qubit.distance
            data_qubits = [
                {
                    "id": f"D{i // d}.{i % d}",
                    "tGateApplied": False,
                    "tGateFidelity": 0.0
                } for i in range(d * d)
            ]
            t_gate = next((t for t, q in self.t_gates if q == logical_qubit.logical_qubit_id), None)
            if t_gate and len(self.t_fidelities) > self.t_gates.index((t_gate, logical_qubit.logical_qubit_id)):
                t_fidelity = self.t_fidelities[self.t_gates.index((t_gate, logical_qubit.logical_qubit_id))]
                data_qubits[logical_qubit.logical_qubit_id]["tGateApplied"] = True
                data_qubits[logical_qubit.logical_qubit_id]["tGateFidelity"] = t_fidelity

            stabilizers = [
                {
                    "id": f"S{idx}",
                    "type": stab_type,
                    "connectedQubits": [f"D{q // d}.{q % d}" for q in qubits]
                } for idx, (stab_type, qubits) in enumerate(logical_qubit.stabilizers)
            ]

            lattice_grid = []
            for i in range(d):
                for j in range(d):
                    qubit_idx = i * d + j
                    lattice_grid.append({
                        "position": [i, j],
                        "type": "data",
                        "value": f"D{i}.{j}",
                        "tGateApplied": data_qubits[qubit_idx]["tGateApplied"],
                        "tGateFidelity": data_qubits[qubit_idx]["tGateFidelity"]
                    })

            for idx, (stab_type, qubits) in enumerate(logical_qubit.stabilizers):
                centers = [(q // d, q % d) for q in qubits]
                center_x = sum(x for x, y in centers) / len(centers)
                center_y = sum(y for x, y in centers) / len(centers)
                lattice_grid.append({
                    "position": [center_x, center_y],
                    "type": "stabilizer",
                    "value": f"S{idx}-{stab_type}",
                    "connectedQubits": [f"D{q // d}.{q % d}" for q in qubits],
                    "syndromeBit": logical_qubit.syndrome_history[-1]["syndrome"][idx] if logical_qubit.syndrome_history else 0
                })

                offset_x = max(0.3 / (d / 3), 0.15) if stab_type == 'Z' else -max(0.3 / (d / 3), 0.15)
                offset_y = max(0.3 / (d / 3), 0.15) if stab_type == 'X' else -max(0.3 / (d / 3), 0.15)
                lattice_grid.append({
                    "position": [center_x + offset_x, center_y + offset_y],
                    "type": "ancilla",
                    "value": f"A{idx}",
                    "associatedStabilizer": f"S{idx}-{stab_type}"
                })

            surface_codes.append({
                "logicalQubitId": logical_qubit.logical_qubit_id,
                "distance": logical_qubit.distance,
                "dataQubits": data_qubits,
                "stabilizers": stabilizers,
                "syndrome": logical_qubit.syndrome_history[-1]["syndrome"] if logical_qubit.syndrome_history else [0] * logical_qubit.num_stabilizers,
                "errorsDetected": sum(1 for error in self.physical_errors if error["logical_qubit"] == logical_qubit.logical_qubit_id),
                "lattice": {
                    "grid": lattice_grid,
                    "dimensions": [d, d]
                }
            })

        # Compute magic state distillation metrics
        magic_state_distillation = {
            "factoryType": "on-demand",
            "inputFidelity": 0.95,  # Hardcoded as per snippet
            "noiseReduction": {
                "initial": self.noise,
                "final": self.noise * (self.msd_factory.rounds ** 2) if self.t_gates else 0.0
            },
            "outputFidelity": avg_fidelity if self.t_gates else 0.0,
            "parallelUnits": self.msd_factory.parallel_units,
            "rounds": self.msd_factory.rounds,
            "successRate": success_rate if self.t_gates else 1.0,
            "tStatesProduced": len(self.t_gates)
        }

        # Compute performance metrics
        theoretical_min_qubits = self.circuit.num_qubits * 5
        physical_qubits_efficiency = (theoretical_min_qubits / self.total_physical_qubits * 100) if self.total_physical_qubits > 0 else 0
        performance_metrics = {
            "actualLogicalErrorRate": logical_error_rate,
            "theoreticalLogicalErrorRate": self.noise ** 2,
            "alternativeApproaches": {
                "codeDistance3": {
                    "errorRate": self.noise ** 3,
                    "physicalQubits": self.circuit.num_qubits * (3 ** 2 + (3 - 1) ** 2)
                }
            },
            "physicalQubitsEfficiency": f"{physical_qubits_efficiency:.1f}% of theoretical minimum",
            "scalabilityProjections": {
                "for6Qubits": {
                    "physicalQubits": 6 * (self.distance ** 2 + (self.distance - 1) ** 2),
                    "errorRate": self.noise ** (self.distance + 1)
                }
            }
        }

        # Simulation results
        simulation_results = {
            "idealProbabilities": self.ideal_probabilities,
            "noisyCounts": self.measurement_stats,
            "simulatorSuccessProbability": self.simulator_success_prob
        }

        json_response = {
            "surfaceCodes": surface_codes,
            "performance": {
                "actualLogicalErrorRate": logical_error_rate,
                "theoreticalLogicalErrorRate": self.noise ** 2,
                "physicalQubitsUsed": self.total_physical_qubits,
                "tGateFidelity": max(self.t_fidelities) if self.t_fidelities else 0.0,
                "executionTime": round(self.execution_time, 2),
                "operationCounts": self.operation_counts
            },
            "magicStateDistillation": magic_state_distillation,
            "performanceMetrics": performance_metrics,
            "simulationResults": simulation_results,
            "visualizations": self.generate_surface_code_visualizations()
        }

        self.logger.info("JSON response generated.")
        return json.dumps(json_response, indent=2, sort_keys=True)

    def analyze_results(self, avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts):
        self.logger.info("Analyzing simulation results...")
        all_debug_logs = self.debug_logs[:]
        for logical_qubit in self.logical_qubits:
            all_debug_logs.extend(logical_qubit.debug_logs)
        all_debug_logs.extend(self.msd_factory.debug_logs)

        self.logger.info("=== Fault-Tolerant Quantum Circuit Execution Results ===")
        self.logger.info(f"Circuit: Custom Quantum Circuit")
        self.logger.info(f"Code Distance: {self.logical_qubits[0].distance if self.logical_qubits else 0}")
        self.logger.info(f"Physical Error Rate: {self.noise:.3f}")

        self.logger.info("\nLogical Measurement Results:")
        total_shots = sum(self.measurement_stats.values())
        for state, count in self.measurement_stats.items():
            confidence = (count / total_shots) * 100 if total_shots > 0 else 0
            self.logger.info(f"- State |{state}> ({count}/{total_shots} shots, {confidence:.1f}% confidence)")

        self.logger.info("\nMeasurement Statistics:")
        for state, count in self.measurement_stats.items():
            probability = count / total_shots if total_shots > 0 else 0
            self.logger.info(f"- |{state}>: {probability:.4f} ({count} shots)")

        self.logger.info("\nParity Check Results (Syndrome Measurement History):")
        for logical_qubit in self.logical_qubits:
            self.logger.info(f"Logical Qubit {logical_qubit.logical_qubit_id}:")
            for entry in logical_qubit.syndrome_history:
                self.logger.info(f"  Iteration {entry['iteration']}: Syndrome={entry['syndrome']}, Simplified={entry['simplified']}")

        self.logger.info("\nError Correction Performance:")
        detected_errors = len(self.physical_errors)
        corrected_errors = sum(1 for correction in self.error_corrections if correction["success"])
        if detected_errors == 0:
            self.logger.info("- Error Correction Success Rate: N/A (no errors detected)")
        else:
            error_correction_success_rate = (corrected_errors / detected_errors * 100)
            self.logger.info(f"- Error Correction Success Rate: {error_correction_success_rate:.1f}%")
        self.logger.info(f"- Detected Errors: {detected_errors}")
        self.logger.info(f"- Corrected Errors: {corrected_errors}")
        self.logger.info(f"- Logical Error Rate: {logical_error_rate:.6f}")

        self.logger.info("\nPhysical Error Events:")
        for error in self.physical_errors:
            self.logger.info(f"- Iteration {error['iteration']}, Logical Qubit {error['logical_qubit']}: Syndrome={error['syndrome']}, Correction={error['correction']}")

        self.logger.info("\nResource Usage:")
        self.logger.info(f"- Physical Qubits Used: {self.total_physical_qubits}")
        self.logger.info(f"- Magic States Consumed: {len(self.t_gates)}")
        distillation_qubits = self.msd_factory.total_qubits_used
        distillation_gates = self.msd_factory.total_gates_used
        distillation_time = self.msd_factory.total_time
        self.logger.info(f"- Distillation Resources: {distillation_qubits} qubits, {distillation_gates} gates, {distillation_time:.2f} seconds")
        self.logger.info(f"- MSD Factory Parallel Units: {self.msd_factory.parallel_units}")
        circuit_depth = len(self.gates) * 2
        self.logger.info(f"- Circuit Depth: {circuit_depth} cycles")

        self.logger.info("\nOperation Counts:")
        for op, count in self.operation_counts.items():
            self.logger.info(f"- {op}: {count}")

        self.logger.info("\nPerformance Metrics:")
        self.logger.info(f"- Estimated Execution Time: {self.execution_time:.2f} seconds")
        logical_operations = len(self.gates)
        error_correction_overhead = self.total_operations / logical_operations if logical_operations > 0 else 0
        self.logger.info(f"- Error Correction Overhead: {error_correction_overhead:.1f}x")
        self.logger.info(f"- Final T-Gate Fidelity: {avg_fidelity * 100:.3f}%")
        threshold_distance = self.error_rate
        threshold_performance = (threshold_distance - self.noise) / threshold_distance * 100 if threshold_distance > 0 else 0
        self.logger.info(f"- Threshold Performance: {threshold_performance:.1f}% below threshold")

        self.logger.info("\nDebugging Information:")
        self.logger.info("Error Chain Visualization (Simplified):")
        for error in self.physical_errors:
            self.logger.info(f"- Logical Qubit {error['logical_qubit']}, Iteration {error['iteration']}: Syndrome={error['syndrome']}")

        self.logger.info("\nDecoder Performance:")
        for correction in self.error_corrections:
            self.logger.info(f"- Logical Qubit {correction['logical_qubit']}, Iteration {correction['iteration']}: Success={correction['success']}")

        self.logger.info("\nFailure Points:")
        if logical_error_rate > 0:
            self.logger.info(f"- Errors accumulated in {detected_errors - corrected_errors} uncorrected events")
        else:
            self.logger.info("- No significant failure points detected")

        self.logger.info("\nCritical Path Analysis:")
        t_gate_ops = len(self.t_gates) * 150
        total_ops = self.total_operations
        t_gate_contribution = (t_gate_ops / total_ops * 100) if total_ops > 0 else 0
        self.logger.info(f"- T-Gate Operations: {t_gate_contribution:.1f}% of total operations")

        self.logger.info("\nDebug Logs:")
        for log in all_debug_logs:
            self.logger.info(f"- {log}")

        self.logger.info("\nComparative Analysis:")
        theoretical_error_rate = self.noise ** 2
        self.logger.info(f"Theoretical vs. Actual Performance:")
        self.logger.info(f"- Theoretical Logical Error Rate: {theoretical_error_rate:.6f}")
        self.logger.info(f"- Actual Logical Error Rate: {logical_error_rate:.6f}")

        self.logger.info("\nResource Efficiency:")
        theoretical_min_qubits = self.circuit.num_qubits * 5
        efficiency = (theoretical_min_qubits / self.total_physical_qubits * 100) if self.total_physical_qubits > 0 else 0
        self.logger.info(f"- Physical Qubits Efficiency: {efficiency:.1f}% of theoretical minimum")

        self.logger.info("\nAlternative Approaches:")
        self.logger.info(f"- Increasing code distance to {self.distance + 1} would use {self.circuit.num_qubits * ((self.distance + 1) ** 2 + (self.distance) ** 2)} qubits but reduce error rate to ~{(self.noise ** (self.distance + 1)):.6f}")

        self.logger.info("\nScalability Projections:")
        larger_qubits = self.circuit.num_qubits * 2
        projected_qubits = larger_qubits * (self.logical_qubits[0].distance ** 2 + (self.logical_qubits[0].distance - 1) ** 2) if self.logical_qubits else 0
        projected_error_rate = self.noise ** (self.logical_qubits[0].distance + 1) if self.logical_qubits else 0
        self.logger.info(f"- For {larger_qubits} qubits: ~{projected_qubits} physical qubits, error rate ~{projected_error_rate:.6f}")
        self.logger.info("Result analysis completed.")

    def run(self):
        self.logger.info("Starting FTQC Pipeline execution...")
        try:
            circuit_info = self.parse_circuit()
            self.setup_surface_code()
            self.apply_gates()
            msd_attempts, magic_states = self.distill_magic_states()
            if msd_attempts is None:
                self.logger.error("Magic state distillation failed. Aborting execution.")
                return None, None, None, None, None, None
            avg_fidelity, logical_error_rate, success_rate, avg_success_prob = self.execute_circuit(magic_states)
            self.analyze_results(avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts)

            self.logger.info("FTQC Pipeline execution completed successfully.")
            return avg_fidelity, logical_error_rate, success_rate, avg_success_prob, msd_attempts, self.generate_surface_code_visualizations()
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise
        finally:
            for handler in self.logger.handlers:
                handler.flush()