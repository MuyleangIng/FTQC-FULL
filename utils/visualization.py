import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from config import FIGURES_DIR  # Updated to use FIGURES_DIR

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

def visualize_surface_code(surface_code, syndrome=None, t_gate_qubit=None, filename=None):
    """
    Visualize a distance-3 surface code with data qubits, stabilizers, and optional syndromes/T-gates.

    Args:
        surface_code: SurfaceCode object with stabilizers and qubit info.
        syndrome: List of syndrome bits (length 8) to highlight errors.
        t_gate_qubit: Qubit index where a T-gate is applied (for red border).
        filename: Path to save the plot (default: figures/surface_code_<timestamp>.png).
    """
    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_title(f"Surface Code (Logical Qubit {surface_code.logical_qubit_id}, Distance {surface_code.distance})")

    # Data qubits (3x3 grid, 9 qubits)
    data_qubits = np.arange(9).reshape(3, 3)
    for i in range(3):
        for j in range(3):
            qubit_idx = data_qubits[i, j]
            color = 'blue'
            linewidth = 2 if qubit_idx == t_gate_qubit else 1
            edgecolor = 'red' if qubit_idx == t_gate_qubit else 'black'
            ax.add_patch(plt.Circle((j, 2 - i), 0.2, color=color, ec=edgecolor, lw=linewidth))
            ax.text(j, 2 - i, f'D{qubit_idx}', ha='center', va='center', fontsize=8)

    # Stabilizers (4 Z, 4 X)
    for idx, (stab_type, qubits) in enumerate(surface_code.stabilizers):
        # Compute center of stabilizer (average of qubit positions)
        centers = [(q % 3, 2 - (q // 3)) for q in qubits]
        center_x = sum(x for x, y in centers) / len(centers)
        center_y = sum(y for x, y in centers) / len(centers)
        color = 'yellow' if stab_type == 'Z' else 'green'
        ax.add_patch(plt.Rectangle(
            (center_x - 0.2, center_y - 0.2), 0.4, 0.4,
            color=color, alpha=0.5, ec='black'
        ))
        ax.text(center_x, center_y, f'{stab_type}{idx}', ha='center', va='center', fontsize=8)

        # Highlight syndrome if non-zero
        if syndrome and idx < len(syndrome) and syndrome[idx] == 1:
            ax.add_patch(plt.Circle((center_x, center_y), 0.1, color='red', zorder=10))

    # Ancilla qubits (place near stabilizers)
    for idx, (stab_type, qubits) in enumerate(surface_code.stabilizers):
        center_x = sum((q % 3) for q in qubits) / len(qubits)
        center_y = sum(2 - (q // 3) for q in qubits) / len(qubits)
        offset_x = 0.3 if stab_type == 'Z' else -0.3
        offset_y = 0.3 if stab_type == 'X' else -0.3
        ax.add_patch(plt.Circle((center_x + offset_x, center_y + offset_y), 0.1, color='gray', ec='black'))
        ax.text(center_x + offset_x, center_y + offset_y, f'A{idx}', ha='center', va='center', fontsize=6)

    # Save or show plot
    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(FIGURES_DIR,
                                f"surface_code_{surface_code.logical_qubit_id}_{timestamp}.png")  # Updated to FIGURES_DIR
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename