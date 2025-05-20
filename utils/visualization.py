import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from config import FIGURES_DIR

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

def visualize_surface_code(surface_code, syndrome=None, t_gate_qubit=None, filename=None):
    """
    Visualize a surface code lattice of given distance with data qubits, stabilizers, and optional syndromes/T-gates.
    """
    d = surface_code.distance
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, d + 0.5)
    ax.set_ylim(-0.5, d + 0.5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(range(d + 1))
    ax.set_yticks(range(d + 1))
    ax.set_title(f"Surface Code (Logical Qubit {surface_code.logical_qubit_id}, Distance {d})")

    # Data qubits as a dxd grid
    data_qubits = np.arange(d * d).reshape(d, d)
    qubit_radius = max(0.2 / (d / 3), 0.1)  # Scale radius with distance
    for i in range(d):
        for j in range(d):
            qubit_idx = data_qubits[i, j]
            color = 'blue'
            linewidth = 2 if qubit_idx == t_gate_qubit else 1
            edgecolor = 'red' if qubit_idx == t_gate_qubit else 'black'
            ax.add_patch(plt.Circle((j, d - 1 - i), qubit_radius, color=color, ec=edgecolor, lw=linewidth))
            ax.text(j, d - 1 - i, f'D{qubit_idx}', ha='center', va='center', fontsize=max(8 / (d / 3), 4))

    # Stabilizers
    for idx, (stab_type, qubits) in enumerate(surface_code.stabilizers):
        centers = [(q % d, d - 1 - (q // d)) for q in qubits]
        center_x = sum(x for x, y in centers) / len(centers)
        center_y = sum(y for x, y in centers) / len(centers)
        color = 'yellow' if stab_type == 'Z' else 'green'
        size = max(0.4 / (d / 3), 0.2)  # Scale stabilizer size
        ax.add_patch(plt.Rectangle(
            (center_x - size/2, center_y - size/2), size, size,
            color=color, alpha=0.5, ec='black'
        ))
        ax.text(center_x, center_y, f'{stab_type}{idx}', ha='center', va='center', fontsize=max(8 / (d / 3), 4))

        if syndrome and idx < len(syndrome) and syndrome[idx] == 1:
            ax.add_patch(plt.Circle((center_x, center_y), qubit_radius / 2, color='red', zorder=10))

    # Ancilla qubits
    for idx, (stab_type, qubits) in enumerate(surface_code.stabilizers):
        center_x = sum((q % d) for q in qubits) / len(qubits)
        center_y = sum(d - 1 - (q // d) for q in qubits) / len(qubits)
        offset_x = max(0.3 / (d / 3), 0.15) if stab_type == 'Z' else -max(0.3 / (d / 3), 0.15)
        offset_y = max(0.3 / (d / 3), 0.15) if stab_type == 'X' else -max(0.3 / (d / 3), 0.15)
        ax.add_patch(plt.Circle((center_x + offset_x, center_y + offset_y), qubit_radius / 2, color='gray', ec='black'))
        ax.text(center_x + offset_x, center_y + offset_y, f'A{idx}', ha='center', va='center', fontsize=max(6 / (d / 3), 3))

    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(FIGURES_DIR, f"surface_code_{surface_code.logical_qubit_id}_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename