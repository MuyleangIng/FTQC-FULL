import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle, Circle


# Plot 1: Surface Code Lattice (d=2)
def plot_surface_code():
    fig, ax = plt.subplots(figsize=(6, 6))

    # Data qubits (circles)
    data_qubits = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    for x, y in data_qubits:
        ax.add_patch(Circle((x + 0.5, y + 0.5), 0.2, color='blue', label='Data Qubit' if (x, y) == (0, 0) else ''))

    # Ancilla qubits (squares)
    ancilla_qubits = [(0.5, 0.5, 'X'), (1.5, 0.5, 'X'), (0.5, 1.5, 'Z'), (1.5, 1.5, 'Z')]
    for x, y, t in ancilla_qubits:
        ax.add_patch(Rectangle((x, y), 0.4, 0.4, color='red' if t == 'X' else 'green',
                               label=f'{t}-Ancilla' if (x, y) == (0.5, 0.5) else ''))

    # Stabilizer connections
    for x, y, t in ancilla_qubits:
        if t == 'X':
            # X-type: connect to 4 nearest data qubits
            for dx, dy in [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]:
                if (x + dx - 0.5, y + dy - 0.5) in data_qubits:
                    ax.plot([x + 0.2, x + dx + 0.5], [y + 0.2, y + dy + 0.5], 'k-')
        else:
            # Z-type: similar connections
            for dx, dy in [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]:
                if (x + dx - 0.5, y + dy - 0.5) in data_qubits:
                    ax.plot([x + 0.2, x + dx + 0.5], [y + 0.2, y + dy + 0.5], 'k-')

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.set_title("Surface Code Lattice (d=2)")
    ax.legend()
    plt.savefig('surface_code.png')
    plt.close()


# Plot 2: FTQC Pipeline Flowchart
def plot_ftqc_pipeline():
    G = nx.DiGraph()
    nodes = [
        "1. Quantum Circuit",
        "2. Gate Decomposition",
        "3. Logical Encoding",
        "4. Magic State Injection",
        "5. Fault-Tolerant Gates",
        "6. Syndrome Measurement",
        "7. Syndrome Decoding",
        "8. Apply Corrections",
        "9. Logical Measurement"
    ]
    for i, node in enumerate(nodes):
        G.add_node(node, pos=(0, -i))
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])

    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(6, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, arrowsize=20)
    plt.title("FTQC Pipeline Flowchart")
    plt.savefig('ftqc_pipeline.png')
    plt.close()


if __name__ == "__main__":
    plot_surface_code()
    plot_ftqc_pipeline()