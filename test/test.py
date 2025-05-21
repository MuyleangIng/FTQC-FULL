from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from math import gcd
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from fractions import Fraction

# Function to implement controlled modular multiplication by a (for a=7, N=15)
def controlled_mod_mult(circuit, ctrl, target, value, aux, N=15):
    """Apply controlled multiplication by 'value' mod N on target register."""
    # For a=7, N=15: 7^1 mod 15 = 7, 7^2 mod 15 = 4, 7^4 mod 15 = 1
    # Simulate multiplication by controlled operations on target qubits
    if value == 7:  # a^1 mod 15
        if ctrl is not None:
            circuit.cx(ctrl, target[0])  # Controlled bit flip
            circuit.cx(ctrl, target[1])
    elif value == 4:  # a^2 mod 15
        if ctrl is not None:
            circuit.cswap(ctrl, target[0], target[1])  # Controlled swap
    elif value == 1:  # a^4 mod 15
        pass  # Identity operation (no gates needed)
    # Note: aux qubits are unused in this simplified version

# Function to implement QFT on counting qubits
def qft(circuit, n):
    """Apply QFT on the first n qubits."""
    for i in range(n):
        circuit.h(i)
        for j in range(i + 1, n):
            circuit.cp(3.14159 / (2 ** (j - i)), j, i)
    # Swap qubits to reverse order
    for i in range(n // 2):
        circuit.swap(i, n - 1 - i)

# Set up the problem: Factor N = 15 using a coprime base a
N = 15
a = 7  # Coprime with N=15

# Check if classical shortcut gives a factor
if gcd(a, N) != 1:
    print(f"{a} and {N} are not coprime. Found factor: {gcd(a, N)}")
else:
    print(f"Trying to factor N = {N} using a = {a}")

    # Create a quantum circuit: 3 counting qubits, 4 work qubits, 3 classical bits
    qc = QuantumCircuit(7, 3)

    # Initialize work register to |1>
    qc.x(3)  # Set q3 to |1> (work register starts at 1)

    # Apply Hadamard to counting register (q0, q1, q2)
    for i in range(3):
        qc.h(i)

    # Barrier for clarity
    qc.barrier()

    # Modular exponentiation: a^x mod N
    for i in range(3):
        # Apply controlled U^(2^i) where U is multiplication by a mod N
        controlled_mod_mult(qc, i, [3, 4, 5, 6], a ** (2 ** i) % N, None, N)
        qc.barrier()

    # Apply QFT to counting register
    qft(qc, 3)

    # Measure counting register
    qc.measure([0, 1, 2], [0, 1, 2])

    # Visualize the circuit (graphical)
    print("\nQuantum Circuit Diagram (Graphical):")
    qc.draw(output='mpl', filename='full_shor_circuit.png')
    plt.show()

    # Visualize the circuit (text)
    print("\nQuantum Circuit Diagram (Text):")
    print(qc.draw(output='text'))

    # Run simulation
    simulator = AerSimulator()
    result = simulator.run(qc, shots=1024).result()
    counts = result.get_counts()

    # Output measurement results
    print("\nMeasurement Result (Shor's Algorithm):")
    print(counts)

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plot_histogram(counts)
    plt.title("Shor's Algorithm Results (N=15, a=7)")
    plt.savefig('shor_histogram.png')
    plt.show()

    # Classical post-processing to find factors
    print("\nPost-Processing to Find Factors:")
    for measured in counts:
        # Convert binary string to integer (e.g., '010' -> 2)
        measured_int = int(measured, 2)
        # Phase is measured_int / 2^n, where n=3
        phase = measured_int / (2 ** 3)
        # Use continued fractions to find r
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        # Try to find factors
        if r % 2 == 0:  # Check if period is even
            guess_plus = gcd(pow(a, r // 2, N) + 1, N)
            guess_minus = gcd(pow(a, r // 2, N) - 1, N)
            factors = set([guess_plus, guess_minus])
            if 1 < guess_plus < N or 1 < guess_minus < N:
                print(f"Measured: {measured} (phase={phase:.3f}), period r={r}, factors={factors}")
            else:
                print(f"Measured: {measured} (phase={phase:.3f}), period r={r}, no factor found")
        else:
            print(f"Measured: {measured} (phase={phase:.3f}), period r={r}, odd period, no factor")