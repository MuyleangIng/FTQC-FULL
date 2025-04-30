from core.surface_code import SurfaceCode

def visualize_surface_code(surface_code: SurfaceCode) -> dict:
    """
    Generate visualization data for a surface code (placeholder).
    """
    return {
        "gridSize": {"rows": 2, "cols": 3},
        "dataQubits": [
            {"id": "D0.0", "row": 0, "col": 0},
            {"id": "D0.1", "row": 0, "col": 1},
            {"id": "D1.0", "row": 1, "col": 0},
            {"id": "D1.1", "row": 1, "col": 1},
            {"id": "A1.1", "row": 1, "col": 2, "type": "ancilla"}
        ],
        "stabilizers": [
            {"id": "S0.0", "row": 0.5, "col": 0.5, "type": "Z", "connectedQubits": ["D0.0", "D0.1", "D1.0", "D1.1"]},
            {"id": "S0.1", "row": 0.5, "col": 1.5, "type": "X", "connectedQubits": ["D0.1", "D1.0", "D1.1"]}
        ]
    }