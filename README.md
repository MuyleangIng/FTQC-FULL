Fault-Tolerant Quantum Computing (FTQC) API
This project implements a simulation framework for fault-tolerant quantum computing, including surface code error correction, magic state distillation, and noisy circuit execution. It uses FastAPI for API integration, Stim and Qiskit for simulations, and Pymatching for decoding.
Project Structure
ftqc_api/
├── app.py                 # FastAPI application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Project dependencies
├── api/                   # API endpoints and models
├── core/                  # Core FTQC components
├── simulation/            # Simulation utilities
└── utils/                 # Logging and visualization

Setup

Clone the repository:
git clone <repository-url>
cd ftqc_api


Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Run the FastAPI server:
uvicorn app:app --host 0.0.0.0 --port 8000


Access the API:

Open http://localhost:8000/docs for interactive API documentation.
Send POST requests to /simulate with a Base64-encoded Python source code.



Usage

Run a simulation via API:curl -X POST "http://localhost:8000/simulate" -H "Content-Type: application/json" -d '{
  "source_code": "ZnJvbSBxaXNraXQgaW1wb3J0IFF1YW50dW1DaXJjdWl0CmZyb20gbWF0aCBpbXBvcnQgcGksIGNvcywgc2luLCBzcXJ0CmZyb20gbWF0aCBpbXBvcnQgZXhwCnRoZXRhID0gcGkgLyAyCnBoaSA9IHBpIC8gNAppZiB0aGV0YSA+IHBoaToKICBxYyA9IFF1YW50dW1DaXJjdWl0KDIsIDIpCiAgcWMuaCgwKQogIHFjLmgoMSkKICBxYy5oKDEpCiAgcWMuY3goMCwgMSkKICBmb3IgaSBpbiByYW5nZSgyKToKICAgIHFjLm1lYXN1cmUoaSwgaSkKICBwcmludCgiQ2lyY3VpdCBjcmVhdGVkIHN1Y2Nlc3NmdWxseS4iKQplbHNlOgogIHByaW50KCJObyBjaXJjdWl0IGNyZWF0ZWQuIik=",
  "iterations": 10,
  "noise": 0.001,
  "distance": 3,
  "rounds": 2,
  "error_rate": 0.01,
  "debug": true
}'



Dependencies

Python 3.8+
FastAPI, Uvicorn
Stim, Qiskit, Qiskit-Aer, Pymatching
NumPy, Pydantic, python-dateutil

Security Note
The API executes Base64-encoded Python code, which is restricted to Qiskit operations but not fully sandboxed. For production, implement proper sandboxing (e.g., RestrictedPython) to prevent code injection.
License
MIT License
