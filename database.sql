CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    status TEXT NOT NULL,
    parameters JSON NOT NULL,
    response_path TEXT,
    execution_time FLOAT,
    key_metrics JSON,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    created_at DATETIME NOT NULL
);

-- Example insert
INSERT INTO jobs (
    job_id, user_id, timestamp, status, parameters, response_path, execution_time, key_metrics
) VALUES (
    '550e8400-e29b-41d4-a716-446655440000',
    '1',
    '2025-05-02 10:03:47',
    'completed',
    '{"source_code": "ZnJvbSBxaXNraXQgaW1wb3J0IFF1YW50dW1DaXJjdWl0CnFjID0gUXVhbnR1bUNpcmN1aXQoMiwgMikKcWMuaCgwKQpxYy5oKDEpCnFjLnQoMCkKcWMudCgxKQpxYy5jeCgwLCAxKQpxYy5tZWFzdXJlKDAsIDApCnFjLm1lYXN1cmUoMSwgMSkKcHJpbnQoIkNpcmN1aXQgd2l0aCBULWdhdGVzIGNyZWF0ZWQgc3VjY2Vzc2Z1bGx5LiIp", "iterations": 10, "noise": 0.05, "distance": 3, "rounds": 2, "error_rate": 0.01, "debug": true}',
    'logs/user_1/responses/550e8400-e29b-41d4-a716-446655440000.json',
    5.123,
    '{"actual_error_rate": 0.0, "theoretical_error_rate": 0.0025, "output_fidelity": 0.999976, "physical_qubits": 18, "efficiency": 0.556}'
);