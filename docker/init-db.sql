-- Create the simulation_results table with updated constraints
CREATE TABLE IF NOT EXISTS simulation_results (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    job_id VARCHAR(50) UNIQUE NOT NULL,
    simulation_id VARCHAR(50) UNIQUE,
    result JSONB,
    timestamp VARCHAR(50) NOT NULL,
    status VARCHAR(20)
);

-- Create the job_logs table
CREATE TABLE IF NOT EXISTS job_logs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(50) UNIQUE NOT NULL,
    log_file VARCHAR(255) NOT NULL,
    timestamp VARCHAR(50) NOT NULL
);
