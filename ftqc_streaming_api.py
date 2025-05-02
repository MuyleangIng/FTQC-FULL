
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import time
import json
import os
import logging

app = FastAPI()

LOG_FILE = "logs/ftqc_pipeline.log"
JSON_OUTPUT = "logs/ftqc_output.json"

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# Configure logging to file
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# In-memory buffer for simulation output (for demo)
simulation_data = {
    "status": "not_started",
    "log": [],
    "result": {}
}

@app.get("/stream/logs")
async def stream_logs():
    def event_stream():
        with open(LOG_FILE, "r") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line}\n\n"
                else:
                    time.sleep(0.1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/result")
async def get_result():
    if os.path.exists(JSON_OUTPUT):
        with open(JSON_OUTPUT, "r") as f:
            result = json.load(f)
        return JSONResponse(content=result)
    return {"message": "Result not available yet."}

@app.post("/simulate")
async def simulate(background_tasks: BackgroundTasks):
    def run_simulation():
        import subprocess
        logging.info("Starting FTQC simulation...")
        subprocess.run(["python", "core/pipeline.py"], check=False)
        logging.info("FTQC simulation completed.")
    background_tasks.add_task(run_simulation)
    return {"message": "Simulation started"}
