import logging
from fastapi import FastAPI
from api.endpoints import router as endpoints_router  # Import the router instead
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,%(msecs)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create the FastAPI app
app = FastAPI(
    title="Fault-Tolerant Quantum Computing API",
    description="API for simulating fault-tolerant quantum circuits using surface codes.",
    version="1.0.0"
)

# Allow all CORS origins (not recommended in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Include the endpoints from api/endpoints.py using a router
logging.info("Including endpoints from api/endpoints.py")
app.include_router(endpoints_router, prefix="")

# Add a root endpoint for health check
@app.get("/")
async def root():
    logging.info("Health check endpoint accessed")
    return {"message": "Fault-Tolerant Quantum Computing API is running!"}

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting Uvicorn server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
