Here's a complete and clear `README.md` to help users set up and run your FastAPI app with `uvicorn` and PostgreSQL via Docker:

---

### ✅ `README.md`

# FTQC Backend API

This project provides a FastAPI backend for handling fault-tolerant quantum computing (FTQC) simulations and storing results in a PostgreSQL database.

---

## 🚀 Requirements

- Python 3.10.17
- Docker & Docker Compose

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/MuyleangIng/ftqc-backend.git
cd ftqc-backend
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 🐘 PostgreSQL Setup (via Docker)

Make sure Docker is installed and running. Then start the PostgreSQL container:

```bash
cd docker && docker-compose up -d
```

* Database Name: `ftqc_db`
* Username: `postgres`
* Password: `12345`
* Host: `localhost`
* Port: `5443`

Your `.env` or config should contain:

```
DATABASE_URL=postgresql://postgres:12345@localhost:5443/ftqc_db
```

---

## 🚦 Run FastAPI Server

Start the server with Uvicorn:

```bash
uvicorn main:app --port 8000 --reload
```

* Open your browser and go to: [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI

---

## 🗃️ Tables Created

* `simulation_results`
* `job_logs`

These are initialized via `init-db.sql` when Docker PostgreSQL container starts.

---
