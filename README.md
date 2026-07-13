# VeritasAI: Enterprise-Grade LLM Fine-Tuning & Dataset Curation Platform

**VeritasAI** is a comprehensive, production-ready environment built for synthetic dataset curation, LLM fine-tuning orchestration using PEFT (LoRA/QLoRA), and comparative performance auditing using an automated asynchronous evaluation engine.

---

## 1. System Architecture

VeritasAI decouples the stateless FastAPI gateway from dynamic compute resources and asynchronous workers to ensure reliable, high-performance execution of long-running GPU processes.

```mermaid
graph TB
    %% Client Tier
    subgraph Client Tier [Next.js SPA Client]
        UI[Brutalist UI App]
    end

    %% API Tier
    subgraph Gateway Tier [FastAPI Gateway]
        GW[APIRouter]
        Auth[JWT Security]
        HMAC[HMAC Callback Validator]
        DB[(PostgreSQL DB)]
    end

    %% Broker Tier
    subgraph Broker Tier [Redis Message Queue]
        Broker((Celery Broker))
    end

    %% Compute Tier
    subgraph Compute Worker Tier [Celery Worker Cluster]
        Worker[Celery Task Consumer]
        Local[Local Subprocess Trainer]
        Cloud[RunPod GPU Provisioner]
        Colab[Colab SSH Tunnel Trainer]
    end

    %% GPU Node Tier
    subgraph Ephemeral GPU Node
        US[Unsloth GPU PEFT Engine]
        Script[training_script.py]
    end

    %% Data Flows
    UI <--->|1. SSE Streams / HTTP REST| GW
    GW <--->|2. Read/Write State| DB
    GW --->|3. Dispatch Async Job| Broker
    Broker <--->|4. Fetch task| Worker
    Worker -->|5. Strategy Pattern Select| Local & Cloud & Colab
    Cloud --->|6. Provision Ephemeral Pod| Ephemeral GPU Node
    Script --->|7. HTTP callback + HMAC Token| HMAC
```

---

## 2. Core Backend Design Principles

### A. Unidirectional SSE (Server-Sent Events) over WebSockets
Rather than bidirectional WebSockets, VeritasAI leverages **Server-Sent Events (SSE)** for streaming telemetry logs and dataset generations.
* **Firewall Compatibility:** SSE operates strictly over standard HTTP (`text/event-stream`), avoiding corporate firewall blocks common to arbitrary WebSocket connections.
* **Auto-Reconnection:** Modern browsers natively handle reconnections and automatic retries for SSE, significantly reducing client-side code complexity.
* **Low Gateway Overhead:** Avoids maintaining memory-heavy WebSocket connection state tables on the FastAPI server.

### B. Decoupled Compute via Celery Task Isolation
Fine-tuning LLMs is a long-running, blocking operation (ranging from 10 minutes to several hours). 
* **Gateway Responsiveness:** Running training loops directly in HTTP threads would cause gateway timeouts. Offloading tasks to Celery ensures the API remains fast and responsive.
* **OOM (Out-of-Memory) Isolation:** Celery spawns isolated worker subprocesses. If a GPU training run triggers an OOM segment fault, it only crashes that specific worker, leaving the primary web gateway completely unaffected.

### C. Strategy Pattern for Compute Providers
The platform supports model training on local hardware, Google Colab notebooks, and RunPod GPU clusters. 
* **Pluggable Architecture:** Code interactions with cloud providers are abstracted behind a unified interface (`BaseTrainer` exposing `start_job`, `stop_job`, and `get_status`). If the system migrates to another GPU cluster provider, only a new subclass needs to be written; the core API routes remain untouched.

### D. HMAC-SHA256 Callback Authentication
Compute containers provisioned on ephemeral hosts like RunPod do not have static IP addresses.
* **Zero-Trust Network Perimeter:** Whitelisting static IPs is impossible. To secure the status update `/callback` endpoint, the gateway issues a transient cryptographically signed token compiled with an HMAC-SHA256 signature using the `job_id` and the server's private `SECRET_KEY`.

### E. Auto-Termination Safeguards (Cost Control)
GPU resources accrue substantial billing by the second.
* **Finally-Block Guarantees:** Celery task executions wrap cloud compute provisioning inside `try-finally` blocks. No matter the outcome (successful finish, user cancellation, runtime errors, or timeout), container teardown requests are guaranteed to run.
* **Hard Timeouts:** Celery tasks enforce a hard runtime limit (default 2 hours) to auto-terminate runaway processes.

---

## 3. Technology Stack

* **Frontend:** Next.js (TypeScript), TailwindCSS, Brutalist UI Design.
* **API Gateway:** FastAPI (Python), Pydantic validation, JWT Security, SQLAlchemy AsyncPG.
* **Task Broker & Worker:** Redis + Celery.
* **Database:** PostgreSQL.
* **Fine-Tuning Engine:** Unsloth (4-bit Quantization + PEFT LoRA) running on NVIDIA GPUs.
* **Evaluation Suite:** LLM-as-Judge scoring using concurrent `asyncio` batching.

---

## 4. Quickstart Guide

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+

### Step 1: Spin up local infrastructure
Use Docker Compose to launch PostgreSQL and Redis:
```bash
docker compose -f docker/docker-compose.yml up -d
```

### Step 2: Configure & run the Backend
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv platform
   platform\Scripts\activate   # On Windows
   source platform/bin/activate # On Unix/macOS
   cd backend
   pip install -r req.txt
   ```
2. Create a `.env` file in the `backend/` directory referencing your Database and Redis URLs:
   ```env
   DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/veritas
   REDIS_URL=redis://localhost:6379/0
   SECRET_KEY=your-jwt-and-hmac-secret-key
   OPENAI_API_KEY=your-openai-api-key
   ```
3. Run migrations and launch the FastAPI server:
   ```bash
   alembic upgrade head
   uvicorn app.main:app --reload --port 8000
   ```
4. Start the Celery worker (in a separate terminal inside your virtual environment):
   ```bash
   celery -A app.workers.training_worker.celery_app worker --loglevel=info
   ```

### Step 3: Launch the Frontend
1. Move to the `frontend/` directory and install packages:
   ```bash
   cd frontend
   npm install
   ```
2. Start the local development server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:3000` in your browser.

---

## 5. Model Evaluation Suite

The evaluation suite executes semantic comparisons between the baseline model and your fine-tuned model using an **LLM-as-Judge** scoring engine.

* **Metric Coverage:** Computes ROUGE-1/2/L, Token Match Accuracy, Cosine Semantic Similarity, and LLM-as-judge score (1.0 - 5.0 scale).
* **High Concurrency:** Grades comparisons in concurrent batches using Python's `asyncio.gather`, reducing evaluation runtimes by **90%** compared to sequential API requests.

To run the local unit and integration test suite:
```bash
cd backend
pytest tests/ -v
```

---

## 6. Model Evaluation Metrics Visualization

The platform includes a utility script to render model performance and latency metrics comparison charts. To generate the comparison chart:
```bash
cd backend
python tests/visualize_metrics.py
```

The output is saved under `backend/tests/metrics_comparison.png` and is displayed below:

![Model Performance & Inference Latency Audit](backend/tests/metrics_comparison.png)
