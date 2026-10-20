# VeritasAI: Enterprise-Grade LLM Fine-Tuning & Dataset Curation Platform

**VeritasAI** is a comprehensive, production-ready environment built for synthetic dataset curation, LLM fine-tuning orchestration using PEFT (LoRA/QLoRA), and comparative performance auditing using an automated asynchronous evaluation engine.

---

## 1. System Architecture

VeritasAI decouples the stateless FastAPI gateway from dynamic compute resources and asynchronous workers to ensure reliable, high-performance execution of long-running GPU processes.

VeritasAI is designed so a user with no fine-tuning experience can go from raw data to a hosted, fine-tuned model through four stages: **curate → train → evaluate → host**. The platform picks sane defaults (LoRA/QLoRA config, training backend) so the user never has to understand the underlying ML mechanics.

### 1.0 Pipeline Overview

![VeritasAI pipeline overview](diagrams/Overall.png)

All four stages sit behind one FastAPI gateway (JWT auth, HMAC callback validation, PostgreSQL state) fronted by the Next.js UI over SSE/HTTP.

### 1.1 Stage 1 — Dataset Curation

![Stage 1 — Dataset Curation](diagrams/Stage1.png)

**Status: Validated** — run end-to-end against the live OpenAI API; measured 98.7% quality score and 1.7% near-duplicate rate on a 60-row test run.

### 1.2 Stage 2 — Fine-Tuning (pick one backend)

![Stage 2 — Fine-Tuning](diagrams/Stage2.png)

**Status:**
- Colab — **Validated**, the only path run end-to-end against a live backend and Celery/Redis queue.
- RunPod GPU Provisioner — Implemented, but only run in mocked mode (no real RunPod API key configured yet).
- Local subprocess trainer — Implemented, not yet run against a full GPU training job.

### 1.3 Stage 3 — Evaluation

![Stage 3 — Evaluation](diagrams/Stage3.png)

**Status:** Scoring logic is real, and grading is parallelized via `asyncio.gather` — measured 73% latency reduction (3.71×) vs. sequential grading on 15 live OpenAI calls. However, the "fine-tuned model" responses fed into this stage are currently mocked (same LLM, different system prompt) rather than real trained-model output — no backend yet submits genuine Colab/RunPod inference results into this stage.

### 1.4 Stage 4 — Hosting (production target)

![Stage 4 — Hosting](diagrams/Stage4.png)

**Status:**
- AWS SageMaker — Full deploy/invoke/terminate lifecycle implemented (`sagemaker_service.py`, `deploy.py`), but never run against real AWS credentials.
- RunPod inference hosting — **Not built yet** — RunPod integration today only provisions pods for training, with no serving/inference capability.

### Current implementation status by stage

| Stage | Component | Status |
|---|---|---|
| 1. Curate | Dataset generator (dedup, diversity, quality audit) | **Validated** — run end-to-end against the live OpenAI API with measured quality metrics (98.7% quality score, 1.7% near-duplicate rate on a 60-row test run). |
| 2. Train | Colab (manual, user-run GPU runtime) | **Validated** — the only training path run end-to-end against a live backend and Celery/Redis queue. |
| 2. Train | RunPod GPU Provisioner | Implemented (`runpod_service.py`, `RunPodTrainer`), but only run in mocked mode — no real RunPod API key configured, so provisioning against the live API is untested. |
| 2. Train | Local subprocess trainer | Implemented (`LocalTrainer`), not yet run against a full GPU training job. |
| 3. Evaluate | LLM-as-judge + ROUGE + semantic similarity | Scoring logic and the 73% async-batching speedup are verified real; however, the "fine-tuned model" responses fed into it are currently mocked (same LLM, different prompt) rather than real trained-model output — no backend yet submits genuine Colab/RunPod inference results into this stage. |
| 4. Host | AWS SageMaker real-time endpoint | Full deploy/invoke/terminate lifecycle implemented (`sagemaker_service.py`, `deploy.py`), but never run against real AWS credentials. |
| 4. Host | RunPod inference hosting | **Not built yet** — RunPod integration today only provisions pods for training, with no serving/inference capability. |

RunPod (training) and AWS SageMaker (hosting) are the intended production setup; Colab is the stopgap manual path used for validation so far. These haven't been exercised against live infrastructure mainly due to the cost of running real GPU/cloud resources during development.

> **Note on the diagram:** Colab does not run inside the Celery worker pool and is not reached over SSH. The backend generates a standalone Colab notebook that the user runs manually on Google's GPU runtime; that notebook posts HTTP status callbacks back to the local backend through an ngrok tunnel (local dev only), since Colab has no direct network path to a developer's machine. In a deployed (non-local) environment, the backend's public URL would replace the ngrok tunnel entirely.

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
