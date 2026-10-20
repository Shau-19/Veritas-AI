import time
import uuid
import sys
import logging

# Configure app logger console handler to bypass uvicorn suppression
app_logger = logging.getLogger("app")
app_logger.setLevel(logging.INFO)
if not app_logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
    )
    app_logger.addHandler(console_handler)
    app_logger.propagate = False

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.core.config import settings
from app.core.database import init_db
from app.api.routes import auth, datasets, jobs, generate, structured, evaluations, admin
from app.api.routes import ingest, deploy

access_logger = logging.getLogger("veritas.access")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB (run DDL)
    await init_db()
    # Start ngrok tunnel for local dev so Colab notebooks can reach this backend
    from app.services.tunnel import start_tunnel
    start_tunnel(port=8000)
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.3.0",
    description="Fine-tuning platform — P1: Structured datasets + Repair engine",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_origin_regex="https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes are registered below

app.include_router(auth.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(generate.router, prefix="/api/v1")
app.include_router(structured.router, prefix="/api/v1")
app.include_router(evaluations.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(deploy.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.3.0"}


@app.get("/api/v1/tunnel/status")
async def tunnel_status():
    """Return the current public URL for Colab notebook generation."""
    from app.services.tunnel import get_tunnel_status
    return get_tunnel_status()

