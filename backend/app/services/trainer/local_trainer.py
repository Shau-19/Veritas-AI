"""
Local execution provider for VeritasAI training.
"""
import sys
import os
import asyncio
import psutil
from typing import Dict, Any
from app.services.trainer.base import BaseTrainer
from app.core.config import settings

class LocalTrainer(BaseTrainer):
    async def start_job(self, job_id: str, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start the training job locally as a subprocess.
        """
        base_model = training_config.get("base_model", "meta-llama/Meta-Llama-3.2-3B")
        dataset_id = training_config.get("dataset_id")
        epochs = training_config.get("epochs", 3)
        lr = training_config.get("learning_rate", 2e-4)
        batch_size = training_config.get("batch_size", 4)
        lora_r = training_config.get("lora_r", 16)
        lora_alpha = training_config.get("lora_alpha", 32)
        lora_dropout = training_config.get("lora_dropout", 0.05)
        max_seq_length = training_config.get("max_seq_length", 512)
        warmup_steps = training_config.get("warmup_steps", 10)
        
        # Point to the local backend callback route with HMAC-SHA256 token
        import hmac
        import hashlib
        token = hmac.new(
            settings.secret_key.encode("utf-8"),
            job_id.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        callback_url = f"http://127.0.0.1:8000/api/v1/jobs/{job_id}/callback?token={token}"
        
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "training_script.py"
        )
        
        python_exe = sys.executable
        
        cmd = [
            python_exe,
            script_path,
            "--job_id", job_id,
            "--base_model", base_model,
            "--dataset_id", dataset_id,
            "--epochs", str(epochs),
            "--learning_rate", str(lr),
            "--batch_size", str(batch_size),
            "--lora_r", str(lora_r),
            "--lora_alpha", str(lora_alpha),
            "--lora_dropout", str(lora_dropout),
            "--max_seq_length", str(max_seq_length),
            "--warmup_steps", str(warmup_steps),
            "--callback_url", callback_url,
            "--hf_token", settings.hf_token or ""
        ]
        
        # Setup log files in the scratch directory
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "scratch"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"training_{job_id}.log")
        
        # Open in write mode, subprocess will hold this file handle
        log_file = open(log_file_path, "w", encoding="utf-8")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_file,
            stderr=log_file
        )
        
        # We can close our reference to the file handle in python main thread;
        # the subprocess inherits the file descriptor and writes directly.
        log_file.close()
        
        return {
            "provider": "local",
            "pod_id": str(process.pid),
            "pid": process.pid,
            "log_file": log_file_path
        }

    async def stop_job(self, job_id: str) -> Dict[str, Any]:
        """
        Kill the local training process by PID retrieved from Database.
        """
        from app.core.database import AsyncSessionLocal
        from app.models.job import TrainingJob
        from sqlalchemy import select
        
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one_or_none()
            if not job or not job.runpod_pod_id:
                return {"status": "not_running"}
            
            try:
                pid = int(job.runpod_pod_id)
                if psutil.pid_exists(pid):
                    p = psutil.Process(pid)
                    # Terminate process and its children recursively
                    for child in p.children(recursive=True):
                        try:
                            child.terminate()
                        except Exception:
                            pass
                    p.terminate()
                    return {"status": "stopped", "pid": pid}
            except (ValueError, TypeError, psutil.NoSuchProcess):
                pass
                
            return {"status": "already_stopped"}

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check if the local training process is running.
        """
        from app.core.database import AsyncSessionLocal
        from app.models.job import TrainingJob
        from sqlalchemy import select
        
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one_or_none()
            if not job or not job.runpod_pod_id:
                return {"status": "failed", "error": "Job not found or missing PID"}
                
            try:
                pid = int(job.runpod_pod_id)
                if psutil.pid_exists(pid):
                    p = psutil.Process(pid)
                    if p.status() == psutil.STATUS_ZOMBIE:
                        return {"status": "completed"}
                    return {"status": "running"}
            except (ValueError, TypeError, psutil.NoSuchProcess):
                pass
                
            return {"status": "finished"}
