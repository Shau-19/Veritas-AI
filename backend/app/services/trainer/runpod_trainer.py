"""
RunPod execution provider for VeritasAI training.
"""
import base64
import os
import hmac
import hashlib
from typing import Dict, Any
from app.services.trainer.base import BaseTrainer
from app.services.runpod_service import RunPodService
from app.core.config import settings

class RunPodTrainer(BaseTrainer):
    def __init__(self):
        self.runpod_service = RunPodService()

    def _generate_callback_token(self, job_id: str) -> str:
        """
        Generate a secure token signature for callback validation.
        """
        return hmac.new(
            settings.secret_key.encode("utf-8"),
            job_id.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def start_job(self, job_id: str, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provision an RTX 4090 pod and run the training script.
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
        
        # Determine the external backend callback base URL
        callback_base = os.getenv("BACKEND_CALLBACK_URL", "http://127.0.0.1:8000")
        token = self._generate_callback_token(job_id)
        callback_url = f"{callback_base}/api/v1/jobs/{job_id}/callback?token={token}"
        
        # Read and base64-encode the local training_script.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "training_script.py")
        
        with open(script_path, "r", encoding="utf-8") as f:
            script_code = f.read()
            
        b64_script = base64.b64encode(script_code.encode("utf-8")).decode("utf-8")
        
        # Construct start command
        # Installs Unsloth and required training packages, base64-decodes training_script, and executes it.
        # RunPod's dockerStartCmd requires a single string or command list. We use a string executed by bash.
        start_cmd = (
            f"bash -c \"echo '{b64_script}' | base64 -d > training_script.py && "
            f"pip install --upgrade pip && "
            f"pip install --no-cache-dir 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' trl peft transformers accelerate bitsandbytes && "
            f"python3 -u training_script.py "
            f"--job_id '{job_id}' "
            f"--base_model '{base_model}' "
            f"--dataset_id '{dataset_id}' "
            f"--epochs {epochs} "
            f"--learning_rate {lr} "
            f"--batch_size {batch_size} "
            f"--lora_r {lora_r} "
            f"--lora_alpha {lora_alpha} "
            f"--lora_dropout {lora_dropout} "
            f"--max_seq_length {max_seq_length} "
            f"--warmup_steps {warmup_steps} "
            f"--callback_url '{callback_url}' "
            f"--hf_token '{settings.hf_token or ''}'\""
        )
        
        # We default to an RTX 4090 GPU (id: "NVIDIA GeForce RTX 4090")
        gpu_type = "NVIDIA GeForce RTX 4090"
        image_name = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
        
        pod_name = f"veritas-train-{job_id[:8]}"
        
        # Provision the pod
        pod_info = self.runpod_service.create_pod(
            name=pod_name,
            image_name=image_name,
            gpu_type=gpu_type,
            start_cmd=start_cmd,
            env={
                "HF_TOKEN": settings.hf_token or "",
                "RUNPOD_API_KEY": settings.runpod_api_key or ""
            }
        )
        
        pod_id = pod_info.get("id")
        if not pod_id:
            raise RuntimeError(f"RunPod creation did not return a pod ID: {pod_info}")
            
        return {
            "provider": "runpod",
            "pod_id": pod_id,
            "pod_info": pod_info
        }

    async def stop_job(self, job_id: str) -> Dict[str, Any]:
        """
        Terminate the RunPod container instance to prevent billing leakage.
        """
        from app.core.database import AsyncSessionLocal
        from app.models.job import TrainingJob
        from sqlalchemy import select
        
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one_or_none()
            if not job or not job.runpod_pod_id:
                return {"status": "not_running"}
                
            pod_id = job.runpod_pod_id
            term_result = self.runpod_service.terminate_pod(pod_id)
            return {
                "status": "stopped",
                "pod_id": pod_id,
                "result": term_result
            }

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieve current pod status from RunPod API.
        """
        from app.core.database import AsyncSessionLocal
        from app.models.job import TrainingJob
        from sqlalchemy import select
        
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one_or_none()
            if not job or not job.runpod_pod_id:
                return {"status": "failed", "error": "Job not found or missing pod ID"}
                
            pod_id = job.runpod_pod_id
            status_info = self.runpod_service.get_pod_status(pod_id)
            pod_status = status_info.get("status")
            
            # Map RunPod state to standard training/lifecycle provider state
            # RunPod states: 'COMMITTED', 'COMPLETED', 'EXITED', 'PENDING', 'RUNNING', 'STOPPED', 'TERMINATED'
            if pod_status in ["PENDING"]:
                return {"status": "provisioning"}
            elif pod_status in ["RUNNING"]:
                return {"status": "running"}
            elif pod_status in ["TERMINATED", "COMPLETED", "EXITED"]:
                return {"status": "finished"}
            else:
                return {"status": "unknown", "runpod_status": pod_status}
