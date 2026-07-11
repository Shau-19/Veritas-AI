import sys
import os
import asyncio
import json
import hmac
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.models.job import TrainingJob, JobStatus
from app.models.dataset import Dataset
from app.api.routes.jobs import verify_callback_token, CallbackPayload, CallbackCompletePayload, CallbackFailedPayload, job_callback, job_callback_complete, job_callback_failed
from app.services.trainer.local_trainer import LocalTrainer
from app.services.trainer.runpod_trainer import RunPodTrainer
from app.workers.training_worker import run_training_job

# 1. Test Callback Token HMAC Verification
def test_callback_token_verification():
    job_id = "test-job-uuid-12345"
    
    # Calculate correct token
    expected_token = hmac.new(
        settings.secret_key.encode("utf-8"),
        job_id.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    # Assert verification succeeds on valid token
    try:
        verify_callback_token(job_id, expected_token)
    except HTTPException:
        assert False, "Verification failed for a valid token"
        
    # Assert verification raises 403 on invalid token
    try:
        verify_callback_token(job_id, "invalid-token-value")
        assert False, "Expected HTTPException for invalid token"
    except HTTPException as e:
        assert e.status_code == 403
        
    print("[OK] test_callback_token_verification passed")

# 2. Test Local Trainer subprocess setup and callback token formatting
async def test_local_trainer_start():
    trainer = LocalTrainer()
    job_id = "test-local-job"
    config = {
        "base_model": "google/gemma-2b",
        "dataset_id": "dataset-123",
        "epochs": 2,
        "learning_rate": 2e-4,
        "batch_size": 4
    }
    
    # Mock asyncio.create_subprocess_exec
    mock_process = MagicMock()
    mock_process.pid = 9999
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
         patch("builtins.open", MagicMock()):
        res = await trainer.start_job(job_id, config)
        
        assert res["provider"] == "local"
        assert res["pid"] == 9999
        assert "training_test-local-job.log" in res["log_file"]
        
        # Verify subprocess was launched with the correct python script and args
        mock_exec.assert_called_once()
        args_passed = mock_exec.call_args[0]
        assert "training_script.py" in args_passed[1]
        
        # Verify callback token exists in parameters
        callback_param = next(arg for arg in args_passed if "/jobs/test-local-job/callback" in arg)
        assert "?token=" in callback_param
        
    print("[OK] test_local_trainer_start passed")

# 3. Test RunPod Trainer startup command generation
async def test_runpod_trainer_start():
    trainer = RunPodTrainer()
    job_id = "test-runpod-job"
    config = {
        "base_model": "meta-llama/Meta-Llama-3.2-3B",
        "dataset_id": "dataset-456",
        "epochs": 3,
        "learning_rate": 3e-4,
        "batch_size": 8
    }
    
    mock_pod_info = {
        "id": "runpod-pod-12345",
        "status": "RUNNING",
        "ip": "10.0.0.1"
    }
    
    with patch.object(trainer.runpod_service, "create_pod", return_value=mock_pod_info) as mock_create_pod, \
         patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock(read=lambda: "print('Hello')"))))):
        res = await trainer.start_job(job_id, config)
        
        assert res["provider"] == "runpod"
        assert res["pod_id"] == "runpod-pod-12345"
        
        # Verify command contains base64 decoding and unsloth installation
        mock_create_pod.assert_called_once()
        kwargs = mock_create_pod.call_args[1]
        assert "NVIDIA GeForce RTX 4090" in kwargs["gpu_type"]
        assert "unsloth" in kwargs["start_cmd"]
        assert "training_script.py" in kwargs["start_cmd"]
        
    print("[OK] test_runpod_trainer_start passed")

# 4. Test Callback Routes updating Database record
async def test_callback_routes():
    job_id = "test-callback-job"
    token = hmac.new(
        settings.secret_key.encode("utf-8"),
        job_id.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    # Mock TrainingJob model instance
    mock_job = TrainingJob(
        id=job_id,
        status=JobStatus.queued,
        metrics_history=[]
    )
    
    # Mock db session
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: mock_job))
    
    # A. Test Live step callback (training state, step, loss, epoch, metrics history)
    body = CallbackPayload(
        status="training",
        step=5,
        epoch=1,
        total_steps=10,
        loss=1.85,
        lr=2e-4
    )
    await job_callback(job_id, body, token, mock_session)
    assert mock_job.status == JobStatus.training
    assert mock_job.current_step == 5
    assert mock_job.train_loss == 1.85
    assert len(mock_job.metrics_history) == 1
    assert mock_job.metrics_history[0]["step"] == 5
    assert mock_job.metrics_history[0]["loss"] == 1.85

    # B. Test Duplicate steps filter
    body_dup = CallbackPayload(
        status="training",
        step=5,
        epoch=1,
        total_steps=10,
        loss=1.85,
        lr=2e-4
    )
    await job_callback(job_id, body_dup, token, mock_session)
    assert len(mock_job.metrics_history) == 1 # still 1, duplicate step ignored

    # C. Test Complete callback
    body_comp = CallbackCompletePayload(
        hf_model_id="user/fine-tuned-model",
        actual_cost_usd=0.15,
        gpu_seconds_used=1200
    )
    await job_callback_complete(job_id, body_comp, token, mock_session)
    assert mock_job.status == JobStatus.completed
    assert mock_job.hf_model_id == "user/fine-tuned-model"
    assert mock_job.actual_cost_usd == 0.15
    assert mock_job.gpu_seconds_used == 1200
    assert mock_job.completed_at is not None

    # D. Test Failed callback
    body_failed = CallbackFailedPayload(
        error_message="OutOfMemory Error"
    )
    await job_callback_failed(job_id, body_failed, token, mock_session)
    assert mock_job.status == JobStatus.failed
    assert mock_job.error_message == "OutOfMemory Error"
    assert mock_job.completed_at is not None
    
    print("[OK] test_callback_routes passed")

# 5. Test Worker validation and failsafe teardown loop
def test_worker_orchestration_and_safety():
    job_id = "test-worker-job"
    
    # Mock database session & records using pure MagicMocks to bypass SQLAlchemy descriptor issues
    mock_job = MagicMock()
    mock_job.id = job_id
    mock_job.dataset_id = "d123"
    mock_job.training_config = {"epochs": 3, "max_seq_length": 512, "batch_size": 4}
    mock_job.status = JobStatus.queued
    
    mock_dataset = MagicMock()
    mock_dataset.row_count = 5050
    
    mock_session = MagicMock()
    # Mock querying dataset and job
    mock_session.query.return_value.filter.return_value.first.side_effect = [mock_job, mock_dataset]
    
    # Mock create_engine
    with patch("sqlalchemy.create_engine", MagicMock()), \
         patch("sqlalchemy.orm.Session", return_value=mock_session), \
         patch("app.services.trainer.get_trainer") as mock_get_trainer:
        
        mock_trainer = MagicMock()
        mock_trainer.stop_job = AsyncMock(return_value={"status": "stopped"})
        mock_get_trainer.return_value = mock_trainer
        
        # Run Celery task synchronously
        try:
            call_count = 0
            def side_effect_first(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_job
                elif call_count == 2:
                    return mock_dataset
                return MagicMock()
                
            mock_session.__enter__.return_value.query.return_value.filter.return_value.first.side_effect = side_effect_first
            
            run_training_job.run(job_id)
            assert False, "Should have raised ValueError due to dataset limit > 5000 rows"
        except ValueError as e:
            assert "exceeds limit of 5000 rows" in str(e)
            
        # Verify stop_job is called in finally block even on validation exception
        mock_trainer.stop_job.assert_called_once_with(job_id)

    print("[OK] test_worker_orchestration_and_safety passed")

if __name__ == "__main__":
    print("=== Running Layer 3 GPU Training Integration tests ===")
    test_callback_token_verification()
    asyncio.run(test_local_trainer_start())
    asyncio.run(test_runpod_trainer_start())
    asyncio.run(test_callback_routes())
    test_worker_orchestration_and_safety()
    print("=== All Layer 3 Training Integration tests passed! ===")
