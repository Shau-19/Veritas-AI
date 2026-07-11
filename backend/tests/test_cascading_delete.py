import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.routes.jobs import delete_job
from app.models.job import TrainingJob, JobStatus
from app.models.deployment import Deployment, DeploymentStatus
from app.models.evaluation import EvaluationJob

async def test_cascading_delete_success():
    user_id = "user-123"
    job_id = "job-456"

    # Mock TrainingJob database record
    mock_job = TrainingJob(
        id=job_id,
        user_id=user_id,
        base_model="microsoft/phi-3-mini-4k-instruct",
        task_type="chat",
        status=JobStatus.completed,
        hf_model_id="shau1905/phi-3-python"
    )

    # Mock Deployment records:
    # 1. An active deployment that needs SageMaker teardown
    mock_dep_active = Deployment(
        id="dep-active",
        user_id=user_id,
        job_id=job_id,
        model_id="shau1905/phi-3-python",
        endpoint_name="ep-phi-3-active-endpoint",
        status=DeploymentStatus.active,
        region="us-east-1"
    )
    # 2. A provisioning deployment that doesn't need SageMaker teardown
    mock_dep_prov = Deployment(
        id="dep-prov",
        user_id=user_id,
        job_id=job_id,
        model_id="shau1905/phi-3-python",
        endpoint_name="provisioning-12345",
        status=DeploymentStatus.deploying,
        region="us-east-1"
    )
    # 3. A terminated deployment that doesn't need SageMaker teardown
    mock_dep_term = Deployment(
        id="dep-term",
        user_id=user_id,
        job_id=job_id,
        model_id="shau1905/phi-3-python",
        endpoint_name="ep-phi-3-terminated-endpoint",
        status=DeploymentStatus.terminated,
        region="us-east-1"
    )

    class MockResult:
        def __init__(self, val):
            self._val = val
        def scalar_one_or_none(self):
            return self._val
        def scalars(self):
            mock_scalars = MagicMock()
            mock_scalars.all.return_value = self._val if isinstance(self._val, list) else [self._val]
            return mock_scalars

    mock_db = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.delete = AsyncMock()
    mock_db.execute = AsyncMock()

    async def mock_execute(query):
        query_str = str(query).lower()
        if "training_jobs" in query_str:
            return MockResult(mock_job)
        elif "deployments" in query_str:
            # Return our mock deployments
            return MockResult([mock_dep_active, mock_dep_prov, mock_dep_term])
        return MockResult(None)

    mock_db.execute.side_effect = mock_execute

    with patch("app.services.sagemaker_service.terminate_sagemaker_endpoint") as mock_terminate:
        res = await delete_job(job_id=job_id, user_id=user_id, db=mock_db)
        
        # 1. Assert sagemaker terminate_sagemaker_endpoint was called ONLY for the active endpoint
        mock_terminate.assert_called_once_with(
            endpoint_name="ep-phi-3-active-endpoint",
            region="us-east-1"
        )

        # 2. Assert all deployments and the job record were deleted from db
        # 3 calls for deployments + 1 call for job = 4 calls total
        assert mock_db.delete.call_count == 4
        deleted_objects = [call[0][0] for call in mock_db.delete.call_args_list]
        assert mock_dep_active in deleted_objects
        assert mock_dep_prov in deleted_objects
        assert mock_dep_term in deleted_objects
        assert mock_job in deleted_objects

        # 3. Assert db.commit was called
        mock_db.commit.assert_called_once()
        
        print("[OK] test_cascading_delete_success passed!")

if __name__ == "__main__":
    print("=== Running Cascading Delete Unit Tests ===")
    asyncio.run(test_cascading_delete_success())
    print("=== All tests passed successfully! ===")
