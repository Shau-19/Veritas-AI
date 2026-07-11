import sys
import os
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import sagemaker_service
from app.api.routes.deploy import (
    create_deployment,
    list_deployments,
    get_deployment,
    terminate_deployment,
    invoke_deployment,
    DeployRequest,
    TerminateRequest,
    InvokeDeployRequest,
)
from app.models.deployment import Deployment, DeploymentStatus
from app.models.job import TrainingJob, JobStatus


# 1. Test SageMaker Service Layer
def test_sagemaker_service():
    mock_sagemaker_client = MagicMock()
    mock_sagemaker_client.create_model.return_value = {}
    mock_sagemaker_client.create_endpoint_config.return_value = {}
    mock_sagemaker_client.create_endpoint.return_value = {}
    mock_sagemaker_client.describe_endpoint.return_value = {
        "EndpointStatus": "InService",
        "EndpointConfigName": "test-cfg"
    }
    mock_sagemaker_client.describe_endpoint_config.return_value = {
        "ProductionVariants": [{"ModelName": "test-model"}]
    }
    mock_sagemaker_client.delete_endpoint.return_value = {}
    mock_sagemaker_client.delete_endpoint_config.return_value = {}
    mock_sagemaker_client.delete_model.return_value = {}

    mock_iam_client = MagicMock()
    mock_iam_client.list_roles.return_value = {
        "Roles": [{"RoleName": "AmazonSageMaker-ExecutionRole", "Arn": "arn:aws:iam::123:role/sagemaker"}]
    }

    mock_runtime_client = MagicMock()
    mock_body = MagicMock()
    mock_body.read.return_value = b'[{"generated_text": "Hello world response"}]'
    mock_runtime_client.invoke_endpoint.return_value = {"Body": mock_body}

    def mock_client_routing(service_name, *args, **kwargs):
        if service_name == "sagemaker":
            return mock_sagemaker_client
        elif service_name == "iam":
            return mock_iam_client
        elif service_name == "sagemaker-runtime":
            return mock_runtime_client
        return MagicMock()

    with patch("boto3.client", side_effect=mock_client_routing):
        # A. Test deploy_sagemaker_endpoint
        ep_name = sagemaker_service.deploy_sagemaker_endpoint(
            model_id="shau1905/python_qa",
            hf_token="test-hf-token",
            access_key="fake-key",
            secret_key="fake-secret",
            region="us-east-1",
        )
        assert ep_name.startswith("ep-python-qa")
        assert mock_sagemaker_client.create_model.called
        assert mock_sagemaker_client.create_endpoint_config.called
        assert mock_sagemaker_client.create_endpoint.called

        # B. Test check_endpoint_status
        status = sagemaker_service.check_endpoint_status(
            endpoint_name=ep_name,
            access_key="fake-key",
            secret_key="fake-secret",
            region="us-east-1",
        )
        assert status == "InService"
        assert mock_sagemaker_client.describe_endpoint.called

        # C. Test invoke_sagemaker_endpoint
        response = sagemaker_service.invoke_sagemaker_endpoint(
            endpoint_name=ep_name,
            prompt="Hello world",
            access_key="fake-key",
            secret_key="fake-secret",
            region="us-east-1",
        )
        assert response == "Hello world response"
        assert mock_runtime_client.invoke_endpoint.called

        # D. Test terminate_sagemaker_endpoint
        sagemaker_service.terminate_sagemaker_endpoint(
            endpoint_name=ep_name,
            access_key="fake-key",
            secret_key="fake-secret",
            region="us-east-1",
        )
        assert mock_sagemaker_client.delete_endpoint.called
        assert mock_sagemaker_client.delete_endpoint_config.called
        assert mock_sagemaker_client.delete_model.called

    print("[OK] test_sagemaker_service passed")


# 2. Test API Routes
async def test_api_routes():
    user_id = "user-123"
    
    # Mock databases objects
    mock_job = TrainingJob(
        id="job-abc",
        user_id=user_id,
        base_model="microsoft/phi-3-mini-4k-instruct",
        task_type="chat",
        status=JobStatus.completed,
        hf_model_id="shau1905/phi-3-python"
    )

    mock_deployment = Deployment(
        id="dep-123",
        user_id=user_id,
        job_id="job-abc",
        model_id="shau1905/phi-3-python",
        endpoint_name="ep-phi-3-python-123",
        instance_type="ml.g5.xlarge",
        region="us-east-1",
        status=DeploymentStatus.active
    )

    class MockResult:
        def __init__(self, val):
            self._val = val
        def scalar_one_or_none(self):
            return self._val
        def scalars(self):
            mock_scalars = MagicMock()
            mock_scalars.all.return_value = [self._val] if self._val else []
            return mock_scalars

    mock_db = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    # Define mock return for DB calls
    async def mock_execute(query):
        # Crude check to simulate query returns
        query_str = str(query).lower()
        if "training_jobs" in query_str:
            return MockResult(mock_job)
        elif "deployments" in query_str:
            return MockResult(mock_deployment)
        return MockResult(None)

    mock_db.execute = mock_execute

    # A. Test List Deployments Route
    deps = await list_deployments(user_id=user_id, db=mock_db)
    assert len(deps) == 1
    assert deps[0].id == "dep-123"
    assert deps[0].status == "active"

    # B. Test Invoke Route
    with patch("app.services.sagemaker_service.invoke_sagemaker_endpoint", return_value="SageMaker live output"):
        body = InvokeDeployRequest(prompt="Test prompt", aws_access_key="key", aws_secret_key="secret")
        inv_res = await invoke_deployment(dep_id="dep-123", body=body, user_id=user_id, db=mock_db)
        assert inv_res["response"] == "SageMaker live output"

    # C. Test Terminate Route
    with patch("app.services.sagemaker_service.terminate_sagemaker_endpoint", return_value=None):
        term_body = TerminateRequest(aws_access_key="key", aws_secret_key="secret")
        term_res = await terminate_deployment(dep_id="dep-123", body=term_body, user_id=user_id, db=mock_db)
        assert term_res["message"] == "SageMaker endpoint successfully deleted."
        assert mock_deployment.status == DeploymentStatus.terminated

    print("[OK] test_api_routes passed")


if __name__ == "__main__":
    print("=== Running AWS SageMaker Deployment & Live Inference Routing Tests ===")
    test_sagemaker_service()
    asyncio.run(test_api_routes())
    print("=== All tests passed successfully! ===")
