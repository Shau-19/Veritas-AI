import logging
import uuid
import json
import boto3
from typing import Dict, Any, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)


def get_tgi_container_uri(region: str) -> str:
    """Returns the official Hugging Face TGI Deep Learning Container URI for a given region."""
    # Standard account ID for Deep Learning Containers in most AWS regions
    account_id = "763104351884"
    if region.startswith("cn-"):
        account_id = "727897471807"
    elif region.startswith("us-gov-"):
        account_id = "442386744353"
    
    # We use a stable, tested TGI release (v2.4.0, PyTorch 2.4.0, CUDA 12.4)
    return f"{account_id}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-tgi-inference:2.4.0-tgi2.4.0-gpu-py311-cu124-ubuntu22.04"


def _get_aws_credentials(
    access_key: str = None,
    secret_key: str = None,
    region: str = None
) -> Tuple[str, str, str]:
    """Resolves AWS credentials, falling back to local .env configuration settings."""
    resolved_key = access_key or settings.aws_access_key_id
    resolved_secret = secret_key or settings.aws_secret_access_key
    resolved_region = region or settings.aws_region or "us-east-1"
    return resolved_key, resolved_secret, resolved_region


def get_sagemaker_client(access_key: str = None, secret_key: str = None, region: str = None):
    key, secret, reg = _get_aws_credentials(access_key, secret_key, region)
    if key and secret:
        return boto3.client(
            "sagemaker",
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=reg
        )
    return boto3.client("sagemaker", region_name=reg)


def get_sagemaker_runtime_client(access_key: str = None, secret_key: str = None, region: str = None):
    key, secret, reg = _get_aws_credentials(access_key, secret_key, region)
    if key and secret:
        return boto3.client(
            "sagemaker-runtime",
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=reg
        )
    return boto3.client("sagemaker-runtime", region_name=reg)


def get_iam_client(access_key: str = None, secret_key: str = None, region: str = None):
    key, secret, reg = _get_aws_credentials(access_key, secret_key, region)
    if key and secret:
        return boto3.client(
            "iam",
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=reg
        )
    return boto3.client("iam", region_name=reg)


def get_execution_role(iam_client, role_arn_override: str = None) -> str:
    """Attempts to find a SageMaker Execution Role in the AWS account or uses override."""
    if role_arn_override:
        return role_arn_override
    if settings.aws_role_arn:
        return settings.aws_role_arn

    try:
        roles = iam_client.list_roles(MaxItems=100)["Roles"]
        # 1. Look for custom SageMaker roles, skipping service-linked roles
        for role in roles:
            name = role["RoleName"]
            arn = role["Arn"]
            if "aws-service-role" in arn.lower() or "aws-service-role" in role.get("Path", "").lower():
                continue
            if "sagemaker" in name.lower():
                return arn
        
        # 2. Fall back to the first non-service-linked role available
        for role in roles:
            arn = role["Arn"]
            if "aws-service-role" not in arn.lower() and "aws-service-role" not in role.get("Path", "").lower():
                logger.warning(f"No role with 'sagemaker' in the name found. Falling back to role: {role['RoleName']}")
                return arn
    except Exception as e:
        logger.warning(f"Could not list IAM roles dynamically: {e}")
        
    raise ValueError(
        "No SageMaker execution role found. Please configure 'aws_role_arn' in your settings or .env file, or provide a role ARN override."
    )


def get_clean_aws_name(prefix: str, identifier: str) -> str:
    """Generates a valid SageMaker resource name (max 63 chars, alphanumeric + hyphens)."""
    clean_id = identifier.replace("/", "-").replace(".", "-").replace("_", "-").replace("@", "-").lower()
    name = f"{prefix}-{clean_id}"
    if len(name) > 50:
        name = name[:50]
    suffix = uuid.uuid4().hex[:8]
    return f"{name}-{suffix}"


def deploy_sagemaker_endpoint(
    model_id: str,
    hf_token: str = None,
    access_key: str = None,
    secret_key: str = None,
    region: str = None,
    role_arn: str = None,
    instance_type: str = "ml.g5.xlarge",
    lora_adapter: str = None,
    endpoint_name: str = None,
) -> str:
    """
    Deploys a Hugging Face model (custom or base) on AWS SageMaker.
    Returns the created endpoint name.
    """
    _, _, reg = _get_aws_credentials(access_key, secret_key, region)
    
    sagemaker = get_sagemaker_client(access_key, secret_key, region)
    iam = get_iam_client(access_key, secret_key, region)
    
    resolved_role = get_execution_role(iam, role_arn)
    container_uri = get_tgi_container_uri(reg)
    
    # Generate unique resources names
    short_id = model_id.split("/")[-1]
    if not endpoint_name:
        endpoint_name = get_clean_aws_name("ep", short_id)
    model_name = get_clean_aws_name("model", short_id)
    config_name = get_clean_aws_name("cfg", short_id)
    
    # 1. Create SageMaker Model
    env_vars = {
        "HF_MODEL_ID": model_id,
        "HF_TASK": "text-generation",
        "MAX_INPUT_LENGTH": "2000",
        "MAX_TOTAL_TOKENS": "4096",
        "SM_NUM_GPUS": "1"
    }
    if lora_adapter:
        env_vars["LORA_ADAPTERS"] = lora_adapter
    if hf_token or settings.hf_token:
        env_vars["HF_TOKEN"] = hf_token or settings.hf_token

    logger.info(f"Creating SageMaker Model: {model_name}...")
    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": container_uri,
            "Environment": env_vars,
        },
        ExecutionRoleArn=resolved_role,
    )

    # 2. Create Endpoint Configuration
    logger.info(f"Creating SageMaker Endpoint Config: {config_name} on {instance_type}...")
    sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    # 3. Create SageMaker Endpoint
    logger.info(f"Deploying SageMaker Endpoint: {endpoint_name}...")
    sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )
    
    return endpoint_name


def check_endpoint_status(
    endpoint_name: str,
    access_key: str = None,
    secret_key: str = None,
    region: str = None
) -> str:
    """Queries SageMaker to check status. Returns status string (e.g. 'InService', 'Creating')."""
    sagemaker = get_sagemaker_client(access_key, secret_key, region)
    try:
        desc = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        return desc["EndpointStatus"]
    except Exception as e:
        logger.error(f"Error describing endpoint {endpoint_name}: {e}")
        return "Failed"


def invoke_sagemaker_endpoint(
    endpoint_name: str,
    prompt: str,
    access_key: str = None,
    secret_key: str = None,
    region: str = None,
    adapter_id: str = None,
) -> str:
    """Invokes the live endpoint and returns the generated text response."""
    runtime = get_sagemaker_runtime_client(access_key, secret_key, region)
    
    # Format payload for Hugging Face TGI container
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "return_full_text": False
        }
    }
    if adapter_id:
        payload["parameters"]["adapter_id"] = adapter_id
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    
    body = json.loads(response["Body"].read().decode("utf-8"))
    
    # TGI returns a list of dictionaries with 'generated_text'
    if isinstance(body, list) and len(body) > 0:
        return body[0].get("generated_text", "")
    elif isinstance(body, dict):
        return body.get("generated_text", "")
    return str(body)


def terminate_sagemaker_endpoint(
    endpoint_name: str,
    access_key: str = None,
    secret_key: str = None,
    region: str = None
):
    """Cleanly deletes Endpoint, Configuration, and Model records from AWS SageMaker."""
    sagemaker = get_sagemaker_client(access_key, secret_key, region)
    
    # 1. Fetch Endpoint Config Name
    config_name = None
    try:
        desc = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        config_name = desc.get("EndpointConfigName")
    except Exception as e:
        logger.warning(f"Could not describe endpoint {endpoint_name} during cleanup: {e}")
        
    # 2. Fetch Model Name from Config
    model_names = []
    if config_name:
        try:
            config_desc = sagemaker.describe_endpoint_config(EndpointConfigName=config_name)
            for variant in config_desc.get("ProductionVariants", []):
                model_names.append(variant.get("ModelName"))
        except Exception as e:
            logger.warning(f"Could not describe endpoint config {config_name} during cleanup: {e}")

    # 3. Delete Endpoint
    try:
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Deleted endpoint {endpoint_name}")
    except Exception as e:
        logger.error(f"Failed to delete endpoint {endpoint_name}: {e}")

    # 4. Delete Endpoint Configuration
    if config_name:
        try:
            sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
            logger.info(f"Deleted endpoint config {config_name}")
        except Exception as e:
            logger.error(f"Failed to delete endpoint config {config_name}: {e}")

    # 5. Delete Model records
    for model_name in model_names:
        if model_name:
            try:
                sagemaker.delete_model(ModelName=model_name)
                logger.info(f"Deleted model {model_name}")
            except Exception as e:
                logger.error(f"Failed to delete model {model_name}: {e}")
