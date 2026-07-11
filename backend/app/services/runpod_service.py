"""
RunPod REST API service wrapper for GPU instance management.
"""
import json
import urllib.request
import urllib.error
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class RunPodService:
    def __init__(self):
        self.api_key = settings.runpod_api_key
        # REST API endpoint for instance management
        self.base_url = "https://api.runpod.io/v1"

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def create_pod(self, name: str, image_name: str, gpu_type: str, start_cmd: list, env: dict = None) -> dict:
        """
        Provision a new GPU pod instance on RunPod.
        """
        url = f"{self.base_url}/pods"
        payload = {
            "name": name,
            "imageName": image_name,
            "gpuTypeIds": [gpu_type],
            "gpuCount": 1,
            "cloudType": "SECURE",
            "containerDiskInGb": 50,
            "volumeInGb": 20,
            "volumeMountPath": "/workspace",
            "dockerStartCmd": start_cmd,
            "env": env or {},
            "ports": "22/tcp,8000/http"
        }
        
        # Graceful fallback mock if API key is not configured
        if not self.api_key or "placeholder" in self.api_key or "your-runpod" in self.api_key:
            logger.warning("[RunPod] API Key is missing or placeholder. Mocking Pod Creation.")
            return {
                "id": f"mock-pod-{name}",
                "name": name,
                "imageName": image_name,
                "gpuTypeIds": [gpu_type],
                "status": "RUNNING",
                "ip": "127.0.0.1"
            }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            error_data = e.read().decode("utf-8")
            logger.error(f"[RunPod] HTTP Error during pod creation: {error_data}")
            raise RuntimeError(f"RunPod Creation Failed: {error_data}")
        except Exception as e:
            logger.error(f"[RunPod] Error during pod creation: {e}")
            raise

    def get_pod_status(self, pod_id: str) -> dict:
        """
        Retrieve current runtime details of a pod.
        """
        if pod_id.startswith("mock-pod-"):
            return {"status": "RUNNING", "ip": "127.0.0.1"}

        url = f"{self.base_url}/pods/{pod_id}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"[RunPod] Failed to get status for pod {pod_id}: {e}")
            return {"status": "TERMINATED"}

    def terminate_pod(self, pod_id: str) -> dict:
        """
        Teardown a GPU pod to avoid billing leakage.
        """
        if pod_id.startswith("mock-pod-"):
            logger.info(f"[RunPod] Terminating Mock Pod {pod_id}")
            return {"status": "TERMINATED"}

        url = f"{self.base_url}/pods/{pod_id}"
        req = urllib.request.Request(url, headers=self._headers(), method="DELETE")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"[RunPod] Failed to terminate pod {pod_id}: {e}")
            return {"status": "TERMINATED"}
