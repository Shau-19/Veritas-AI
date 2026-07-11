"""
Abstract base class for VeritasAI fine-tuning providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTrainer(ABC):
    @abstractmethod
    async def start_job(self, job_id: str, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start the training job. Returns job identifier/metadata.
        """
        pass

    @abstractmethod
    async def stop_job(self, job_id: str) -> Dict[str, Any]:
        """
        Stop/cancel the training job. Force cleanup/termination.
        """
        pass

    @abstractmethod
    async def get_status(self, job_id: str) -> Dict[str, Any]:
        """
        Query status from the provider.
        """
        pass
