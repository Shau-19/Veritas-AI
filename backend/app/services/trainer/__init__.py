import os
from app.services.trainer.base import BaseTrainer
from app.services.trainer.local_trainer import LocalTrainer
from app.services.trainer.runpod_trainer import RunPodTrainer
from app.services.trainer.colab_trainer import ColabTrainer


def get_trainer(provider: str | None = None) -> BaseTrainer:
    """
    Factory to return the active trainer provider.

    Sections:
      local  — simulation / subprocess on this machine (default for dev)
      colab  — generates a pre-filled Colab notebook via GitHub Gist
      runpod — dispatches to a serverless RunPod GPU pod
    """
    if provider is None:
        provider = os.getenv("TRAINING_PROVIDER", "local").strip().lower()
    else:
        provider = provider.strip().lower()

    if provider == "runpod":
        return RunPodTrainer()
    if provider == "colab":
        return ColabTrainer()
    return LocalTrainer()

