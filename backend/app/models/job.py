import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, DateTime, JSON, ForeignKey, Enum, Text
from sqlalchemy.orm import Mapped, mapped_column
import enum
from app.core.database import Base


class JobStatus(str, enum.Enum):
    queued = "queued"
    provisioning = "provisioning"   # spinning up RunPod GPU
    training = "training"
    evaluating = "evaluating"
    uploading = "uploading"         # pushing to HuggingFace
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.id"), nullable=False)

    # Model config
    base_model: Mapped[str] = mapped_column(String(100), nullable=False)
    # e.g. "meta-llama/Meta-Llama-3.2-3B", "mistralai/Mistral-7B-v0.1"
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # e.g. "chat", "completion", "classification"

    # Training hyperparameters
    training_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    # {
    #   "epochs": int,
    #   "learning_rate": float,
    #   "batch_size": int,
    #   "lora_r": int,
    #   "lora_alpha": int,
    #   "lora_dropout": float,
    #   "max_seq_length": int,
    #   "warmup_steps": int
    # }

    # Runtime
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued)
    runpod_pod_id: Mapped[str] = mapped_column(String(100), nullable=True)
    celery_task_id: Mapped[str] = mapped_column(String(100), nullable=True)

    # Progress (updated live)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, nullable=True)
    train_loss: Mapped[float] = mapped_column(Float, nullable=True)
    metrics_history: Mapped[dict] = mapped_column(JSON, default=list)
    # [{"step": int, "loss": float, "lr": float, "epoch": int}]

    # Cost tracking
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=True)
    actual_cost_usd: Mapped[float] = mapped_column(Float, nullable=True)
    gpu_seconds_used: Mapped[int] = mapped_column(Integer, default=0)

    # Output
    hf_model_id: Mapped[str] = mapped_column(String(200), nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
