import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column
import enum
from app.core.database import Base


class DeploymentStatus(str, enum.Enum):
    deploying = "deploying"
    active = "active"
    failed = "failed"
    terminated = "terminated"


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(String, ForeignKey("training_jobs.id"), nullable=True)  # Null if base model hosting

    model_id: Mapped[str] = mapped_column(String(255), nullable=False)  # e.g. "shau1905/python_qa" or "microsoft/phi-3-mini-4k-instruct"
    endpoint_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    instance_type: Mapped[str] = mapped_column(String(100), default="ml.g5.xlarge")
    region: Mapped[str] = mapped_column(String(50), default="us-east-1")

    status: Mapped[DeploymentStatus] = mapped_column(Enum(DeploymentStatus), default=DeploymentStatus.deploying)
    error_message: Mapped[str] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    terminated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
