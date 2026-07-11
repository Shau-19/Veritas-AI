import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, DateTime, JSON, ForeignKey, Enum, Text
from sqlalchemy.orm import Mapped, mapped_column
import enum
from app.core.database import Base

class EvaluationStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"

class EvaluationJob(Base):
    __tablename__ = "evaluation_jobs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False, index=True)
    training_job_id: Mapped[str] = mapped_column(String, ForeignKey("training_jobs.id"), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String, ForeignKey("datasets.id"), nullable=False)

    status: Mapped[EvaluationStatus] = mapped_column(Enum(EvaluationStatus), default=EvaluationStatus.queued)
    
    # Summary Metrics (e.g. ROUGE, Accuracy, Latency, Similarity)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=True)
    
    # Comparison data containing detailed generation inputs and outputs
    comparison_data: Mapped[dict] = mapped_column(JSON, nullable=True)

    error_message: Mapped[str] = mapped_column(Text, nullable=True)

    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
