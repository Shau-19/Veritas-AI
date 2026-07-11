import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column
import enum
from app.core.database import Base


class DatasetStatus(str, enum.Enum):
    uploading = "uploading"
    generating = "generating"
    auditing = "auditing"
    ready = "ready"
    failed = "failed"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(String(1000), nullable=True)

    # File info
    s3_key: Mapped[str] = mapped_column(String(500), nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=True)
    file_format: Mapped[str] = mapped_column(String(20), nullable=True)  # csv, jsonl

    # Audit results (populated after quality check)
    quality_score: Mapped[float] = mapped_column(Float, nullable=True)
    audit_results: Mapped[dict] = mapped_column(JSON, nullable=True)
    # audit_results shape:
    # {
    #   "duplicate_rows": int,
    #   "empty_rows": int,
    #   "avg_input_length": float,
    #   "avg_output_length": float,
    #   "issues": [{"type": str, "count": int, "severity": str}]
    # }

    status: Mapped[DatasetStatus] = mapped_column(
        Enum(DatasetStatus), default=DatasetStatus.uploading
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
