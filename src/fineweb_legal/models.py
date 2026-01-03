"""Data models for the FineWeb-Legal pipeline."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document from the FineWeb dataset."""

    id: str = Field(..., description="Unique document ID from FineWeb")
    text: str = Field(..., description="Full document text")
    url: Optional[str] = Field(None, description="Source URL if available")
    
    @property
    def text_length(self) -> int:
        """Return the character count of the text."""
        return len(self.text)


class AnnotationResult(BaseModel):
    """Result of annotating a single document."""

    id: str = Field(..., description="Document ID")
    score: int = Field(..., ge=0, le=5, description="Legal value score (0-5)")
    reasoning: str = Field("", description="Brief reasoning from the LLM")
    text: str = Field(..., description="Full original text (not truncated)")
    url: Optional[str] = Field(None, description="Source URL")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Annotation timestamp")
    model: str = Field(..., description="Model used for annotation")
    input_tokens: int = Field(0, description="Tokens sent to API (truncated text)")
    text_length: int = Field(0, description="Full text character count")


class BatchResult(BaseModel):
    """Result of processing a batch of documents."""

    batch_number: int = Field(..., description="Batch number")
    total_documents: int = Field(..., description="Documents in this batch")
    successful: int = Field(0, description="Successfully annotated")
    failed: int = Field(0, description="Failed annotations")
    annotations: list[AnnotationResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list, description="Error messages")


class PipelineState(BaseModel):
    """State of the annotation pipeline for checkpointing."""

    total_annotated: int = Field(0, description="Total documents annotated")
    total_batches: int = Field(0, description="Total batches completed")
    last_batch_number: int = Field(0, description="Last completed batch number")
    score_distribution: dict[int, int] = Field(
        default_factory=lambda: {i: 0 for i in range(6)},
        description="Count of each score (0-5)"
    )
    total_tokens_used: int = Field(0, description="Total input tokens sent to API")
    total_errors: int = Field(0, description="Total annotation errors")
    started_at: Optional[datetime] = Field(None, description="Pipeline start time")
    last_updated: Optional[datetime] = Field(None, description="Last update time")
