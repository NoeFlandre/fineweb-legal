"""Configuration management for the FineWeb-Legal pipeline."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnnotationConfig(BaseSettings):
    """Configuration for the annotation pipeline.
    
    Settings are loaded from environment variables with FINEWEB_ prefix,
    or from a .env file in the current directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Settings
    mistral_api_key: Optional[SecretStr] = Field(
        None,
        description="Mistral AI API key (required for live runs)",
        alias="MISTRAL_API_KEY",
    )
    mistral_model: str = Field(
        "mistral-medium-latest",
        description="Mistral model to use for annotation",
        alias="MISTRAL_MODEL",
    )

    # Dataset Settings
    dataset_name: str = Field(
        "HuggingFaceFW/fineweb",
        description="HuggingFace dataset name",
    )
    dataset_split: str = Field(
        "train",
        description="Dataset split to use",
    )
    dataset_subset: Optional[str] = Field(
        "sample-10BT",
        description="Dataset subset/config (e.g., 'sample-10BT' for smaller subset)",
    )
    target_samples: int = Field(
        100_000,
        ge=1,
        description="Target number of samples to annotate",
        alias="TARGET_SAMPLES",
    )
    batch_size: int = Field(
        1000,
        ge=1,
        le=10000,
        description="Documents per batch file",
        alias="BATCH_SIZE",
    )

    # Rate Limiting & Retry
    requests_per_minute: int = Field(
        60,
        ge=1,
        le=1000,
        description="Maximum API requests per minute",
        alias="REQUESTS_PER_MINUTE",
    )
    max_retries: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum retry attempts for API calls",
    )
    retry_base_delay: float = Field(
        1.0,
        ge=0.1,
        le=30.0,
        description="Base delay between retries (seconds)",
    )
    request_delay: float = Field(
        2.0,
        ge=0.0,
        le=30.0,
        description="Delay between individual API requests (seconds)",
        alias="REQUEST_DELAY",
    )

    # Storage
    output_dir: Path = Field(
        Path("./data"),
        description="Output directory for batch files",
        alias="OUTPUT_DIR",
    )

    # Text Filtering & Truncation
    min_text_length: int = Field(
        1000,
        ge=10,
        description="Minimum text length (chars) to include",
    )
    max_text_length: int = Field(
        50_000,
        ge=1000,
        description="Maximum text length (chars) for storage filtering",
    )
    inference_truncation_chars: int = Field(
        3000,
        ge=500,
        le=20000,
        description="Maximum chars sent to API (cost control)",
        alias="INFERENCE_TRUNCATION_CHARS",
    )
    
    # Pre-filtering
    keyword_filter_enabled: bool = Field(
        True,
        description="Enable legal keyword pre-filtering to reduce API costs",
        alias="KEYWORD_FILTER_ENABLED",
    )

    # Concurrency
    max_concurrent_requests: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum concurrent API requests",
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Convert string to Path if needed."""
        return Path(v) if isinstance(v, str) else v

    def get_batches_dir(self) -> Path:
        """Get the batches subdirectory path."""
        return self.output_dir / "batches"

    def ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.get_batches_dir().mkdir(parents=True, exist_ok=True)


def load_config(**overrides: object) -> AnnotationConfig:
    """Load configuration from environment and apply overrides.
    
    Args:
        **overrides: Configuration values to override.
        
    Returns:
        Loaded and validated configuration.
    """
    # Load from env/.env file first
    config = AnnotationConfig(**overrides)  # type: ignore[arg-type]
    return config
