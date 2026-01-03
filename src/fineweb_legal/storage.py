"""Partitioned Parquet storage with batch-file checkpointing."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from fineweb_legal.models import AnnotationResult, PipelineState

logger = logging.getLogger(__name__)

# Parquet schema for annotation output
ANNOTATION_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("score", pa.int8()),
        ("reasoning", pa.string()),
        ("text", pa.string()),
        ("url", pa.string()),
        ("timestamp", pa.timestamp("us")),
        ("model", pa.string()),
        ("input_tokens", pa.int32()),
        ("text_length", pa.int32()),
    ]
)


class AnnotationStorage:
    """Manages partitioned Parquet storage with batch-file checkpointing.
    
    Each batch is written as an immutable file (batch_XXXXXX.parquet).
    If a crash occurs, only the current batch is lost—all previous
    batches remain intact.
    """

    def __init__(self, output_dir: Path):
        """Initialize storage.
        
        Args:
            output_dir: Base output directory.
        """
        self.output_dir = output_dir
        self.batches_dir = output_dir / "batches"
        self.state_file = output_dir / "pipeline_state.json"
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

    def get_batch_files(self) -> list[Path]:
        """Get all batch files sorted by batch number.
        
        Returns:
            Sorted list of batch file paths.
        """
        files = list(self.batches_dir.glob("batch_*.parquet"))
        return sorted(files)

    def get_next_batch_number(self) -> int:
        """Count existing batch files to determine next batch number.
        
        Returns:
            Next batch number (1-indexed).
        """
        existing = self.get_batch_files()
        if not existing:
            return 1
        
        # Extract batch numbers and find max
        numbers = []
        for f in existing:
            try:
                num = int(f.stem.split("_")[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue
        
        return max(numbers, default=0) + 1

    def _results_to_table(self, results: list[AnnotationResult]) -> pa.Table:
        """Convert annotation results to PyArrow table.
        
        Args:
            results: List of annotation results.
            
        Returns:
            PyArrow table with schema.
        """
        # Build column arrays
        data = {
            "id": [r.id for r in results],
            "score": [r.score for r in results],
            "reasoning": [r.reasoning for r in results],
            "text": [r.text for r in results],
            "url": [r.url for r in results],
            "timestamp": [r.timestamp for r in results],
            "model": [r.model for r in results],
            "input_tokens": [r.input_tokens for r in results],
            "text_length": [r.text_length for r in results],
        }
        
        return pa.Table.from_pydict(data, schema=ANNOTATION_SCHEMA)

    def write_batch(self, batch_num: int, results: list[AnnotationResult]) -> Path:
        """Write a single immutable batch file.
        
        Uses atomic write (temp file + rename) to prevent corruption.
        
        Args:
            batch_num: Batch number for filename.
            results: Annotation results to write.
            
        Returns:
            Path to the written batch file.
        """
        filename = f"batch_{batch_num:06d}.parquet"
        filepath = self.batches_dir / filename
        temp_path = filepath.with_suffix(".parquet.tmp")
        
        # Convert to PyArrow table
        table = self._results_to_table(results)
        
        # Atomic write: write to temp file, then rename
        try:
            pq.write_table(table, temp_path, compression="snappy")
            temp_path.rename(filepath)
            logger.info(f"Wrote batch {batch_num} with {len(results)} samples to {filepath}")
            return filepath
        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

    def get_total_annotated(self) -> int:
        """Count total rows across all batch files.
        
        Returns:
            Total number of annotated samples.
        """
        total = 0
        for f in self.get_batch_files():
            try:
                metadata = pq.read_metadata(f)
                total += metadata.num_rows
            except Exception as e:
                logger.warning(f"Failed to read metadata from {f}: {e}")
        return total

    def get_score_distribution(self) -> dict[int, int]:
        """Calculate score distribution across all batches.
        
        Returns:
            Dictionary mapping score (0-5) to count.
        """
        distribution = {i: 0 for i in range(6)}
        
        for f in self.get_batch_files():
            try:
                table = pq.read_table(f, columns=["score"])
                scores = table["score"].to_pylist()
                for score in scores:
                    if 0 <= score <= 5:
                        distribution[score] += 1
            except Exception as e:
                logger.warning(f"Failed to read scores from {f}: {e}")
        
        return distribution

    def save_state(self, state: PipelineState) -> None:
        """Save pipeline state for resumption.
        
        Args:
            state: Current pipeline state.
        """
        state.last_updated = datetime.utcnow()
        
        # Atomic write
        temp_path = self.state_file.with_suffix(".json.tmp")
        with open(temp_path, "w") as f:
            f.write(state.model_dump_json(indent=2))
        temp_path.rename(self.state_file)

    def load_state(self) -> Optional[PipelineState]:
        """Load pipeline state from disk.
        
        Returns:
            PipelineState if file exists, None otherwise.
        """
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            return PipelineState.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return None

    def merge_all_batches(self, output_file: Path) -> int:
        """Combine all batch files into a single Parquet file.
        
        Args:
            output_file: Path for the merged output file.
            
        Returns:
            Total number of rows in merged file.
            
        Raises:
            ValueError: If no batch files exist.
        """
        batch_files = self.get_batch_files()
        if not batch_files:
            raise ValueError("No batch files to merge")
        
        logger.info(f"Merging {len(batch_files)} batch files...")
        
        # Read and concatenate all tables
        tables = []
        for f in batch_files:
            try:
                table = pq.read_table(f)
                tables.append(table)
            except Exception as e:
                logger.error(f"Failed to read {f}: {e}")
                raise
        
        merged = pa.concat_tables(tables)
        
        # Atomic write
        temp_path = output_file.with_suffix(".parquet.tmp")
        pq.write_table(merged, temp_path, compression="snappy")
        temp_path.rename(output_file)
        
        logger.info(f"Merged {len(merged)} samples to {output_file}")
        return len(merged)

    def read_batch(self, batch_num: int) -> list[AnnotationResult]:
        """Read a specific batch file.
        
        Args:
            batch_num: Batch number to read.
            
        Returns:
            List of annotation results.
        """
        filename = f"batch_{batch_num:06d}.parquet"
        filepath = self.batches_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Batch file not found: {filepath}")
        
        table = pq.read_table(filepath)
        results = []
        
        for i in range(len(table)):
            row = {col: table[col][i].as_py() for col in table.column_names}
            results.append(AnnotationResult.model_validate(row))
        
        return results
