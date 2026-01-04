"""Main annotation pipeline orchestrator."""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from fineweb_legal.annotator import AnnotationError, LegalAnnotator
from fineweb_legal.config import AnnotationConfig
from fineweb_legal.models import PipelineState
from fineweb_legal.storage import AnnotationStorage
from fineweb_legal.streaming import FineWebStreamer

logger = logging.getLogger(__name__)
console = Console()


class GracefulExit(Exception):
    """Raised when graceful shutdown is requested."""
    pass


class AnnotationPipeline:
    """Orchestrates the full annotation pipeline."""

    def __init__(self, config: AnnotationConfig, dry_run: bool = False):
        """Initialize the pipeline.
        
        Args:
            config: Pipeline configuration.
            dry_run: If True, skip annotator initialization.
        """
        self.config = config
        self.storage = AnnotationStorage(config.output_dir)
        self.streamer = FineWebStreamer(config)
        self._annotator: Optional[LegalAnnotator] = None
        self._dry_run = dry_run
        self.state: Optional[PipelineState] = None
        self._shutdown_requested = False
        
        # Only initialize annotator for live runs
        if not dry_run:
            self._annotator = LegalAnnotator(config)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    @property
    def annotator(self) -> LegalAnnotator:
        """Get annotator, raising if not initialized."""
        if self._annotator is None:
            raise RuntimeError("Annotator not available in dry-run mode")
        return self._annotator

    def _signal_handler(self, signum: int, frame: object) -> None:
        """Handle shutdown signals gracefully."""
        if self._shutdown_requested:
            console.print("\n[red]Force quitting...[/red]")
            sys.exit(1)
        
        self._shutdown_requested = True
        console.print("\n[yellow]Graceful shutdown requested. Finishing current batch...[/yellow]")

    def _load_or_create_state(self) -> PipelineState:
        """Load existing state or create new one."""
        state = self.storage.load_state()
        if state is None:
            state = PipelineState(started_at=datetime.utcnow())
        return state

    def _create_progress(self) -> Progress:
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

    async def run(
        self,
        target_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        dry_run: bool = False,
    ) -> PipelineState:
        """Run the annotation pipeline.
        
        Args:
            target_samples: Override target sample count.
            batch_size: Override batch size.
            dry_run: If True, simulate without API calls.
            
        Returns:
            Final pipeline state.
        """
        target = target_samples or self.config.target_samples
        batch_sz = batch_size or self.config.batch_size
        
        # Load state and determine starting point
        self.state = self._load_or_create_state()
        start_batch = self.storage.get_next_batch_number()
        total_annotated = self.storage.get_total_annotated()
        
        console.print(Panel(
            f"[bold]FineWeb-Legal Annotation Pipeline[/bold]\n\n"
            f"Target: {target:,} samples\n"
            f"Batch size: {batch_sz}\n"
            f"Starting from batch: {start_batch}\n"
            f"Already annotated: {total_annotated:,}\n"
            f"Mode: {'DRY RUN' if dry_run else 'LIVE'}",
            title="🚀 Pipeline Starting",
            border_style="green" if not dry_run else "yellow",
        ))
        
        if total_annotated >= target:
            console.print("[green]✓ Target already reached![/green]")
            return self.state
        
        remaining = target - total_annotated
        batches_needed = (remaining + batch_sz - 1) // batch_sz
        
        with self._create_progress() as progress:
            overall_task = progress.add_task(
                "Overall Progress",
                total=target,
                completed=total_annotated,
            )
            batch_task = progress.add_task(
                "Current Batch",
                total=batch_sz,
                completed=0,
            )
            
            current_batch = start_batch
            
            for batch_docs in self.streamer.stream_batches(
                batch_size=batch_sz,
                skip_batches=start_batch - 1,
                max_batches=batches_needed,
            ):
                if self._shutdown_requested:
                    console.print("\n[yellow]Shutdown requested, saving state...[/yellow]")
                    break
                
                # Check if we've reached target
                current_total = self.storage.get_total_annotated()
                if current_total >= target:
                    break
                
                # Filter out duplicates before annotation
                non_dup_docs = []
                dup_count = 0
                for doc in batch_docs:
                    if self.storage.is_duplicate(doc.text):
                        dup_count += 1
                    else:
                        non_dup_docs.append(doc)
                
                if dup_count > 0:
                    console.print(f"[dim]Batch {current_batch}: Skipped {dup_count} duplicates[/dim]")
                
                if not non_dup_docs:
                    # All duplicates, skip this batch
                    current_batch += 1
                    continue
                
                progress.reset(batch_task, total=len(non_dup_docs))
                
                if dry_run:
                    # Simulate annotation
                    await asyncio.sleep(0.1)
                    progress.update(batch_task, completed=len(non_dup_docs))
                    progress.update(overall_task, advance=len(non_dup_docs))
                    console.print(f"[dim]Dry run: would annotate {len(non_dup_docs)} docs[/dim]")
                else:
                    # Real annotation
                    def on_progress(success: bool) -> None:
                        progress.advance(batch_task)
                        if success:
                            progress.advance(overall_task)
                    
                    results, errors = await self.annotator.annotate_batch(
                        non_dup_docs,
                        on_progress=on_progress,
                    )
                    
                    if results:
                        # Write batch to storage
                        self.storage.write_batch(current_batch, results)
                        
                        # Update state
                        self.state.total_annotated += len(results)
                        self.state.total_batches = current_batch
                        self.state.last_batch_number = current_batch
                        self.state.total_errors += len(errors)
                        
                        for r in results:
                            self.state.score_distribution[r.score] += 1
                            self.state.total_tokens_used += r.input_tokens
                        
                        self.storage.save_state(self.state)
                        
                        # Update hash registry with new texts
                        for r in results:
                            self.storage.add_hash(r.text)
                        
                        # Save hash registry periodically (every batch)
                        self.storage.save_hash_registry()
                        
                        if errors:
                            console.print(
                                f"[yellow]Batch {current_batch}: {len(results)} success, "
                                f"{len(errors)} errors[/yellow]"
                            )
                
                current_batch += 1
        
        # Final summary
        final_total = self.storage.get_total_annotated()
        self._display_summary(final_total, target)
        
        return self.state

    def _display_summary(self, total: int, target: int) -> None:
        """Display final summary."""
        distribution = self.storage.get_score_distribution()
        
        table = Table(title="Score Distribution")
        table.add_column("Score", justify="center")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        
        for score in range(6):
            count = distribution[score]
            pct = (count / total * 100) if total > 0 else 0
            table.add_row(str(score), f"{count:,}", f"{pct:.1f}%")
        
        console.print()
        console.print(table)
        
        status = "✓ Complete" if total >= target else "⏸ Paused"
        status_color = "green" if total >= target else "yellow"
        
        api_requests = self._annotator.request_count if self._annotator else 0
        console.print(Panel(
            f"[bold]Total Annotated:[/bold] {total:,} / {target:,}\n"
            f"[bold]API Requests:[/bold] {api_requests:,}\n"
            f"[bold]Status:[/bold] [{status_color}]{status}[/{status_color}]",
            title="📊 Pipeline Summary",
            border_style=status_color,
        ))


async def run_pipeline(
    config: AnnotationConfig,
    target_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    dry_run: bool = False,
) -> PipelineState:
    """Run the annotation pipeline.
    
    Args:
        config: Pipeline configuration.
        target_samples: Override target sample count.
        batch_size: Override batch size.
        dry_run: If True, simulate without API calls.
        
    Returns:
        Final pipeline state.
    """
    pipeline = AnnotationPipeline(config, dry_run=dry_run)
    return await pipeline.run(target_samples=target_samples, batch_size=batch_size, dry_run=dry_run)
