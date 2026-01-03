#!/usr/bin/env python3
"""
Phase 3 Alpha: FineWeb Streaming Filter (Stage 1 - CPU Heuristics)

A highly efficient, resumable script to stream the sample-10BT subset of FineWeb,
apply CPU-based heuristic filtering, and save passing documents to Parquet.

Hardware Target: Mac M2 (8GB RAM), 40 Mbps bandwidth
Output: Snappy-compressed Parquet partitions

Usage:
    uv run python scripts/stream_filter_stage1.py
    uv run python scripts/stream_filter_stage1.py --max-docs 10000  # Test run
"""

from __future__ import annotations

import json
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import typer
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

# =============================================================================
# CLI Setup
# =============================================================================
app = typer.Typer(
    name="stream-filter-stage1",
    help="Stream FineWeb sample-10BT and filter for legal documents.",
    add_completion=False,
)
console = Console()

# =============================================================================
# STAGE A: NEGATIVE FILTER - Boilerplate & News Removal
# Pre-compiled as frozenset for O(1) membership testing
# =============================================================================
BOILERPLATE_KEYWORDS: frozenset[str] = frozenset({
    # Privacy & Legal boilerplate
    "privacy policy", "cookie policy", "terms of use", "terms of service",
    "terms and conditions", "all rights reserved", "copyright ©",
    
    # Commercial
    "subscribe to our newsletter", "sign up for our newsletter",
    "shopping cart", "add to cart", "buy now", "free shipping",
    
    # Navigation
    "skip to content", "skip to main", "back to top", "click here to",
    
    # Cookies
    "we use cookies", "this website uses cookies", "accept cookies",
    
    # Social
    "share on facebook", "share on twitter", "follow us on",
    
    # News & Media
    "latest news", "trending stories", "breaking news", "editorial",
    "opinion piece", "op-ed", "advertisement", "sponsored content",
    "leave a comment", "comments section", "related articles",
    "reporter", "correspondent", "journalism", "newsroom",
})

# =============================================================================
# STAGE B: POSITIVE FILTER - Strict Legal Keywords
# =============================================================================
STRICT_LEGAL_KEYWORDS: frozenset[str] = frozenset({
    # Litigation parties
    "plaintiff", "defendant", "appellant", "appellee", "respondent", "petitioner",
    
    # Writs & Procedures
    "writ", "habeas corpus", "certiorari", "injunction", "mandamus",
    
    # Court documents
    "affidavit", "testimony", "deposition", "subpoena", "pleading", "motion to",
    
    # Legal language
    "pursuant to", "hereby ordered", "it is ordered", "court finds", "court holds",
    
    # Judgments
    "decree", "adjudicated", "remanded", "reversed", "affirmed", "vacated",
    "dismissed", "sustained", "overruled",
    
    # Statutes
    "statute", "codified", "legislature", "constitutionality", "unconstitutional",
    
    # Court mechanics
    "docket", "jurisdiction", "venue", "standing",
    
    # Criminal
    "verdict", "acquittal", "conviction", "sentencing", "indictment",
    
    # Citations (lowercase for matching)
    "v.", "vs.", "u.s.c.", "c.f.r.", "f.2d", "f.3d", "s.ct.",
})

# =============================================================================
# STAGE C: NEWS URL FILTER
# =============================================================================
NEWS_URL_PATTERNS: frozenset[str] = frozenset({
    "/news/", "/blog/", "/article/", "/story/", "/opinion/", "/editorial/",
    "nytimes.com", "cnn.com", "theguardian.com", "washingtonpost.com",
    "forbes.com", "bbc.com", "reuters.com", "apnews.com", "nbcnews.com",
    "foxnews.com", "usatoday.com", "huffpost.com", "politico.com",
    "buzzfeed.com", "vice.com", "medium.com",
})

# =============================================================================
# STAGE D: ELITE CITATION REGEX FILTER
# PRE-COMPILED for maximum performance (critical for streaming)
# =============================================================================
CITATION_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"v\.\s+[A-Z]", re.IGNORECASE),       # "Roe v. Wade", "State v. Smith"
    re.compile(r"§\s*\d+"),                           # "§ 1983", "§501"
    re.compile(r"Section\s+\d+", re.IGNORECASE),      # "Section 12"
    re.compile(r"\d+\s+U\.S\.C\.", re.IGNORECASE),    # "42 U.S.C."
    re.compile(r"Article\s+[IVX]+", re.IGNORECASE),   # "Article III"
    re.compile(r"No\.\s+\d+"),                        # "No. 12-345"
    re.compile(r"\bId\.", re.IGNORECASE),             # "Id." - common shorthand
    re.compile(r"Ct\.\s+App\.", re.IGNORECASE),       # "Ct. App."
    re.compile(r"\d+\s+F\.\d+d\s+\d+"),               # "123 F.2d 456"
    re.compile(r"\d+\s+S\.Ct\.\s+\d+"),               # "123 S.Ct. 456"
    re.compile(r"\d+\s+L\.Ed\.\s*\d*"),               # "123 L.Ed. 456"
    re.compile(r"C\.F\.R\.\s*§?\s*\d+"),              # "C.F.R. § 12"
    re.compile(r"Pub\.\s*L\.\s*No\.", re.IGNORECASE), # "Pub. L. No."
    re.compile(r"Stat\.\s+\d+"),                      # "Stat. 123"
)

# Filter thresholds
MIN_STRICT_KEYWORD_MATCHES: int = 2
BOILERPLATE_CHECK_CHARS: int = 1000
POSITIVE_CHECK_CHARS: int = 5000
CITATION_CHECK_CHARS: int = 8000


# =============================================================================
# Filter Functions (Optimized for Speed)
# =============================================================================

def is_news_url(url: Optional[str]) -> bool:
    """Stage C: Check if URL is from a news/media source."""
    if not url:
        return False
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in NEWS_URL_PATTERNS)


def contains_boilerplate(text: str) -> bool:
    """Stage A: Check first/last 1000 chars for boilerplate."""
    first_chunk = text[:BOILERPLATE_CHECK_CHARS].lower()
    if any(kw in first_chunk for kw in BOILERPLATE_KEYWORDS):
        return True
    
    if len(text) > BOILERPLATE_CHECK_CHARS:
        last_chunk = text[-BOILERPLATE_CHECK_CHARS:].lower()
        if any(kw in last_chunk for kw in BOILERPLATE_KEYWORDS):
            return True
    return False


def count_legal_keywords(text: str) -> int:
    """Stage B: Count unique legal keywords in first 5000 chars."""
    check_text = text[:POSITIVE_CHECK_CHARS].lower()
    count = 0
    for kw in STRICT_LEGAL_KEYWORDS:
        if kw in check_text:
            count += 1
            if count >= MIN_STRICT_KEYWORD_MATCHES:
                # Early exit once threshold met
                return count
    return count


def has_legal_citation(text: str) -> bool:
    """Stage D: Check for formal legal citation patterns."""
    check_text = text[:CITATION_CHECK_CHARS]
    # Use any() with generator for short-circuit evaluation
    return any(regex.search(check_text) for regex in CITATION_REGEXES)


def passes_filter(text: str, url: Optional[str], min_len: int, max_len: int) -> bool:
    """
    Four-stage Elite filter for legal content.
    
    Order optimized for speed (fastest rejections first):
    1. Length check (fastest)
    2. Stage C: News URL check
    3. Stage A: Boilerplate check
    4. Stage B: Legal keywords check
    5. Stage D: Citation pattern check (slowest, but most selective)
    
    Returns True only for structurally professional legal documents.
    """
    # Length filter (extremely fast)
    text_len = len(text)
    if text_len < min_len or text_len > max_len:
        return False
    
    # Stage C: Reject news URLs
    if is_news_url(url):
        return False
    
    # Stage A: Reject boilerplate
    if contains_boilerplate(text):
        return False
    
    # Stage B: Require strict legal keywords
    if count_legal_keywords(text) < MIN_STRICT_KEYWORD_MATCHES:
        return False
    
    # Stage D: ELITE - Require formal citation
    if not has_legal_citation(text):
        return False
    
    return True


# =============================================================================
# State Management
# =============================================================================

class ProcessingState:
    """Lightweight state for resumability."""
    
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.total_processed_count: int = 0
        self.total_passed_count: int = 0
        self.batch_index: int = 0
        self.last_updated: Optional[str] = None
    
    def load(self) -> bool:
        """Load state from file. Returns True if state existed."""
        if not self.state_path.exists():
            return False
        
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
            self.total_processed_count = data.get("total_processed_count", 0)
            self.total_passed_count = data.get("total_passed_count", 0)
            self.batch_index = data.get("batch_index", 0)
            self.last_updated = data.get("last_updated")
            return self.total_processed_count > 0
        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[yellow]Warning: Could not load state file: {e}[/yellow]")
            return False
    
    def save(self) -> None:
        """Save state to file."""
        self.last_updated = datetime.now(timezone.utc).isoformat()
        data = {
            "total_processed_count": self.total_processed_count,
            "total_passed_count": self.total_passed_count,
            "batch_index": self.batch_index,
            "last_updated": self.last_updated,
        }
        # Write atomically to prevent corruption
        tmp_path = self.state_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        tmp_path.rename(self.state_path)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "total_processed_count": self.total_processed_count,
            "total_passed_count": self.total_passed_count,
            "batch_index": self.batch_index,
            "last_updated": self.last_updated,
        }


# =============================================================================
# Parquet Writing
# =============================================================================

def create_parquet_schema() -> pa.Schema:
    """Create PyArrow schema for output Parquet files."""
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.large_string()),  # large_string for texts > 2GB total
        pa.field("url", pa.string()),
    ])


def write_parquet_batch(
    output_dir: Path,
    batch_index: int,
    documents: list[dict[str, Any]],
) -> Path:
    """Write a batch of documents to a Parquet partition file."""
    filename = f"part_{batch_index:05d}.parquet"
    filepath = output_dir / filename
    
    schema = create_parquet_schema()
    table = pa.Table.from_pylist(documents, schema=schema)
    
    # Write with Snappy compression
    pq.write_table(table, filepath, compression="snappy")
    
    return filepath


# =============================================================================
# Main Streaming Logic
# =============================================================================

class StreamProcessor:
    """Main processor for streaming and filtering."""
    
    def __init__(
        self,
        output_dir: Path,
        buffer_size: int = 1000,
        min_text_length: int = 500,
        max_text_length: int = 50000,
        max_docs: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.max_docs = max_docs
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State management
        self.state = ProcessingState(output_dir / "processing_state.json")
        
        # Document buffer
        self.buffer: list[dict[str, Any]] = []
        
        # Shutdown handling
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle graceful shutdown."""
        if self._shutdown_requested:
            console.print("\n[red]Force quitting...[/red]")
            sys.exit(1)
        
        self._shutdown_requested = True
        console.print("\n[yellow]Graceful shutdown requested. Flushing buffer...[/yellow]")
    
    def flush_buffer(self) -> None:
        """Flush document buffer to Parquet file."""
        if not self.buffer:
            return
        
        self.state.batch_index += 1
        filepath = write_parquet_batch(
            self.output_dir,
            self.state.batch_index,
            self.buffer,
        )
        console.print(f"[dim]Wrote {len(self.buffer)} docs → {filepath.name}[/dim]")
        
        self.buffer = []
        self.state.save()
    
    def run(self) -> None:
        """Main processing loop."""
        # Load existing state if any
        resumed = self.state.load()
        
        if resumed:
            console.print(Panel(
                f"[bold yellow]Resuming from previous state[/bold yellow]\n\n"
                f"Processed: {self.state.total_processed_count:,}\n"
                f"Passed: {self.state.total_passed_count:,}\n"
                f"Batches: {self.state.batch_index}\n"
                f"Last updated: {self.state.last_updated}",
                title="♻️ Resume Mode",
                border_style="yellow",
            ))
        else:
            console.print(Panel(
                f"[bold green]Starting fresh[/bold green]\n\n"
                f"Output: {self.output_dir}\n"
                f"Buffer size: {self.buffer_size:,} docs\n"
                f"Text length: {self.min_text_length:,} - {self.max_text_length:,} chars",
                title="🚀 New Run",
                border_style="green",
            ))
        
        # Load dataset in streaming mode
        console.print("[dim]Connecting to HuggingFace Hub...[/dim]")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        
        # Skip to resume position if needed
        # NOTE: skip() on a streaming dataset iterates through N items.
        # This can be slow but is necessary to avoid duplicating data.
        if self.state.total_processed_count > 0:
            console.print(
                f"[yellow]Skipping {self.state.total_processed_count:,} docs "
                f"(this may take a while)...[/yellow]"
            )
            dataset = dataset.skip(self.state.total_processed_count)
        
        # Create progress bar
        # Using total=None for infinite stream, or max_docs if specified
        total = self.max_docs
        desc = "Streaming FineWeb"
        
        pbar = tqdm(
            dataset,
            total=total,
            desc=desc,
            unit="doc",
            dynamic_ncols=True,
            smoothing=0.1,  # Smoother rate estimation
        )
        
        local_processed = 0
        local_passed = 0
        
        try:
            for row in pbar:
                if self._shutdown_requested:
                    break
                
                # Check max docs limit
                if self.max_docs is not None and local_processed >= self.max_docs:
                    break
                
                text = row.get("text", "")
                url = row.get("url")
                doc_id = row.get("id", "")
                
                local_processed += 1
                self.state.total_processed_count += 1
                
                # Apply filter
                if passes_filter(text, url, self.min_text_length, self.max_text_length):
                    local_passed += 1
                    self.state.total_passed_count += 1
                    
                    # Add to buffer
                    self.buffer.append({
                        "id": doc_id,
                        "text": text,
                        "url": url or "",
                    })
                    
                    # Flush buffer when full
                    if len(self.buffer) >= self.buffer_size:
                        self.flush_buffer()
                
                # Update progress bar postfix
                if local_processed % 100 == 0:
                    pass_rate = (local_passed / local_processed * 100) if local_processed > 0 else 0
                    pbar.set_postfix({
                        "pass_rate": f"{pass_rate:.2f}%",
                        "passed": f"{self.state.total_passed_count:,}",
                        "batches": self.state.batch_index,
                    })
        
        finally:
            # Always flush remaining buffer and save state
            pbar.close()
            self.flush_buffer()
            self.state.save()
            
            # Final summary
            pass_rate = (
                self.state.total_passed_count / self.state.total_processed_count * 100
                if self.state.total_processed_count > 0 else 0
            )
            
            console.print(Panel(
                f"[bold]Processed:[/bold] {self.state.total_processed_count:,}\n"
                f"[bold]Passed:[/bold] {self.state.total_passed_count:,}\n"
                f"[bold]Pass Rate:[/bold] {pass_rate:.2f}%\n"
                f"[bold]Batches:[/bold] {self.state.batch_index}\n"
                f"[bold]Output:[/bold] {self.output_dir}",
                title="📊 Final Summary",
                border_style="blue",
            ))


# =============================================================================
# CLI Entry Point
# =============================================================================

@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("data/stage1"),
        "--output-dir", "-o",
        help="Directory to save Parquet partitions and state file",
    ),
    buffer_size: int = typer.Option(
        1000,
        "--buffer-size", "-b",
        help="Number of passing documents to buffer before writing to Parquet",
    ),
    min_text_length: int = typer.Option(
        500,
        "--min-text-length",
        help="Minimum text length (chars) to include",
    ),
    max_text_length: int = typer.Option(
        50000,
        "--max-text-length",
        help="Maximum text length (chars) to include",
    ),
    max_docs: Optional[int] = typer.Option(
        None,
        "--max-docs", "-n",
        help="Maximum documents to process (for testing). Default: unlimited",
    ),
) -> None:
    """
    Stream FineWeb sample-10BT and filter for legal documents.
    
    This script implements a 4-stage heuristic filter:
    
    1. News URL rejection
    2. Boilerplate rejection  
    3. Legal keyword requirement (≥2 keywords)
    4. Legal citation requirement (formal citations)
    
    Documents passing all stages are saved to Parquet partitions.
    Progress is automatically saved and can be resumed after interruption.
    """
    processor = StreamProcessor(
        output_dir=output_dir,
        buffer_size=buffer_size,
        min_text_length=min_text_length,
        max_text_length=max_text_length,
        max_docs=max_docs,
    )
    processor.run()


if __name__ == "__main__":
    app()
