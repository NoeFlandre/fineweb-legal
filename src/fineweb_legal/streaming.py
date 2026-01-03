"""HuggingFace dataset streaming with Elite Citation & Structure filtering."""

import logging
import re
from collections.abc import Iterator
from typing import Any, Optional

from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset

from fineweb_legal.config import AnnotationConfig
from fineweb_legal.models import Document

logger = logging.getLogger(__name__)


# =============================================================================
# STAGE A: NEGATIVE FILTER - Boilerplate & News Removal
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
    
    # Citations
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
# Documents MUST contain at least one formal legal citation pattern
# =============================================================================
CITATION_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"v\.\s+[A-Z]", re.IGNORECASE),      # "Roe v. Wade", "State v. Smith"
    re.compile(r"§\s*\d+"),                          # "§ 1983", "§501"
    re.compile(r"Section\s+\d+", re.IGNORECASE),     # "Section 12"
    re.compile(r"\d+\s+U\.S\.C\.", re.IGNORECASE),   # "42 U.S.C."
    re.compile(r"Article\s+[IVX]+", re.IGNORECASE),  # "Article III"
    re.compile(r"No\.\s+\d+"),                       # "No. 12-345"
    re.compile(r"\bId\.", re.IGNORECASE),            # "Id." - common shorthand
    re.compile(r"Ct\.\s+App\.", re.IGNORECASE),      # "Ct. App."
    re.compile(r"\d+\s+F\.\d+d\s+\d+"),              # "123 F.2d 456"
    re.compile(r"\d+\s+S\.Ct\.\s+\d+"),              # "123 S.Ct. 456"
    re.compile(r"\d+\s+L\.Ed\.\s*\d*"),              # "123 L.Ed. 456"
    re.compile(r"C\.F\.R\.\s*§?\s*\d+"),             # "C.F.R. § 12"
    re.compile(r"Pub\.\s*L\.\s*No\.", re.IGNORECASE),# "Pub. L. No."
    re.compile(r"Stat\.\s+\d+"),                     # "Stat. 123"
]

MIN_STRICT_KEYWORD_MATCHES: int = 2
BOILERPLATE_CHECK_CHARS: int = 1000
POSITIVE_CHECK_CHARS: int = 5000
CITATION_CHECK_CHARS: int = 8000  # Check more text for citations


def _is_news_url(url: Optional[str]) -> bool:
    """Check if URL is from a news/media source."""
    if not url:
        return False
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in NEWS_URL_PATTERNS)


def _contains_boilerplate(text: str) -> bool:
    """Check if text contains boilerplate in first/last 1000 chars."""
    first_chunk = text[:BOILERPLATE_CHECK_CHARS].lower()
    if any(kw in first_chunk for kw in BOILERPLATE_KEYWORDS):
        return True
    
    if len(text) > BOILERPLATE_CHECK_CHARS:
        last_chunk = text[-BOILERPLATE_CHECK_CHARS:].lower()
        if any(kw in last_chunk for kw in BOILERPLATE_KEYWORDS):
            return True
    return False


def _count_strict_legal_keywords(text: str) -> int:
    """Count unique strict legal keywords in first 5000 chars."""
    check_text = text[:POSITIVE_CHECK_CHARS].lower()
    matches = sum(1 for kw in STRICT_LEGAL_KEYWORDS if kw in check_text)
    return min(matches, MIN_STRICT_KEYWORD_MATCHES)  # Cap for efficiency


def _has_legal_citation(text: str) -> bool:
    """Check if text contains at least one formal legal citation.
    
    This is the ELITE filter - ensures documents have proper legal structure.
    """
    check_text = text[:CITATION_CHECK_CHARS]
    return any(regex.search(check_text) for regex in CITATION_REGEXES)


def is_likely_legal(text: str, url: Optional[str] = None) -> bool:
    """Four-stage Elite filter for legal content.
    
    Stage A: Reject boilerplate/news keywords
    Stage B: Require ≥2 strict legal keywords  
    Stage C: Reject news URLs
    Stage D: REQUIRE at least one legal citation pattern
    
    Returns True only for structurally professional legal documents.
    """
    # Stage C: Reject news URLs (fastest check first)
    if _is_news_url(url):
        return False
    
    # Stage A: Reject boilerplate
    if _contains_boilerplate(text):
        return False
    
    # Stage B: Require strict legal keywords
    if _count_strict_legal_keywords(text) < MIN_STRICT_KEYWORD_MATCHES:
        return False
    
    # Stage D: ELITE - Require formal citation
    if not _has_legal_citation(text):
        return False
    
    return True


class FineWebStreamer:
    """Streams documents with Elite Citation & Structure filtering.
    
    Four-stage filtering for maximum quality:
    1. Stage C: Reject news URLs
    2. Stage A: Reject boilerplate  
    3. Stage B: Require litigation keywords
    4. Stage D: REQUIRE legal citations (Elite gate)
    """

    def __init__(self, config: AnnotationConfig):
        self.config = config
        self._dataset: Optional[IterableDataset] = None
        self._iterator: Optional[Iterator[dict[str, Any]]] = None
        self._filtered_count: int = 0
        self._passed_count: int = 0
        self._news_rejected: int = 0
        self._boilerplate_rejected: int = 0
        self._keyword_rejected: int = 0
        self._citation_rejected: int = 0

    def _init_dataset(self) -> None:
        if self._dataset is not None:
            return

        logger.info(
            f"Connecting to {self.config.dataset_name} "
            f"(subset: {self.config.dataset_subset}, split: {self.config.dataset_split})"
        )
        
        mode = "ELITE (Citation Required)" if self.config.keyword_filter_enabled else "OFF"
        logger.info(f"Filtering mode: {mode}")

        self._dataset = load_dataset(
            self.config.dataset_name,
            name=self.config.dataset_subset,
            split=self.config.dataset_split,
            streaming=True,
        )
        self._iterator = iter(self._dataset)
        logger.info("Dataset connection established")

    def _row_to_document(self, row: dict[str, Any]) -> Optional[Document]:
        text = row.get("text", "")
        url = row.get("url")
        
        # Length filter
        if len(text) < self.config.min_text_length:
            return None
        if len(text) > self.config.max_text_length:
            return None
        
        # Elite four-stage filter
        if self.config.keyword_filter_enabled:
            # Stage C: News URL
            if _is_news_url(url):
                self._news_rejected += 1
                self._filtered_count += 1
                return None
            
            # Stage A: Boilerplate
            if _contains_boilerplate(text):
                self._boilerplate_rejected += 1
                self._filtered_count += 1
                return None
            
            # Stage B: Legal keywords
            if _count_strict_legal_keywords(text) < MIN_STRICT_KEYWORD_MATCHES:
                self._keyword_rejected += 1
                self._filtered_count += 1
                return None
            
            # Stage D: ELITE Citation check
            if not _has_legal_citation(text):
                self._citation_rejected += 1
                self._filtered_count += 1
                return None
        
        self._passed_count += 1
        return Document(
            id=row.get("id", ""),
            text=text,
            url=url,
        )

    def stream_documents(self, skip: int = 0) -> Iterator[Document]:
        self._init_dataset()
        assert self._iterator is not None

        skipped = 0
        for row in self._iterator:
            if skipped < skip:
                skipped += 1
                continue

            doc = self._row_to_document(row)
            if doc is not None:
                yield doc

    def stream_batches(
        self,
        batch_size: int,
        skip_batches: int = 0,
        max_batches: Optional[int] = None,
    ) -> Iterator[list[Document]]:
        self._init_dataset()
        assert self._iterator is not None

        current_batch: list[Document] = []
        batch_count = 0
        skipped_docs = skip_batches * batch_size

        for doc in self.stream_documents(skip=skipped_docs):
            current_batch.append(doc)

            if len(current_batch) >= batch_size:
                batch_count += 1
                yield current_batch
                current_batch = []

                if max_batches is not None and batch_count >= max_batches:
                    break

        if current_batch:
            yield current_batch

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
    
    @property
    def filter_stats(self) -> dict[str, int]:
        return {
            "passed": self._passed_count,
            "filtered_total": self._filtered_count,
            "news_rejected": self._news_rejected,
            "boilerplate_rejected": self._boilerplate_rejected,
            "keyword_rejected": self._keyword_rejected,
            "citation_rejected": self._citation_rejected,
        }
