"""Mistral API client for legal value annotation - Optimized with tenacity."""

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Any, Callable, Optional

from mistralai import Mistral
from mistralai.models import HTTPValidationError, SDKError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from fineweb_legal.config import AnnotationConfig
from fineweb_legal.models import AnnotationResult, Document

logger = logging.getLogger(__name__)


class AnnotationError(Exception):
    """Base exception for annotation errors."""
    pass


class AnnotationParseError(AnnotationError):
    """Failed to parse LLM response."""
    pass


class AnnotationAPIError(AnnotationError):
    """API call failed after retries."""
    pass


class RateLimitError(AnnotationAPIError):
    """Rate limit hit - triggers longer backoff."""
    pass


# System prompt for JSON mode annotation
SYSTEM_PROMPT = """You are an expert legal scholar and data quality judge. Your task is to score the following web text on a scale from 0 to 5 based on its "Legal Reasoning and Value."

Scoring Guidelines:
- **Score 0 (Noise/Spam):** Navigation menus, login pages, unrelated ads, adult content, or gibberish.
- **Score 1 (General/Marketing):** Law firm landing pages ("Call us for a free consultation"), generic news summaries, "legal zoom" style basic templates.
- **Score 2 (Basic Info):** Simple definitions of laws, Wikipedia-style legal summaries, non-expert forum questions (e.g., Reddit r/legaladvice questions).
- **Score 3 (Useful):** Detailed legal news analysis, government handouts explaining rights, paralegal guides, clear textbook definitions.
- **Score 4 (High Value):** Full case text, statutes, contracts, law review abstracts, detailed verified legal analysis.
- **Score 5 (Gold Standard):** Complex appellate reasoning, Supreme Court opinions, dense statutory interpretation, academic law journals, official legislative history.

You MUST respond with a JSON object in this exact format:
{"score": <integer 0-5>, "reasoning": "<brief 1-sentence explanation>"}

Respond ONLY with the JSON object, no additional text."""


class LegalAnnotator:
    """Mistral-Medium based legal value annotator with tenacity-powered retries.
    
    Features:
    - Exponential backoff via tenacity (no manual sleep)
    - Up to 20 retry attempts with automatic speed-finding
    - Graceful degradation on persistent failures
    - Detailed logging of retry attempts
    """

    def __init__(self, config: AnnotationConfig):
        """Initialize the annotator."""
        self.config = config
        
        if config.mistral_api_key is None:
            raise ValueError("Mistral API key is required for annotation")
        
        self.client = Mistral(api_key=config.mistral_api_key.get_secret_value())
        self.model = config.mistral_model
        self.truncation_limit = config.inference_truncation_chars
        self._request_count: int = 0
        self._error_count: int = 0
        self._retry_count: int = 0

    def _truncate_for_inference(self, text: str) -> str:
        """Truncate text for API call while preserving word boundaries."""
        if len(text) <= self.truncation_limit:
            return text
        truncated = text[: self.truncation_limit].rsplit(" ", 1)[0]
        return truncated + "..."

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token)."""
        return max(1, len(text) // 4)

    def _parse_json_response(self, content: str) -> tuple[int, str]:
        """Safely parse JSON response with validation."""
        try:
            content = content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            
            data = json.loads(content)
            score = int(data["score"])
            
            if not 0 <= score <= 5:
                raise ValueError(f"Score {score} out of range [0-5]")
            
            reasoning = str(data.get("reasoning", ""))
            return score, reasoning
            
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise AnnotationParseError(f"Failed to parse: {content[:100]}...") from e

    async def _make_api_call(self, truncated_text: str) -> str:
        """Make a single API call with timeout (no retries - handled by tenacity wrapper)."""
        import asyncio
        
        try:
            # 30 second timeout to prevent indefinite hangs
            async with asyncio.timeout(30):
                response = await self.client.chat.complete_async(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": truncated_text},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=100,
                )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content
            
            raise AnnotationAPIError("Empty response from API")
        
        except asyncio.TimeoutError:
            self._error_count += 1
            raise AnnotationAPIError("Request timed out after 30s")
            
        except SDKError as e:
            error_str = str(e)
            self._error_count += 1
            
            # Convert 429 to RateLimitError for special handling
            if "429" in error_str or "rate" in error_str.lower():
                raise RateLimitError(f"Rate limit: {error_str[:100]}") from e
            
            raise AnnotationAPIError(f"API error: {error_str[:100]}") from e
            
        except HTTPValidationError as e:
            self._error_count += 1
            raise AnnotationAPIError(f"Validation error: {e}") from e

    async def _call_api_with_retry(self, truncated_text: str) -> str:
        """Make API call with tenacity-powered exponential backoff.
        
        Configuration:
        - wait_exponential: min=0.5s, max=60s, multiplier=1 
        - stop: after 20 attempts
        - retry: on AnnotationAPIError and RateLimitError
        - logging: before each retry sleep
        """
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(20),
                wait=wait_exponential(multiplier=1, min=0.5, max=60),
                retry=retry_if_exception_type((AnnotationAPIError, RateLimitError)),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    self._retry_count += attempt.retry_state.attempt_number - 1
                    result = await self._make_api_call(truncated_text)
                    self._request_count += 1
                    return result
                    
        except RetryError as e:
            # All retries exhausted - propagate the last exception
            logger.error(f"All 20 retry attempts exhausted: {e}")
            raise AnnotationAPIError(f"Failed after 20 attempts: {e.last_attempt.exception()}") from e
        
        raise AnnotationAPIError("Unexpected: retry loop exited without result")

    async def annotate(self, document: Document) -> AnnotationResult:
        """Annotate a single document for legal value.
        
        Uses tenacity for automatic retry with exponential backoff.
        No manual sleep - tenacity finds the optimal speed automatically.
        """
        truncated_text = self._truncate_for_inference(document.text)
        input_tokens = self._estimate_tokens(truncated_text)
        
        content = await self._call_api_with_retry(truncated_text)
        score, reasoning = self._parse_json_response(content)
        
        return AnnotationResult(
            id=document.id,
            score=score,
            reasoning=reasoning,
            text=document.text,
            url=document.url,
            timestamp=datetime.utcnow(),
            model=self.model,
            input_tokens=input_tokens,
            text_length=len(document.text),
        )

    async def annotate_batch(
        self,
        documents: list[Document],
        on_progress: Optional[Callable[[bool], None]] = None,
    ) -> tuple[list[AnnotationResult], list[tuple[str, str]]]:
        """Annotate a batch of documents SEQUENTIALLY.
        
        Sequential processing ensures we don't overwhelm the API.
        Tenacity handles individual request retries with backoff.
        
        Returns:
            Tuple of (successful results, list of (doc_id, error_message)).
        """
        results: list[AnnotationResult] = []
        errors: list[tuple[str, str]] = []
        
        for i, doc in enumerate(documents):
            try:
                result = await self.annotate(doc)
                results.append(result)
                
                if on_progress:
                    on_progress(success=True)
                    
            except AnnotationError as e:
                # Individual document failed after all retries
                # Log but don't crash - allow batch to continue
                error_msg = str(e)
                logger.warning(
                    f"Failed to annotate doc {i+1}/{len(documents)} "
                    f"({doc.id[:20]}...): {error_msg[:100]}"
                )
                errors.append((doc.id, error_msg))
                
                if on_progress:
                    on_progress(success=False)
        
        return results, errors

    @property
    def request_count(self) -> int:
        """Return the total number of successful API requests."""
        return self._request_count
    
    @property
    def error_count(self) -> int:
        """Return the total number of errors encountered."""
        return self._error_count
    
    @property
    def retry_count(self) -> int:
        """Return the total number of retry attempts."""
        return self._retry_count
