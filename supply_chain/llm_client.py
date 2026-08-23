"""
LLM Client — Vertex AI / Google GenAI wrapper.

Uses the `google-genai` SDK with Application Default Credentials.
Configured via .env:
  GOOGLE_CLOUD_PROJECT    e.g. btp-2-493115
  GOOGLE_CLOUD_LOCATION   e.g. us-central1 (defaults to us-central1)
  GOOGLE_GENAI_USE_VERTEXAI=true
  MODEL_ID                e.g. gemini-2.5-flash-preview-04-17
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get_client():
    """Lazy-import and construct the GenAI client (avoids import-time errors)."""
    from google import genai
    return genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


class LLMClient:
    """
    Thin wrapper around the Google GenAI SDK for Vertex AI.

    Usage
    -----
    client = LLMClient()
    text   = client.call("What is 2+2?")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 200,   # paper uses 90 but we allow a little extra margin
    ) -> None:
        self.model       = model or os.environ.get("MODEL_ID", "gemini-2.5-flash-preview-04-17")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._client     = None          # lazy-initialised

    @property
    def client(self):
        if self._client is None:
            self._client = _get_client()
        return self._client

    def call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """
        Send a prompt and return the model's text response.

        Retries up to `retries` times on transient errors (rate-limits, 5xx).
        Raises on persistent failure.
        """
        from google.genai import types

        temp   = temperature if temperature is not None else self.temperature
        tokens = max_tokens  if max_tokens  is not None else self.max_tokens

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=tokens,
        )

        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                # response.text can be None if the model produced no text parts
                text = response.text
                if text is None:
                    # Extract from candidates directly
                    try:
                        text = response.candidates[0].content.parts[0].text or ""
                    except (IndexError, AttributeError):
                        text = ""
                return text
            except Exception as exc:
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
        raise RuntimeError(
            f"LLM call failed after {retries} attempts: {last_exc}"
        ) from last_exc


def parse_order(response: str, max_order: int = 100, fallback: int = 0) -> int:
    """
    Extract an integer order quantity from the LLM's text response.

    Strategy (in priority order):
      1. A line that contains ONLY a number (clean single-integer response).
      2. The last integer found anywhere in the response.
      3. Return `fallback` if no integer is found.

    Always clips the result to [0, max_order].
    """
    text = response.strip()

    # Priority 1: line that is just a number (possibly with whitespace)
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"\d+", line):
            return int(min(max(int(line), 0), max_order))

    # Priority 2: last integer token in the full text
    matches = re.findall(r"\b\d+\b", text)
    if matches:
        return int(min(max(int(matches[-1]), 0), max_order))

    return fallback
