"""Thin wrapper around the OpenAI chat-completions API for market analysis."""

from __future__ import annotations

import json
import sys
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from ..config import OPENAI_MAX_TOKENS, OPENAI_MODEL, OPENAI_TEMPERATURE
    from ..reports.prompt_generator import PromptGenerator
except ImportError:
    from config import OPENAI_MAX_TOKENS, OPENAI_MODEL, OPENAI_TEMPERATURE
    from reports.prompt_generator import PromptGenerator


class AIAdvisor:
    """Call OpenAI-compatible chat-completions API for market analysis.

    Supports OpenAI, DeepSeek, and any OpenAI-compatible endpoint via base_url.
    """

    DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str = OPENAI_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        max_tokens: int = OPENAI_MAX_TOKENS,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    def analyze(self, prompt: str) -> str:
        """Send *prompt* (the full user-prompt text) to the model and return
        the assistant's reply as a string.

        Tries the ``openai`` library first; falls back to raw urllib.
        """
        result = self._try_openai_library(prompt)
        if result is not None:
            return result
        return self._call_via_urllib(prompt)

    def _try_openai_library(self, prompt: str) -> Optional[str]:
        try:
            import openai  # noqa: F811
        except ImportError:
            return None

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = openai.OpenAI(**kwargs)
        system_msg = (
            PromptGenerator.ANALYSIS_SYSTEM_PROMPT
            + "\n\n"
            + PromptGenerator.OUTPUT_FORMAT_TEMPLATE
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _call_via_urllib(self, prompt: str) -> str:
        system_msg = (
            PromptGenerator.ANALYSIS_SYSTEM_PROMPT
            + "\n\n"
            + PromptGenerator.OUTPUT_FORMAT_TEMPLATE
        )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")

        api_url = (self.base_url.rstrip("/") + "/chat/completions") if self.base_url else self.DEFAULT_API_URL
        request = Request(
            api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        max_attempts = 2
        delay = 2.0
        import time

        for attempt in range(1, max_attempts + 1):
            try:
                with urlopen(request, timeout=120) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    return payload["choices"][0]["message"]["content"]
            except HTTPError as exc:
                try:
                    err_body = exc.read().decode("utf-8")
                except Exception:
                    err_body = str(exc.reason)
                raise RuntimeError(
                    f"OpenAI API HTTP {exc.code}: {err_body}"
                ) from exc
            except URLError as exc:
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"OpenAI API connection error: {exc.reason}"
                    ) from exc
                time.sleep(delay)
                delay *= 2

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: ~1 token per 3.5 Chinese chars or 4 English chars."""
        cn_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        en_chars = len(text) - cn_chars
        return int(cn_chars / 3.5 + en_chars / 4)
