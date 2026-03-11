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
    """Call OpenAI chat-completions and return a trading analysis.

    Uses urllib so we don't add a hard dependency on the ``openai`` package.
    If the ``openai`` library *is* installed it will be preferred for its
    richer error handling and streaming support.
    """

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str = OPENAI_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        max_tokens: int = OPENAI_MAX_TOKENS,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

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

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PromptGenerator.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _call_via_urllib(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": PromptGenerator.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")

        request = Request(
            self.API_URL,
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
