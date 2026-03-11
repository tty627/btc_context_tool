"""Send trading analysis to Telegram via Bot API (no extra dependencies)."""

from __future__ import annotations

import json
import logging
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger("btc_context.telegram")

MAX_MESSAGE_LENGTH = 4096


class TelegramNotifier:
    API_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send(self, text: str, parse_mode: str = "Markdown") -> bool:
        chunks = self._split_message(text)
        ok = True
        for chunk in chunks:
            if not self._send_chunk(chunk, parse_mode):
                if parse_mode == "Markdown":
                    ok = self._send_chunk(chunk, parse_mode="")
                else:
                    ok = False
        return ok

    def _send_chunk(self, text: str, parse_mode: str = "Markdown") -> bool:
        url = self.API_URL.format(token=self.bot_token, method="sendMessage")
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        data = json.dumps(payload).encode("utf-8")
        request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

        try:
            with urlopen(request, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("ok"):
                    return True
                logger.warning("Telegram API returned ok=false: %s", result)
                return False
        except HTTPError as exc:
            logger.error("Telegram API HTTP %d: %s", exc.code, exc.reason)
            return False
        except URLError as exc:
            logger.error("Telegram API connection error: %s", exc.reason)
            return False
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    @staticmethod
    def _split_message(text: str) -> list[str]:
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]

        chunks: list[str] = []
        while text:
            if len(text) <= MAX_MESSAGE_LENGTH:
                chunks.append(text)
                break
            split_at = text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
            if split_at <= 0:
                split_at = MAX_MESSAGE_LENGTH
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return chunks

    def send_analysis_summary(
        self,
        signal_score: dict,
        analysis_text: Optional[str] = None,
        report_text: Optional[str] = None,
    ) -> bool:
        """Send a compact trading summary to Telegram."""
        score = signal_score.get("composite_score", "?")
        bias = signal_score.get("bias", "unknown")
        strength = signal_score.get("strength", "unknown")

        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(bias, "⚪")

        lines = [
            f"{emoji} *BTC Signal: {score}/100 ({strength})*",
            "",
        ]

        components = signal_score.get("components", {})
        for name, comp in components.items():
            bar = self._score_bar(comp.get("score", 50))
            lines.append(f"  {name}: {bar} {comp.get('score')}")

        if analysis_text:
            lines.append("")
            truncated = analysis_text[:3000] if len(analysis_text) > 3000 else analysis_text
            lines.append(truncated)
        elif report_text:
            lines.append("")
            lines.append("_(Full report generated, use --auto-analyze for AI plan)_")

        return self.send("\n".join(lines))

    @staticmethod
    def _score_bar(score: float) -> str:
        filled = int(score / 10)
        return "█" * filled + "░" * (10 - filled)
