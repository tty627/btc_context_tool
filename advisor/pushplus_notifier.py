"""Send trading notifications via PushPlus (微信推送)."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Optional

logger = logging.getLogger("btc_context.pushplus")


class PushPlusNotifier:
    API_URL = "https://www.pushplus.plus/send"

    def __init__(self, token: str) -> None:
        self.token = token

    def send(self, title: str, content: str, template: str = "txt") -> bool:
        payload = {
            "token": self.token,
            "title": title,
            "content": content,
            "template": template,
        }
        try:
            result = subprocess.run(
                ["curl", "-s", "-m", "20", self.API_URL,
                 "-X", "POST",
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps(payload, ensure_ascii=False)],
                capture_output=True, text=True, timeout=25,
            )
            if result.returncode != 0:
                logger.error("PushPlus curl failed: %s", result.stderr)
                return False
            resp = json.loads(result.stdout)
            if resp.get("code") == 200:
                logger.info("PushPlus sent OK: %s", title)
                return True
            logger.warning("PushPlus returned code=%s: %s", resp.get("code"), resp.get("msg"))
            return False
        except Exception as exc:
            logger.error("PushPlus send failed: %s", exc)
            return False

    def send_trade_signal(self, analysis_text: str, context: dict) -> bool:
        """Parse AI analysis and send a concise trade notification."""
        price = context.get("price", 0)
        symbol = context.get("symbol", "BTCUSDT")

        is_wait = _is_wait_decision(analysis_text)
        if is_wait:
            return False

        title = f"📊 {symbol} 交易信号 @{price:.0f}"
        body = _extract_signal_summary(analysis_text, max_len=800)
        return self.send(title, body)

    def send_status(self, message: str) -> bool:
        return self.send("BTC Monitor", message)


def _is_wait_decision(text: str) -> bool:
    lower = text.lower()
    for marker in ("execution_mode: wait", "execution_mode:wait",
                    "主结论: 等待", "主结论: 不交易", "主结论:等待", "主结论:不交易",
                    "position_action: hold_and_wait", "position_action: hold"):
        if marker in lower or marker in text:
            return True
    if "wait_plan:" in text and "immediate_entry_plan:" not in text and "pullback_plan:" not in text and "trigger_plan:" not in text:
        return True
    return False


def _extract_signal_summary(text: str, max_len: int = 800) -> str:
    lines = text.split("\n")
    keep_sections = []
    current_section: list[str] = []
    important = False

    important_headers = (
        "【当前动作】", "【执行模式】", "【执行方案】",
        "【持仓主结论】", "【持仓处理】",
        "【一句人话总结】",
        "immediate_entry_plan:", "pullback_plan:", "trigger_plan:",
    )

    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(h) for h in important_headers):
            if current_section and important:
                keep_sections.extend(current_section)
            current_section = [line]
            important = True
        elif stripped.startswith("【") and stripped.endswith("】"):
            if current_section and important:
                keep_sections.extend(current_section)
            current_section = [line]
            important = False
        else:
            current_section.append(line)

    if current_section and important:
        keep_sections.extend(current_section)

    result = "\n".join(keep_sections).strip()
    if len(result) > max_len:
        result = result[:max_len] + "\n..."
    if not result:
        result = text[:max_len]
    return result
