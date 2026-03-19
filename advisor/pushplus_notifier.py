"""Send trading notifications via PushPlus (微信推送)."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Optional

logger = logging.getLogger("btc_context.pushplus")

try:
    from .analysis_parser import is_wait_decision
except ImportError:
    from advisor.analysis_parser import is_wait_decision  # type: ignore


class PushPlusNotifier:
    API_URL = "https://www.pushplus.plus/send"

    def __init__(self, token: str) -> None:
        self.token = token

    def send(self, title: str, content: str, template: str = "txt") -> bool:
        import time as _time
        payload = {
            "token": self.token,
            "title": title,
            "content": content,
            "template": template,
        }
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = subprocess.run(
                    ["curl", "-s", "-m", "20", self.API_URL,
                     "-X", "POST",
                     "-H", "Content-Type: application/json",
                     "-d", json.dumps(payload, ensure_ascii=False)],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=25,
                )
                if result.returncode != 0:
                    logger.warning("PushPlus curl attempt %d/%d failed: %s",
                                   attempt, max_attempts, result.stderr or "(no stderr)")
                    if attempt < max_attempts:
                        _time.sleep(3)
                        continue
                    return False
                resp = json.loads(result.stdout)
                if resp.get("code") == 200:
                    logger.info("PushPlus sent OK: %s", title)
                    return True
                logger.warning("PushPlus returned code=%s: %s", resp.get("code"), resp.get("msg"))
                return False
            except Exception as exc:
                logger.warning("PushPlus attempt %d/%d error: %s", attempt, max_attempts, exc)
                if attempt < max_attempts:
                    _time.sleep(3)
                    continue
                return False
        return False

    def send_trade_signal(self, analysis_text: str, context: dict) -> bool:
        """Parse AI analysis and send full analysis as trade notification."""
        price = context.get("price", 0)
        symbol = context.get("symbol", "BTCUSDT")

        is_wait = _is_wait_decision(analysis_text)
        if is_wait:
            return False

        title = f"📊 {symbol} 交易信号 @{price:.0f}"
        body = _full_content(analysis_text, max_len=4000)
        return self.send(title, body, template="markdown")

    def send_status(self, message: str) -> bool:
        return self.send("BTC Monitor", message)


def _is_wait_decision(text: str) -> bool:
    return is_wait_decision(text)


def _extract_signal_summary(text: str, max_len: int = 4000) -> str:
    lines = text.split("\n")
    keep_sections: list[str] = []
    current_section: list[str] = []
    important = False

    important_headers = (
        "【主判断】", "【执行与风险】", "【当前动作】", "【执行模式】", "【执行方案】",
        "【持仓主判断】", "【持仓主结论】", "【持仓处理】",
        "【一句人话总结】",
        "immediate_entry_plan:", "pullback_plan:", "trigger_plan:",
    )

    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(h) for h in important_headers):
            if current_section and important:
                keep_sections.extend(current_section)
                keep_sections.append("")
            # 将 【XXX】 转成 markdown 加粗标题
            header_line = f"**{stripped}**" if stripped.startswith("【") else line
            current_section = [header_line]
            important = True
        elif stripped.startswith("【") and stripped.endswith("】"):
            if current_section and important:
                keep_sections.extend(current_section)
                keep_sections.append("")
            header_line = f"**{stripped}**"
            current_section = [header_line]
            important = False
        else:
            current_section.append(line)

    if current_section and important:
        keep_sections.extend(current_section)

    result = "\n".join(keep_sections).strip()
    if len(result) > max_len:
        # 截断时尽量在换行处切，避免切断句子
        cutoff = result.rfind("\n", 0, max_len)
        if cutoff < max_len // 2:
            cutoff = max_len
        result = result[:cutoff] + "\n\n> *(内容过长，已截断)*"
    if not result:
        result = text[:max_len]
    return result


def _full_content(text: str, max_len: int = 4000) -> str:
    """Return the full analysis text, truncating gracefully only if necessary."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    cutoff = text.rfind("\n", 0, max_len)
    if cutoff < max_len // 2:
        cutoff = max_len
    return text[:cutoff] + "\n\n> *(内容过长，已截断)*"
