"""Track recent AI directional calls and their outcomes for bias calibration.

Records each non-wait AI analysis alongside the BTC price at the time of the call.
On the next run, outcomes are filled in using already-fetched 1h klines (zero extra
API calls). The resulting compact history block is injected into the prompt as
calibration context so the AI can detect systematic directional bias in its recent calls.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("btc_context.analysis_history")

try:
    from ..config import OUTPUT_DIR
except ImportError:
    from config import OUTPUT_DIR

try:
    from .analysis_parser import parse_analysis_snapshot
except ImportError:
    from advisor.analysis_parser import parse_analysis_snapshot  # type: ignore

_HISTORY_FILE = OUTPUT_DIR / ".analysis_history.json"
_MAX_ENTRIES = 10
_TZ8 = timezone(timedelta(hours=8))


def _parse_field(text: str, key: str) -> str:
    """Extract value of the first matching 'key: value' line."""
    m = re.search(rf"^{re.escape(key)}\s*[:：]\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _parse_direction(analysis_text: str) -> str:
    snapshot = parse_analysis_snapshot(analysis_text)
    if snapshot.get("has_open_position"):
        return ""
    direction = snapshot.get("primary_decision")
    if direction == "long":
        return "开多"
    if direction == "short":
        return "开空"
    if direction == "wait":
        return "等待"
    return ""


def _lookup_price_after(klines_1h: List[Dict], ref_ts_ms: int, bars_ahead: int) -> Optional[float]:
    """Find the close price of the bar `bars_ahead` candles after ref_ts_ms in 1h klines."""
    if not klines_1h:
        return None
    # Find the index of the bar whose open_time is closest to and >= ref_ts_ms
    target_idx = None
    for i, k in enumerate(klines_1h):
        if int(k.get("open_time", 0)) >= ref_ts_ms:
            target_idx = i
            break
    if target_idx is None:
        return None
    result_idx = target_idx + bars_ahead
    if result_idx >= len(klines_1h):
        return None
    return float(klines_1h[result_idx].get("close", 0) or 0)


class AnalysisHistory:
    """Persist recent AI analysis calls with price outcomes for prompt calibration."""

    def __init__(self, history_file: Path = _HISTORY_FILE) -> None:
        self._file = history_file

    # ── public API ────────────────────────────────────────────────────────────

    def record(self, analysis_text: str, price: float) -> None:
        """Parse key fields from analysis_text and append to history. Keeps last _MAX_ENTRIES."""
        direction = _parse_direction(analysis_text)
        if not direction:
            logger.debug("analysis_history: could not parse direction, skipping record")
            return
        snapshot = parse_analysis_snapshot(analysis_text)

        entry: Dict = {
            "ts_ms": int(time.time() * 1000),
            "price": round(float(price), 2),
            "direction": direction,
            "execution_mode": str(snapshot.get("execution_mode", "")),
            "confidence": _parse_field(analysis_text, "confidence"),
            "stop_loss": str(snapshot.get("stop_loss", "")),
            "outcome_1h_price": None,
            "outcome_4h_price": None,
            "outcome_resolved": False,
        }

        history = self._load()
        history.append(entry)
        history = history[-_MAX_ENTRIES:]
        self._save(history)
        logger.debug("analysis_history: recorded %s @%.0f", direction, price)

    def update_outcomes(self, klines_1h: List[Dict]) -> None:
        """Fill in outcome prices for unresolved entries using already-fetched 1h klines."""
        history = self._load()
        changed = False
        now_ms = int(time.time() * 1000)

        for entry in history:
            if entry.get("outcome_resolved"):
                continue
            ts_ms = int(entry.get("ts_ms", 0))
            # Only resolve after enough time has passed (4h minimum)
            if now_ms - ts_ms < 4 * 3600 * 1000:
                # Try filling 1h outcome if 1h has passed
                if now_ms - ts_ms >= 1 * 3600 * 1000 and entry.get("outcome_1h_price") is None:
                    p1h = _lookup_price_after(klines_1h, ts_ms, 1)
                    if p1h:
                        entry["outcome_1h_price"] = round(p1h, 2)
                        changed = True
                continue

            if entry.get("outcome_1h_price") is None:
                p1h = _lookup_price_after(klines_1h, ts_ms, 1)
                if p1h:
                    entry["outcome_1h_price"] = round(p1h, 2)
                    changed = True

            if entry.get("outcome_4h_price") is None:
                p4h = _lookup_price_after(klines_1h, ts_ms, 4)
                if p4h:
                    entry["outcome_4h_price"] = round(p4h, 2)
                    changed = True

            if entry.get("outcome_1h_price") and entry.get("outcome_4h_price"):
                entry["outcome_resolved"] = True
                changed = True

        if changed:
            self._save(history)

    def get_context_block(self) -> str:
        """Return a compact history string for prompt injection. Empty string if no history."""
        history = self._load()
        if not history:
            return ""

        lines = [
            "=== PRIOR_DECISIONS_CONTEXT ===",
            "[校准参考 — 请在完成 PHASE A 独立判断后再查阅此段，勿将历史决策作为当次方向依据]",
        ]

        direction_counts: Dict[str, int] = {}
        tradeable_entries = [e for e in history if e.get("direction") not in ("等待", "")]

        for entry in history[-5:]:
            ts = int(entry.get("ts_ms", 0))
            dt = datetime.fromtimestamp(ts / 1000, tz=_TZ8)
            time_str = dt.strftime("%m-%d %H:%M")
            price = entry.get("price", 0)
            direction = entry.get("direction", "?")
            sl = entry.get("stop_loss", "")
            p1h = entry.get("outcome_1h_price")
            p4h = entry.get("outcome_4h_price")

            direction_counts[direction] = direction_counts.get(direction, 0) + 1

            parts = [f"{time_str}  BTC@{price:.0f}  结论:{direction}"]
            if sl:
                parts.append(f" stop:{sl}")

            if p1h:
                delta1h = p1h - price
                hit1h = "HIT" if (direction == "开多" and delta1h > 0) or (direction == "开空" and delta1h < 0) else "MISS"
                parts.append(f"  → 1h:{p1h:.0f}({delta1h:+.0f})[{hit1h}]")
            else:
                parts.append("  → 1h:pending")

            if p4h:
                delta4h = p4h - price
                hit4h = "HIT" if (direction == "开多" and delta4h > 0) or (direction == "开空" and delta4h < 0) else "MISS"
                parts.append(f"  4h:{p4h:.0f}({delta4h:+.0f})[{hit4h}]")
            else:
                parts.append("  4h:pending")

            lines.append("".join(parts))

        # Bias summary
        if tradeable_entries:
            long_n = direction_counts.get("开多", 0)
            short_n = direction_counts.get("开空", 0)
            total_dir = long_n + short_n
            if total_dir > 0:
                if long_n > short_n:
                    bias = f"long_heavy ({long_n}多/{short_n}空/{total_dir}次)"
                elif short_n > long_n:
                    bias = f"short_heavy ({short_n}空/{long_n}多/{total_dir}次)"
                else:
                    bias = f"balanced ({long_n}多/{short_n}空)"
                lines.append(f"recent_direction_bias: {bias}")

        return "\n".join(lines)

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self) -> List[Dict]:
        if not self._file.is_file():
            return []
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("analysis_history: load failed: %s", exc)
            return []

    def _save(self, history: List[Dict]) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("analysis_history: save failed: %s", exc)
