"""Detect whether market context has changed enough to warrant an AI analysis.

Compares current context snapshot against the last-analyzed snapshot.
Returns True only when at least one material change is detected,
so the caller can skip the AI call (and save money) when nothing moved.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("btc_context.change_detector")

_DEFAULT_STATE_FILE = Path(__file__).resolve().parents[1] / "output" / ".last_analysis_state.json"


class ChangeDetector:
    """Compare current context against the last snapshot to decide
    whether an AI analysis call is justified."""

    def __init__(
        self,
        price_pct: float = 0.4,
        oi_pct: float = 1.5,
        max_stale_minutes: float = 20,
        state_file: Path = _DEFAULT_STATE_FILE,
    ) -> None:
        self.price_pct = price_pct
        self.oi_pct = oi_pct
        self.max_stale_minutes = max_stale_minutes
        self.state_file = state_file

    def should_analyze(self, context: Dict) -> tuple[bool, List[str]]:
        """Return (should_call_ai, list_of_reasons).

        If no previous state exists, always returns True.
        """
        acc = context.get("account_positions", {})
        has_position = False
        if acc.get("available"):
            sym = acc.get("symbol_position")
            if sym and abs(float(sym.get("position_amt", 0) or 0)) > 0:
                has_position = True
        if has_position:
            return True, ["has_open_position"]

        prev = self._load_state()
        if prev is None:
            return True, ["first_run"]

        if prev.get("had_actionable_signal"):
            return True, ["previous_analysis_had_pending_order"]

        reasons: List[str] = []

        age_min = (time.time() - prev.get("timestamp", 0)) / 60
        if age_min >= self.max_stale_minutes:
            reasons.append(f"stale ({age_min:.0f}min since last analysis)")

        cur_price = float(context.get("price", 0))
        prev_price = float(prev.get("price", 0))
        if prev_price > 0 and cur_price > 0:
            pct = abs(cur_price - prev_price) / prev_price * 100
            if pct >= self.price_pct:
                reasons.append(f"price moved {pct:.2f}%")

        cur_oi = float(context.get("open_interest", 0))
        prev_oi = float(prev.get("open_interest", 0))
        if prev_oi > 0 and cur_oi > 0:
            oi_chg = abs(cur_oi - prev_oi) / prev_oi * 100
            if oi_chg >= self.oi_pct:
                reasons.append(f"OI changed {oi_chg:.2f}%")

        cur_session = context.get("session_context", {}).get("current_session", "")
        prev_session = prev.get("session", "")
        if cur_session and prev_session and cur_session != prev_session:
            reasons.append(f"session changed {prev_session}->{cur_session}")

        cur_gates = self._extract_gates(context)
        prev_gates = prev.get("gates", {})
        for gate_name, cur_val in cur_gates.items():
            if prev_gates.get(gate_name) != cur_val:
                reasons.append(f"gate flip: {gate_name}={cur_val}")

        cur_trend = self._extract_trend(context)
        prev_trend = prev.get("trend", {})
        for tf, state in cur_trend.items():
            if prev_trend.get(tf) != state:
                reasons.append(f"trend change: {tf} {prev_trend.get(tf, '?')}->{state}")

        if reasons:
            return True, reasons
        return False, []

    def save_state(self, context: Dict, analysis_text: str = "") -> None:
        had_signal = False
        if analysis_text:
            wait_markers = (
                "execution_mode: wait", "execution_mode:wait",
                "主结论: 等待", "主结论: 不交易", "主结论:等待", "主结论:不交易",
            )
            has_plan = ("pullback_plan:" in analysis_text
                        or "trigger_plan:" in analysis_text
                        or "immediate_entry_plan:" in analysis_text)
            is_wait = any(m in analysis_text or m in analysis_text.lower() for m in wait_markers)
            had_signal = has_plan and not is_wait

        state = {
            "timestamp": time.time(),
            "price": float(context.get("price", 0)),
            "open_interest": float(context.get("open_interest", 0)),
            "session": context.get("session_context", {}).get("current_session", ""),
            "gates": self._extract_gates(context),
            "trend": self._extract_trend(context),
            "had_actionable_signal": had_signal,
        }
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(state), encoding="utf-8")
        except OSError as exc:
            logger.warning("failed to save analysis state: %s", exc)

    def _load_state(self) -> Optional[Dict]:
        if not self.state_file.exists():
            return None
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _extract_gates(context: Dict) -> Dict[str, str]:
        tf = context.get("trade_flow", {})
        windows = tf.get("windows", {})
        od = context.get("orderbook_dynamics", {})
        gates = {}
        for label, threshold in (("15m", 0.30), ("30m", 0.20)):
            w = windows.get(label, {})
            r = w.get("coverage_ratio")
            if r is not None:
                gates[f"{label}_flow"] = "PASS" if float(r) >= threshold else "FAIL"
        spoof = od.get("spoofing_risk", "unknown")
        dur = float(od.get("sample_duration_seconds", 0) or 0)
        if dur < 20 or spoof in ("high", "medium"):
            gates["dom"] = "BLOCKED"
        else:
            gates["dom"] = "PASS"
        return gates

    @staticmethod
    def _extract_trend(context: Dict) -> Dict[str, str]:
        ms = context.get("market_structure", {})
        if isinstance(ms, dict):
            return {k: str(v) for k, v in ms.items()}
        return {}
