"""Trading discipline risk monitor.

Checks 4 rules against today's Binance Futures income records and sends
PushPlus WeChat alerts when a rule is triggered. Runs at the start of each
--monitor loop iteration. Failures are non-fatal: exceptions are caught and
logged as warnings so the main market-analysis flow is never interrupted.

Rules:
  1. Daily loss limit    — today's net P&L < -40 USDT
  2. Revenge cooldown    — 2 losing trades within 60 minutes
  3. Post-big-win warn   — yesterday net P&L > +50 USDT (alert once at day start)
  4. High frequency      — >= 5 realized PnL events in the last 60 minutes
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("btc_context.risk_monitor")

try:
    from ..config import OUTPUT_DIR
except ImportError:
    from config import OUTPUT_DIR

_STATE_FILE = OUTPUT_DIR / ".risk_monitor_state.json"
_TZ8 = timezone(timedelta(hours=8))

# ── configurable thresholds ───────────────────────────────────────────────────
DAILY_LOSS_LIMIT = -40.0          # USDT — alert when today net < this
DAILY_LOSS_STEP = 10.0            # USDT — re-alert every additional N USDT loss
BIG_WIN_THRESHOLD = 50.0          # USDT — yesterday profit that triggers caution today
REVENGE_WINDOW_MINUTES = 60       # minutes — look-back window for consecutive losses
REVENGE_MIN_LOSSES = 2            # losses within the window to trigger cooldown
FREQ_WINDOW_MINUTES = 60          # minutes — rolling window for frequency check
FREQ_MAX_TRADES = 5               # trades within the window before alerting
FREQ_ALERT_COOLDOWN_MINUTES = 60  # minutes — suppress repeated frequency alerts


def _today_str() -> str:
    return datetime.now(tz=_TZ8).strftime("%Y-%m-%d")


def _yesterday_str() -> str:
    return (datetime.now(tz=_TZ8) - timedelta(days=1)).strftime("%Y-%m-%d")


def _day_of_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=_TZ8).strftime("%Y-%m-%d")


class RiskMonitor:
    """Check trading discipline rules and push WeChat alerts when triggered."""

    def __init__(self, state_file: Path = _STATE_FILE) -> None:
        self._state_file = state_file

    # ── main entry point ──────────────────────────────────────────────────────

    def check_and_alert(self, collector, pushplus_token: str) -> None:
        """Fetch today's income, check all rules, send alerts as needed.

        Args:
            collector: BinanceFuturesCollector instance (already authenticated).
            pushplus_token: PushPlus token string for WeChat delivery.
        """
        try:
            self._run(collector, pushplus_token)
        except Exception as exc:
            logger.warning("risk_monitor: check failed (non-fatal): %s", exc)

    # ── internal ──────────────────────────────────────────────────────────────

    def _run(self, collector, pushplus_token: str) -> None:
        state = self._load_state()
        now_ms = int(time.time() * 1000)
        today = _today_str()
        yesterday = _yesterday_str()

        # Reset per-day counters when the calendar day rolls over
        if state.get("alert_date") != today:
            state = {
                "alert_date": today,
                "daily_loss_alerted_level": 0.0,
                "big_win_warned": False,
                "last_freq_alert_ms": 0,
            }

        # Fetch today's income records (realized PnL + commission)
        today_income = self._fetch_today_income(collector)
        if today_income is None:
            return  # API error; skip silently

        # Separate yesterday's records for big-win check (fetched in one call below)
        yesterday_income = self._fetch_yesterday_income(collector)

        alerts: List[str] = []

        # ── Rule 1: daily loss limit ──────────────────────────────────────────
        today_net = self._net_pnl(today_income, today)
        if today_net < DAILY_LOSS_LIMIT:
            # Alert on first breach, then every DAILY_LOSS_STEP further
            prev_level = float(state.get("daily_loss_alerted_level", 0))
            threshold_crossed = int(today_net / DAILY_LOSS_STEP) * DAILY_LOSS_STEP
            if threshold_crossed < prev_level or prev_level == 0:
                alerts.append(
                    f"⛔ 【日亏损限额】今日净亏损已达 {today_net:.1f} U\n"
                    f"建议立即停止交易，冷静复盘。"
                )
                state["daily_loss_alerted_level"] = float(threshold_crossed)

        # ── Rule 2: revenge-trade cooldown ────────────────────────────────────
        window_start_ms = now_ms - REVENGE_WINDOW_MINUTES * 60 * 1000
        recent_losses = [
            r for r in today_income
            if r.get("incomeType") == "REALIZED_PNL"
            and float(r.get("income", 0)) < 0
            and int(r.get("time", 0)) >= window_start_ms
        ]
        if len(recent_losses) >= REVENGE_MIN_LOSSES:
            loss_sum = sum(float(r["income"]) for r in recent_losses)
            alerts.append(
                f"🚨 【复仇冷静期】过去 {REVENGE_WINDOW_MINUTES} 分钟内连续亏损 "
                f"{len(recent_losses)} 次（合计 {loss_sum:.1f} U）\n"
                f"请暂停交易至少 {REVENGE_WINDOW_MINUTES} 分钟，避免情绪驱动下单。"
            )

        # ── Rule 3: post-big-win warning (once per day, first check) ─────────
        if not state.get("big_win_warned") and yesterday_income is not None:
            yesterday_net = self._net_pnl(yesterday_income, yesterday)
            if yesterday_net >= BIG_WIN_THRESHOLD:
                alerts.append(
                    f"⚠️ 【大赢次日提醒】昨天净盈利 {yesterday_net:.1f} U\n"
                    f"今天请注意控制仓位，避免昨天盈利带来的过度自信。"
                )
            state["big_win_warned"] = True

        # ── Rule 5: high trade frequency ──────────────────────────────────────
        recent_trades = [
            r for r in today_income
            if r.get("incomeType") == "REALIZED_PNL"
            and int(r.get("time", 0)) >= window_start_ms
        ]
        last_freq_alert_ms = int(state.get("last_freq_alert_ms", 0))
        if (
            len(recent_trades) >= FREQ_MAX_TRADES
            and now_ms - last_freq_alert_ms > FREQ_ALERT_COOLDOWN_MINUTES * 60 * 1000
        ):
            alerts.append(
                f"📊 【交易频率过高】过去 {FREQ_WINDOW_MINUTES} 分钟内已成交 "
                f"{len(recent_trades)} 次\n"
                f"过度交易会显著增加手续费损耗，建议降低频率。"
            )
            state["last_freq_alert_ms"] = now_ms

        # ── Send alerts ───────────────────────────────────────────────────────
        if alerts:
            self._send(pushplus_token, alerts, today_net)

        self._save_state(state)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_today_income(collector) -> Optional[List[Dict]]:
        """Fetch today's income records (00:00 CST to now)."""
        try:
            now_ms = int(time.time() * 1000)
            today_start = datetime.now(tz=_TZ8).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            start_ms = int(today_start.timestamp() * 1000)
            result = collector._signed_get_json(
                "/fapi/v1/income",
                {"startTime": str(start_ms), "endTime": str(now_ms), "limit": "1000"},
            )
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.debug("risk_monitor: fetch today income failed: %s", exc)
            return None

    @staticmethod
    def _fetch_yesterday_income(collector) -> Optional[List[Dict]]:
        """Fetch yesterday's income records for the big-win check."""
        try:
            yesterday_start = (
                datetime.now(tz=_TZ8).replace(hour=0, minute=0, second=0, microsecond=0)
                - timedelta(days=1)
            )
            yesterday_end = yesterday_start + timedelta(days=1) - timedelta(milliseconds=1)
            start_ms = int(yesterday_start.timestamp() * 1000)
            end_ms = int(yesterday_end.timestamp() * 1000)
            result = collector._signed_get_json(
                "/fapi/v1/income",
                {"startTime": str(start_ms), "endTime": str(end_ms), "limit": "1000"},
            )
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.debug("risk_monitor: fetch yesterday income failed: %s", exc)
            return None

    @staticmethod
    def _net_pnl(income_records: List[Dict], day: str) -> float:
        """Sum REALIZED_PNL + COMMISSION + FUNDING_FEE for the given day."""
        total = 0.0
        for r in income_records:
            if _day_of_ts(int(r.get("time", 0))) != day:
                continue
            if r.get("incomeType") in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE"):
                total += float(r.get("income", 0))
        return total

    @staticmethod
    def _send(pushplus_token: str, alerts: List[str], today_net: float) -> None:
        try:
            try:
                from .pushplus_notifier import PushPlusNotifier
            except ImportError:
                from advisor.pushplus_notifier import PushPlusNotifier

            notifier = PushPlusNotifier(token=pushplus_token)
            body = "\n\n".join(alerts)
            body += f"\n\n---\n今日累计净盈亏: {today_net:+.2f} U"
            sent = notifier.send("⚡ BTC 交易风控提醒", body, template="txt")
            if sent:
                logger.info("risk_monitor: alert sent (%d rules triggered)", len(alerts))
            else:
                logger.warning("risk_monitor: PushPlus delivery failed")
        except Exception as exc:
            logger.warning("risk_monitor: send failed: %s", exc)

    def _load_state(self) -> Dict:
        if not self._state_file.is_file():
            return {}
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self, state: Dict) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("risk_monitor: state save failed: %s", exc)
