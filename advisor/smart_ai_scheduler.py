"""Smart AI 轮询调度：连续跳过 N 次后强制分析；有信号时每 FAST_POLL 秒分析直到 wait。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("btc_context.smart_scheduler")

try:
    from ..config import OUTPUT_DIR
except ImportError:
    from config import OUTPUT_DIR

STATE_FILE = OUTPUT_DIR / ".smart_ai_scheduler.json"
SKIPS_BEFORE_FORCE = 3
FAST_POLL_SECONDS = 240  # 4 分钟


def _default_state() -> Dict[str, Any]:
    return {"consecutive_skips": 0, "fast_ai_mode": False}


def load_scheduler_state() -> Dict[str, Any]:
    st = _default_state()
    if not STATE_FILE.is_file():
        return st
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        st["consecutive_skips"] = int(data.get("consecutive_skips", 0))
        st["fast_ai_mode"] = bool(data.get("fast_ai_mode", False))
        st["consecutive_skips"] = max(0, min(st["consecutive_skips"], 1000))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("smart scheduler state load failed: %s", exc)
    return st


def save_scheduler_state(state: Dict[str, Any]) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps(
                {
                    "consecutive_skips": int(state.get("consecutive_skips", 0)),
                    "fast_ai_mode": bool(state.get("fast_ai_mode", False)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("smart scheduler state save failed: %s", exc)


def analysis_has_open_position(text: str) -> bool:
    """检测 AI 输出中是否存在未平仓持仓（持仓管理场景）。有持仓就必须维持快速模式。"""
    if not (text or "").strip():
        return False
    lower = text.lower()
    # position_state: flat 表示空仓，其余均视为有持仓
    if "position_state: flat" in lower or "position_state:flat" in lower:
        return False
    # 有持仓管理字段就判定为持仓中
    has_position_mgmt = (
        "current_position:" in lower
        or "position_action:" in lower
        or "reduce_plan:" in lower
        or "hold_conditions:" in lower
        or "exit_trigger:" in lower
        or "hard_invalidation:" in lower
        or "soft_invalidation:" in lower
        or "【持仓处理】" in text
        or "【持仓主判断】" in text
    )
    return has_position_mgmt


def analysis_has_actionable_signal(text: str) -> bool:
    """与 PushPlus 一致：有执行方案且非 wait。"""
    if not (text or "").strip():
        return False
    try:
        try:
            from .pushplus_notifier import _is_wait_decision
        except ImportError:
            from advisor.pushplus_notifier import _is_wait_decision
        if _is_wait_decision(text):
            return False
    except Exception:
        return False
    lower = text.lower()
    # 旧格式：pullback_plan / trigger_plan / immediate_entry_plan
    # 新格式：execution_mode: limit_pullback / immediate_entry / stop_trigger / market_now
    #         + trade_plan: status: armed
    has_plan = (
        "pullback_plan:" in lower
        or "trigger_plan:" in lower
        or "immediate_entry_plan:" in lower
        or "execution_mode: limit_pullback" in lower
        or "execution_mode:limit_pullback" in lower
        or "execution_mode: immediate_entry" in lower
        or "execution_mode:immediate_entry" in lower
        or "execution_mode: stop_trigger" in lower
        or "execution_mode:stop_trigger" in lower
        or "execution_mode: market_now" in lower
        or "execution_mode:market_now" in lower
        or ("trade_plan:" in lower and "status: armed" in lower)
    )
    return has_plan
