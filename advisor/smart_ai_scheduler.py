"""Smart AI 轮询调度：连续跳过 N 次后强制分析；只基于最终决策切换快速模式。"""

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

try:
    from .analysis_parser import (
        analysis_has_actionable_signal as _analysis_has_actionable_signal,
        analysis_has_open_position as _analysis_has_open_position,
    )
except ImportError:
    from advisor.analysis_parser import (  # type: ignore
        analysis_has_actionable_signal as _analysis_has_actionable_signal,
        analysis_has_open_position as _analysis_has_open_position,
    )

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
    """检测最终 AI 输出中是否存在未平仓持仓（持仓管理场景）。"""
    return _analysis_has_open_position(text)


def analysis_has_actionable_signal(text: str) -> bool:
    """是否存在空仓场景下的可执行新单方案。"""
    return _analysis_has_actionable_signal(text)
