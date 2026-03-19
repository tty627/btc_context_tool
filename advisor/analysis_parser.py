"""Utilities for parsing the final decision-style AI output."""

from __future__ import annotations

import re
from typing import Dict, List

_FIELD_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<value>.*)$")


def _collect_fields(text: str) -> List[Dict[str, object]]:
    fields: List[Dict[str, object]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        match = _FIELD_RE.match(line)
        if not match:
            continue
        fields.append(
            {
                "indent": len(match.group("indent")),
                "key": match.group("key"),
                "value": match.group("value").strip(),
            }
        )
    return fields


def _first_value(fields: List[Dict[str, object]], key: str) -> str:
    for item in fields:
        if item["key"] == key and isinstance(item["value"], str) and item["value"]:
            return item["value"]
    return ""


def _has_key(fields: List[Dict[str, object]], key: str) -> bool:
    return any(item["key"] == key for item in fields)


def _normalize_primary_decision(raw: str) -> str:
    lowered = (raw or "").strip().lower()
    if "开多" in raw or lowered in {"long", "buy"}:
        return "long"
    if "开空" in raw or lowered in {"short", "sell"}:
        return "short"
    if "等待" in raw or "不交易" in raw or lowered == "wait":
        return "wait"
    return ""


def _normalize_position_state(raw: str) -> str:
    lowered = (raw or "").strip().lower()
    if lowered.startswith("flat"):
        return "flat"
    return lowered


def parse_analysis_snapshot(text: str) -> Dict[str, object]:
    """Parse the current decision-template AI output into a stable snapshot."""
    fields = _collect_fields(text)
    primary_decision_raw = _first_value(fields, "primary_decision")
    execution_mode = _first_value(fields, "execution_mode").strip().lower()
    position_state_raw = _first_value(fields, "position_state")
    position_state = _normalize_position_state(position_state_raw)
    position_action = _first_value(fields, "position_action").strip().lower()
    current_position = _first_value(fields, "current_position")
    stop_loss = _first_value(fields, "stop_loss")
    primary_decision = _normalize_primary_decision(primary_decision_raw)
    has_trade_plan = _has_key(fields, "trade_plan")
    has_wait_plan = _has_key(fields, "wait_plan")

    has_open_position = False
    if position_state and position_state != "flat":
        has_open_position = True
    elif current_position:
        has_open_position = True
    elif position_action:
        has_open_position = True
    elif "【持仓主判断】" in text or "【持仓处理】" in text:
        has_open_position = True

    is_wait = False
    if not has_open_position:
        if primary_decision == "wait" or execution_mode == "wait":
            is_wait = True
        elif has_wait_plan and not has_trade_plan:
            is_wait = True

    actionable_signal = False
    if not has_open_position and primary_decision in {"long", "short"}:
        if execution_mode in {"market_now", "limit_pullback", "stop_trigger"}:
            actionable_signal = True
        elif has_trade_plan and not has_wait_plan:
            actionable_signal = True

    return {
        "primary_decision_raw": primary_decision_raw,
        "primary_decision": primary_decision,
        "execution_mode": execution_mode,
        "position_state_raw": position_state_raw,
        "position_state": position_state,
        "position_action": position_action,
        "current_position": current_position,
        "stop_loss": stop_loss,
        "has_trade_plan": has_trade_plan,
        "has_wait_plan": has_wait_plan,
        "has_open_position": has_open_position,
        "is_wait": is_wait,
        "actionable_signal": actionable_signal,
    }


def analysis_has_open_position(text: str) -> bool:
    return bool(parse_analysis_snapshot(text).get("has_open_position"))


def analysis_has_actionable_signal(text: str) -> bool:
    return bool(parse_analysis_snapshot(text).get("actionable_signal"))


def is_wait_decision(text: str) -> bool:
    return bool(parse_analysis_snapshot(text).get("is_wait"))
