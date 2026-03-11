"""Session context feature extraction mixin."""

from datetime import datetime, timezone
from typing import Dict, List, Sequence

from ._base import FeatureBase


class SessionMixin(FeatureBase):
    @staticmethod
    def _parse_iso_timestamp(value: str) -> int:
        try:
            return int(datetime.fromisoformat(value).timestamp() * 1000)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _session_name(timestamp_ms: int) -> str:
        hour = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).hour
        if 0 <= hour < 8:
            return "asia"
        if 8 <= hour < 16:
            return "europe"
        return "us"

    @classmethod
    def extract_session_context(cls, candles: Sequence[Dict], funding: Dict, stats_24h: Dict) -> Dict:
        rows = list(candles)
        if not rows:
            return {
                "current_session": "unknown",
                "session_high": 0.0,
                "session_low": 0.0,
                "day_high": round(float(stats_24h.get("high_price", 0.0)), 6),
                "day_low": round(float(stats_24h.get("low_price", 0.0)), 6),
                "funding_countdown_seconds": 0.0,
                "funding_countdown_label": "unknown",
            }

        current_session = cls._session_name(int(rows[-1].get("open_time", 0)))
        session_rows: List[Dict] = []
        for row in reversed(rows):
            if cls._session_name(int(row.get("open_time", 0))) != current_session:
                break
            session_rows.append(row)
        session_rows.reverse()

        day_window = rows[-96:] if len(rows) >= 96 else rows
        session_high = max((float(row.get("high", 0.0)) for row in session_rows), default=0.0)
        session_low = min((float(row.get("low", 0.0)) for row in session_rows), default=0.0)
        day_high = float(stats_24h.get("high_price") or max((float(row.get("high", 0.0)) for row in day_window), default=0.0))
        day_low = float(stats_24h.get("low_price") or min((float(row.get("low", 0.0)) for row in day_window), default=0.0))
        latest_close_ms = int(rows[-1].get("close_time", 0))
        next_funding_ms = cls._parse_iso_timestamp(str(funding.get("next_funding_time") or ""))
        countdown_seconds = max(0.0, (next_funding_ms - latest_close_ms) / 1000) if next_funding_ms else 0.0
        if countdown_seconds >= 3600:
            funding_countdown_label = f"{int(countdown_seconds // 3600)}h {int((countdown_seconds % 3600) // 60)}m"
        elif countdown_seconds >= 60:
            funding_countdown_label = f"{int(countdown_seconds // 60)}m {int(countdown_seconds % 60)}s"
        else:
            funding_countdown_label = f"{int(countdown_seconds)}s"

        return {
            "current_session": current_session,
            "session_high": round(session_high, 6),
            "session_low": round(session_low, 6),
            "session_open_time": int(session_rows[0].get("open_time", 0)) if session_rows else 0,
            "session_close_time": int(session_rows[-1].get("close_time", 0)) if session_rows else 0,
            "day_high": round(day_high, 6),
            "day_low": round(day_low, 6),
            "funding_countdown_seconds": round(countdown_seconds, 6),
            "funding_countdown_label": funding_countdown_label,
        }
