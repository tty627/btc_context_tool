"""Shared utility methods for feature extraction."""

from datetime import datetime, timezone
from typing import Dict, Sequence


class FeatureBase:
    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _price_precision(price: float) -> int:
        absolute = abs(price)
        if absolute >= 1000:
            return 1
        if absolute >= 1:
            return 3
        return 6

    @classmethod
    def _price_key(cls, price: float) -> float:
        return round(float(price), cls._price_precision(price))

    @staticmethod
    def _window_seconds(label: str) -> int:
        mapping = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
        return mapping.get(label, 0)

    @staticmethod
    def _snapshot_timestamp_ms(snapshot: Dict) -> int:
        event_time_ms = snapshot.get("event_time_ms")
        if isinstance(event_time_ms, int):
            return event_time_ms

        raw = snapshot.get("event_time")
        if isinstance(raw, str):
            try:
                return int(datetime.fromisoformat(raw).timestamp() * 1000)
            except ValueError:
                return 0
        return 0

    @staticmethod
    def _quantile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        if q <= 0:
            return min(values)
        if q >= 1:
            return max(values)
        sorted_values = sorted(values)
        index = int((len(sorted_values) - 1) * q)
        return float(sorted_values[index])
