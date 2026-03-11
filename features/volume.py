"""Volume-related feature extraction: volume change, volume profile, session profiles, anchored profiles."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

from ._base import FeatureBase


class VolumeMixin(FeatureBase):
    @staticmethod
    def extract_volume_change(klines_by_timeframe: Dict[str, Sequence[Dict]]) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        for timeframe, candles in klines_by_timeframe.items():
            rows = list(candles)
            if not rows:
                result[timeframe] = {
                    "last_volume": 0.0,
                    "prev_volume": 0.0,
                    "delta_volume": 0.0,
                    "delta_pct": 0.0,
                    "avg20_volume": 0.0,
                    "vs_avg20_pct": 0.0,
                }
                continue

            last_volume = rows[-1]["volume"]
            prev_volume = rows[-2]["volume"] if len(rows) > 1 else 0.0
            delta_volume = last_volume - prev_volume
            delta_pct = 0.0 if prev_volume <= 0 else delta_volume / prev_volume * 100
            lookback = rows[-20:]
            avg20 = sum(row["volume"] for row in lookback) / len(lookback)
            vs_avg20_pct = 0.0 if avg20 <= 0 else (last_volume - avg20) / avg20 * 100

            result[timeframe] = {
                "last_volume": round(last_volume, 6),
                "prev_volume": round(prev_volume, 6),
                "delta_volume": round(delta_volume, 6),
                "delta_pct": round(delta_pct, 6),
                "avg20_volume": round(avg20, 6),
                "vs_avg20_pct": round(vs_avg20_pct, 6),
            }
        return result

    @staticmethod
    def extract_volume_profile(candles: Sequence[Dict], bins: int = 24, window: int = 72) -> Dict:
        rows = list(candles[-window:]) if window > 0 else list(candles)
        if not rows:
            return {
                "window_size": 0,
                "bins": bins,
                "price_low": 0.0,
                "price_high": 0.0,
                "poc_price": 0.0,
                "hvn_prices": [],
                "lvn_prices": [],
            }

        low = min(float(row["low"]) for row in rows)
        high = max(float(row["high"]) for row in rows)
        if high <= low or bins <= 0:
            midpoint = (high + low) / 2 if (high + low) > 0 else 0.0
            return {
                "window_size": len(rows),
                "bins": max(1, bins),
                "price_low": round(low, 6),
                "price_high": round(high, 6),
                "poc_price": round(midpoint, 6),
                "hvn_prices": [round(midpoint, 6)],
                "lvn_prices": [round(midpoint, 6)],
            }

        bin_size = (high - low) / bins
        volume_bins = [0.0 for _ in range(bins)]
        centers = [low + (idx + 0.5) * bin_size for idx in range(bins)]

        for row in rows:
            typical_price = (float(row["high"]) + float(row["low"]) + float(row["close"])) / 3
            raw_idx = int((typical_price - low) / bin_size)
            idx = min(bins - 1, max(0, raw_idx))
            volume_bins[idx] += float(row["volume"])

        poc_idx = max(range(bins), key=lambda idx: volume_bins[idx])
        ranked = sorted(range(bins), key=lambda idx: volume_bins[idx], reverse=True)
        hvn_idx = ranked[: min(3, bins)]
        non_zero_idx = [idx for idx in range(bins) if volume_bins[idx] > 0]
        lvn_ranked = sorted(non_zero_idx, key=lambda idx: volume_bins[idx])
        lvn_idx = lvn_ranked[: min(3, len(lvn_ranked))]

        return {
            "window_size": len(rows),
            "bins": bins,
            "price_low": round(low, 6),
            "price_high": round(high, 6),
            "bin_size": round(bin_size, 6),
            "poc_price": round(centers[poc_idx], 6),
            "hvn_prices": [round(centers[idx], 6) for idx in hvn_idx],
            "lvn_prices": [round(centers[idx], 6) for idx in lvn_idx],
        }

    @classmethod
    def extract_session_profiles(cls, candles: Sequence[Dict], bins: int = 24) -> Dict:
        rows = list(candles)
        if not rows:
            return {
                "session_clock": "UTC_8H",
                "current_session": "unknown",
                "profiles": {},
            }

        session_labels = ("asia", "europe", "us")
        groups: Dict[str, List[Dict]] = {label: [] for label in session_labels}
        for row in rows[-96:]:
            open_time = int(row.get("open_time", 0))
            hour = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).hour
            if 0 <= hour < 8:
                groups["asia"].append(row)
            elif 8 <= hour < 16:
                groups["europe"].append(row)
            else:
                groups["us"].append(row)

        latest_hour = datetime.fromtimestamp(int(rows[-1].get("open_time", 0)) / 1000, tz=timezone.utc).hour
        if 0 <= latest_hour < 8:
            current_session = "asia"
        elif 8 <= latest_hour < 16:
            current_session = "europe"
        else:
            current_session = "us"

        profiles: Dict[str, Dict] = {}
        for label, session_rows in groups.items():
            profile = cls.extract_volume_profile(session_rows, bins=bins, window=0)
            profile["from_open_time"] = int(session_rows[0]["open_time"]) if session_rows else 0
            profile["to_close_time"] = int(session_rows[-1]["close_time"]) if session_rows else 0
            profile["high"] = round(max((float(row["high"]) for row in session_rows), default=0.0), 6)
            profile["low"] = round(min((float(row["low"]) for row in session_rows), default=0.0), 6)
            profiles[label] = profile

        return {
            "session_clock": "UTC_8H",
            "current_session": current_session,
            "profiles": profiles,
        }

    @staticmethod
    def _anchored_vwap(candles: Sequence[Dict]) -> float:
        rows = list(candles)
        if not rows:
            return 0.0
        total_quote = 0.0
        total_volume = 0.0
        for row in rows:
            typical_price = (float(row["high"]) + float(row["low"]) + float(row["close"])) / 3
            volume = float(row["volume"])
            total_quote += typical_price * volume
            total_volume += volume
        return 0.0 if total_volume == 0 else total_quote / total_volume

    @staticmethod
    def _recent_swings(candles: Sequence[Dict], pivot: int = 2, lookback: int = 72) -> List[Tuple[str, int]]:
        rows = list(candles[-lookback:])
        if len(rows) < pivot * 2 + 1:
            return []

        swings: List[Tuple[str, int]] = []
        offset = len(candles) - len(rows)
        for idx in range(pivot, len(rows) - pivot):
            center = rows[idx]
            highs = [float(row["high"]) for row in rows[idx - pivot : idx + pivot + 1]]
            lows = [float(row["low"]) for row in rows[idx - pivot : idx + pivot + 1]]
            if float(center["high"]) >= max(highs):
                swings.append(("recent_swing_high", offset + idx))
            if float(center["low"]) <= min(lows):
                swings.append(("recent_swing_low", offset + idx))
        return swings

    @classmethod
    def extract_anchored_profiles(cls, candles: Sequence[Dict], bins: int = 24) -> List[Dict]:
        rows = list(candles)
        if not rows:
            return []

        anchors = cls._recent_swings(rows)
        deduped: List[Tuple[str, int]] = []
        seen_types: set[str] = set()
        for anchor_type, idx in reversed(anchors):
            if anchor_type in seen_types:
                continue
            deduped.append((anchor_type, idx))
            seen_types.add(anchor_type)
            if len(deduped) >= 2:
                break
        deduped.reverse()

        output: List[Dict] = []
        current_price = float(rows[-1].get("close", 0.0))
        for anchor_type, idx in deduped:
            anchor_rows = rows[idx:]
            anchor_row = rows[idx]
            profile = cls.extract_volume_profile(anchor_rows, bins=bins, window=0)
            vwap = cls._anchored_vwap(anchor_rows)
            distance_bps = cls._safe_div(current_price - vwap, vwap) * 10000 if vwap > 0 else 0.0
            output.append(
                {
                    "anchor_type": anchor_type,
                    "anchor_open_time": int(anchor_row.get("open_time", 0)),
                    "anchor_price": round(float(anchor_row.get("high" if "high" in anchor_type else "low", anchor_row.get("close", 0.0))), 6),
                    "window_size": len(anchor_rows),
                    "anchored_vwap": round(vwap, 6),
                    "distance_to_vwap_bps": round(distance_bps, 6),
                    "poc_price": profile.get("poc_price", 0.0),
                    "hvn_prices": profile.get("hvn_prices", []),
                    "lvn_prices": profile.get("lvn_prices", []),
                }
            )
        return output
