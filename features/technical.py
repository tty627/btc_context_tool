from typing import Dict, Sequence
from ._base import FeatureBase


class TechnicalMixin(FeatureBase):
    @staticmethod
    def classify_trend(ema: Dict[str, float]) -> str:
        ema7 = ema["7"]
        ema25 = ema["25"]
        ema99 = ema["99"]
        if ema7 > ema25 > ema99:
            return "bullish"
        if ema7 < ema25 < ema99:
            return "bearish"
        return "neutral"

    @staticmethod
    def classify_momentum(macd: Dict[str, float]) -> str:
        if macd["dif"] > macd["dea"] and macd["hist"] > 0:
            return "momentum_up"
        if macd["dif"] < macd["dea"] and macd["hist"] < 0:
            return "momentum_down"
        return "momentum_neutral"

    @staticmethod
    def classify_kdj(kdj: Dict[str, float]) -> str:
        k = kdj["k"]
        if k > 80:
            return "overbought"
        if k < 20:
            return "oversold"
        return "neutral"

    @classmethod
    def extract_timeframe_features(cls, indicators_by_timeframe: Dict[str, Dict]) -> Dict[str, Dict]:
        output = {}
        for timeframe, metrics in indicators_by_timeframe.items():
            rsi = metrics.get("rsi", {})
            output[timeframe] = {
                "trend": cls.classify_trend(metrics["ema"]),
                "momentum": cls.classify_momentum(metrics["macd"]),
                "kdj_state": cls.classify_kdj(metrics["kdj"]),
                "rsi_state": rsi.get("state", "unknown"),
                "rsi_divergence": rsi.get("divergence", "none"),
            }
        return output

    @staticmethod
    def extract_recent_4h_range(candles_15m: Sequence[Dict]) -> Dict:
        window = list(candles_15m[-16:])
        if not window:
            return {
                "high": 0.0,
                "low": 0.0,
                "range_abs": 0.0,
                "range_pct": 0.0,
                "from_open_time": 0,
                "to_close_time": 0,
                "high_sweep_zone": {"zone_low": 0.0, "zone_high": 0.0},
                "low_sweep_zone": {"zone_low": 0.0, "zone_high": 0.0},
            }
        recent_high = max(row["high"] for row in window)
        recent_low = min(row["low"] for row in window)
        range_abs = recent_high - recent_low
        range_pct = 0.0 if recent_low <= 0 else range_abs / recent_low * 100
        return {
            "high": round(recent_high, 6),
            "low": round(recent_low, 6),
            "range_abs": round(range_abs, 6),
            "range_pct": round(range_pct, 6),
            "from_open_time": int(window[0]["open_time"]),
            "to_close_time": int(window[-1]["close_time"]),
            "high_sweep_zone": {
                "zone_low": round(recent_high * 0.998, 6),
                "zone_high": round(recent_high * 1.01, 6),
            },
            "low_sweep_zone": {
                "zone_low": round(recent_low * 0.99, 6),
                "zone_high": round(recent_low * 1.002, 6),
            },
        }
