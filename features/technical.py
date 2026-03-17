from datetime import datetime, timezone
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

    @staticmethod
    def extract_daily_anchors(candles_1d: Sequence[Dict]) -> Dict:
        """Extract higher-timeframe anchor levels from daily klines.

        Returns prev_day high/low, weekly VWAP (5-day), and monthly open.
        """
        rows = list(candles_1d)
        if len(rows) < 2:
            return {
                "available": False,
                "prev_day_high": 0.0,
                "prev_day_low": 0.0,
                "weekly_vwap": 0.0,
                "month_open": 0.0,
            }

        prev = rows[-2]
        prev_day_high = float(prev.get("high", 0))
        prev_day_low = float(prev.get("low", 0))

        vwap_window = rows[-5:]
        total_vq = 0.0
        total_vol = 0.0
        for c in vwap_window:
            h, l, cl, v = (
                float(c.get("high", 0)),
                float(c.get("low", 0)),
                float(c.get("close", 0)),
                float(c.get("volume", 0)),
            )
            typical = (h + l + cl) / 3 if (h + l + cl) > 0 else 0.0
            total_vq += typical * v
            total_vol += v
        weekly_vwap = total_vq / total_vol if total_vol > 0 else 0.0

        month_open = 0.0
        for c in rows:
            ot = int(c.get("open_time", 0))
            if ot <= 0:
                continue
            dt = datetime.fromtimestamp(ot / 1000, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            if dt.year == now.year and dt.month == now.month:
                month_open = float(c.get("open", 0))
                break

        week_window = rows[-5:]
        week_high = max(float(c.get("high", 0)) for c in week_window)
        week_low = min(float(c.get("low", 0)) for c in week_window if float(c.get("low", 0)) > 0)

        now = datetime.now(timezone.utc)
        month_candles = [
            c for c in rows
            if int(c.get("open_time", 0)) > 0
            and datetime.fromtimestamp(int(c["open_time"]) / 1000, tz=timezone.utc).month == now.month
            and datetime.fromtimestamp(int(c["open_time"]) / 1000, tz=timezone.utc).year == now.year
        ]
        if month_candles:
            month_high = max(float(c.get("high", 0)) for c in month_candles)
            month_low = min(float(c.get("low", 0)) for c in month_candles if float(c.get("low", 0)) > 0)
        else:
            month_high = 0.0
            month_low = 0.0

        return {
            "available": True,
            "prev_day_high": round(prev_day_high, 2),
            "prev_day_low": round(prev_day_low, 2),
            "weekly_vwap": round(weekly_vwap, 2),
            "week_high": round(week_high, 2),
            "week_low": round(week_low, 2),
            "month_open": round(month_open, 2),
            "month_high": round(month_high, 2),
            "month_low": round(month_low, 2),
        }
