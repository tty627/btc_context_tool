"""Candle structure feature extraction.

Computes human-readable structural descriptions for the most recent N candles
on key timeframes (4h, 1h).  Designed to surface the information that chart
images cannot reliably deliver as text: exact body/wick ratios, close position,
volume context, and basic pattern labels.
"""

from typing import Dict, List, Sequence

from ._base import FeatureBase


class CandleStructureMixin(FeatureBase):

    @staticmethod
    def _candle_direction(open_: float, close: float, body_pct: float) -> str:
        if body_pct < 15.0:
            return "doji"
        return "bullish" if close >= open_ else "bearish"

    @staticmethod
    def _close_position(open_: float, close: float, high: float, low: float) -> str:
        """Where does the close sit within the full range?"""
        rng = high - low
        if rng <= 0:
            return "mid"
        pos = (close - low) / rng
        if pos >= 0.67:
            return "upper_third"
        if pos <= 0.33:
            return "lower_third"
        return "mid"

    @staticmethod
    def _detect_pattern(
        candles: Sequence[Dict],
        idx: int,
        body_pct: float,
        lower_wick_pct: float,
        upper_wick_pct: float,
        direction: str,
    ) -> str:
        """Detect simple 1-2 bar patterns.  Returns the most specific match."""
        c = candles[idx]
        o, h, l_, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
        rng = h - l_
        body = abs(cl - o)

        # ── Pin bar: one wick dominates (>3× body), body small ──────────────
        if body > 0:
            if lower_wick_pct > 0 and (l_ < min(o, cl)) and (min(o, cl) - l_) > 3 * body:
                return "pin_bar_bull"
            if upper_wick_pct > 0 and (max(o, cl) < h) and (h - max(o, cl)) > 3 * body:
                return "pin_bar_bear"

        # ── Hammer (bullish): lower shadow > 2× body, close in upper third ──
        if direction == "bullish" and body > 0:
            lower_shadow = min(o, cl) - l_
            upper_shadow = h - max(o, cl)
            if lower_shadow > 2 * body and upper_shadow < body:
                return "hammer"

        # ── Shooting star (bearish): upper shadow > 2× body, small lower wick
        if direction == "bearish" and body > 0:
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l_
            if upper_shadow > 2 * body and lower_shadow < body:
                return "shooting_star"

        # ── Doji ─────────────────────────────────────────────────────────────
        if direction == "doji":
            return "doji"

        # ── Engulfing (requires previous candle) ─────────────────────────────
        if idx > 0:
            prev = candles[idx - 1]
            po, pcl = float(prev["open"]), float(prev["close"])
            prev_body_top = max(po, pcl)
            prev_body_bot = min(po, pcl)
            cur_body_top = max(o, cl)
            cur_body_bot = min(o, cl)
            if (
                direction == "bullish"
                and cl > po  # close above prev open
                and cur_body_bot < prev_body_bot
                and cur_body_top > prev_body_top
            ):
                return "engulfing_bull"
            if (
                direction == "bearish"
                and cl < po  # close below prev open
                and cur_body_bot < prev_body_bot
                and cur_body_top > prev_body_top
            ):
                return "engulfing_bear"

        # ── Inside bar (requires previous candle) ────────────────────────────
        if idx > 0:
            prev = candles[idx - 1]
            ph, pl = float(prev["high"]), float(prev["low"])
            if h <= ph and l_ >= pl:
                return "inside_bar"

        return "none"

    @classmethod
    def _describe_candle(
        cls,
        candles: Sequence[Dict],
        idx: int,
        vol_ma20: float,
    ) -> Dict:
        c = candles[idx]
        o = float(c["open"])
        h = float(c["high"])
        l_ = float(c["low"])
        cl = float(c["close"])
        vol = float(c.get("volume", 0))

        rng = h - l_
        body = abs(cl - o)
        body_pct = round(body / rng * 100, 1) if rng > 0 else 0.0
        upper_wick = h - max(o, cl)
        lower_wick = min(o, cl) - l_
        upper_wick_pct = round(upper_wick / rng * 100, 1) if rng > 0 else 0.0
        lower_wick_pct = round(lower_wick / rng * 100, 1) if rng > 0 else 0.0

        direction = cls._candle_direction(o, cl, body_pct)
        close_pos = cls._close_position(o, cl, h, l_)
        pattern = cls._detect_pattern(
            candles, idx, body_pct, lower_wick_pct, upper_wick_pct, direction
        )
        vol_ratio = round(vol / vol_ma20, 2) if vol_ma20 > 0 else None

        return {
            "direction": direction,
            "body_pct": body_pct,
            "upper_wick_pct": upper_wick_pct,
            "lower_wick_pct": lower_wick_pct,
            "close_pos": close_pos,
            "vol_ratio": vol_ratio,
            "pattern": pattern,
        }

    @classmethod
    def extract_candle_structure(
        cls,
        candles_by_timeframe: Dict[str, Sequence[Dict]],
        timeframes: Sequence[str] = ("4h", "1h"),
        n: int = 3,
        vol_lookback: int = 20,
    ) -> Dict[str, List[Dict]]:
        """Return structural description for the last *n* closed candles on each TF.

        Args:
            candles_by_timeframe: dict of timeframe → candle list (sorted ascending).
            timeframes: which timeframes to process.
            n: number of recent candles to describe (default 3).
            vol_lookback: bars to use for volume MA baseline (default 20).

        Returns:
            dict keyed by timeframe, each value is a list of candle dicts
            ordered from oldest to newest ([-n] … [-1]).
        """
        result: Dict[str, List[Dict]] = {}
        for tf in timeframes:
            candles = list(candles_by_timeframe.get(tf, []))
            if len(candles) < max(n + 1, vol_lookback):
                result[tf] = []
                continue

            # Use only closed candles (exclude the live bar at -1 if it may be open)
            closed = candles[:-1]  # last bar may be incomplete — use all but current
            if len(closed) < n:
                result[tf] = []
                continue

            # Volume MA over lookback window ending before the window we describe
            window_start = max(0, len(closed) - n - vol_lookback)
            vol_window = closed[window_start: len(closed) - n]
            if vol_window:
                vol_ma20 = sum(float(c.get("volume", 0)) for c in vol_window) / len(vol_window)
            else:
                vol_ma20 = 0.0

            # Describe the last n closed candles
            target = closed[-n:]
            descriptions = []
            for i, _ in enumerate(target):
                abs_idx = len(closed) - n + i
                desc = cls._describe_candle(closed, abs_idx, vol_ma20)
                descriptions.append(desc)

            result[tf] = descriptions  # index 0 = oldest, -1 = most recent closed

        return result
