from typing import Dict, List


class IndicatorEngine:
    def calculate_ema(self, values: List[float], period: int) -> List[float]:
        if not values:
            return []
        alpha = 2 / (period + 1)
        ema_values = [values[0]]
        for value in values[1:]:
            ema_values.append(alpha * value + (1 - alpha) * ema_values[-1])
        return ema_values

    def calculate_macd(self, closes: List[float]) -> Dict[str, float]:
        if not closes:
            raise ValueError("closes cannot be empty for MACD")
        ema12 = self.calculate_ema(closes, 12)
        ema26 = self.calculate_ema(closes, 26)
        dif = [fast - slow for fast, slow in zip(ema12, ema26)]
        dea = self.calculate_ema(dif, 9)
        hist = [d - e for d, e in zip(dif, dea)]
        return {
            "dif": round(dif[-1], 6),
            "dea": round(dea[-1], 6),
            "hist": round(hist[-1], 6),
        }

    def calculate_kdj(self, highs: List[float], lows: List[float], closes: List[float], period: int = 9) -> Dict[str, float]:
        if not highs or not lows or not closes:
            raise ValueError("highs/lows/closes cannot be empty for KDJ")
        if len(highs) != len(lows) or len(highs) != len(closes):
            raise ValueError("highs/lows/closes lengths must match for KDJ")
        k, d = 50.0, 50.0
        for i in range(len(closes)):
            start = max(0, i - period + 1)
            window_high = max(highs[start : i + 1])
            window_low = min(lows[start : i + 1])
            if window_high == window_low:
                rsv = 50.0
            else:
                rsv = (closes[i] - window_low) / (window_high - window_low) * 100
            k = (2 / 3) * k + (1 / 3) * rsv
            d = (2 / 3) * d + (1 / 3) * k
        j = 3 * k - 2 * d
        return {"k": round(k, 6), "d": round(d, 6), "j": round(j, 6)}

    def calculate_rsi_series(self, closes: List[float], period: int = 14) -> List[float]:
        if not closes:
            raise ValueError("closes cannot be empty for RSI")
        if len(closes) == 1:
            return [50.0]

        rsi_values: List[float] = [50.0]
        gains: List[float] = []
        losses: List[float] = []
        avg_gain = 0.0
        avg_loss = 0.0

        for idx in range(1, len(closes)):
            delta = closes[idx] - closes[idx - 1]
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            gains.append(gain)
            losses.append(loss)

            if idx < period:
                rsi_values.append(50.0)
                continue

            if idx == period:
                avg_gain = sum(gains) / period
                avg_loss = sum(losses) / period
            else:
                avg_gain = ((avg_gain * (period - 1)) + gain) / period
                avg_loss = ((avg_loss * (period - 1)) + loss) / period

            if avg_loss <= 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)

        return rsi_values

    def detect_rsi_divergence(self, closes: List[float], rsi_values: List[float], lookback: int = 30) -> str:
        if len(closes) < 3 or len(rsi_values) != len(closes):
            return "none"

        window = min(lookback, len(closes) - 1)
        previous_prices = closes[-(window + 1) : -1]
        if not previous_prices:
            return "none"

        start_idx = len(closes) - window - 1
        prev_high = max(previous_prices)
        prev_low = min(previous_prices)
        prev_high_idx = start_idx + previous_prices.index(prev_high)
        prev_low_idx = start_idx + previous_prices.index(prev_low)

        current_price = closes[-1]
        current_rsi = rsi_values[-1]
        prev_high_rsi = rsi_values[prev_high_idx]
        prev_low_rsi = rsi_values[prev_low_idx]

        # Require a minimum RSI difference to avoid noisy false positives.
        threshold = 3.0
        if current_price > prev_high and current_rsi < (prev_high_rsi - threshold):
            return "bearish_divergence"
        if current_price < prev_low and current_rsi > (prev_low_rsi + threshold):
            return "bullish_divergence"
        return "none"

    @staticmethod
    def classify_rsi(rsi: float) -> str:
        if rsi >= 70:
            return "overbought"
        if rsi <= 30:
            return "oversold"
        return "neutral"

    def calculate_atr(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14,
    ) -> Dict[str, float]:
        """Average True Range — used for stop-loss sizing and position management."""
        if not highs or len(highs) != len(lows) or len(highs) != len(closes):
            return {"atr": 0.0, "atr_pct": 0.0}

        true_ranges: List[float] = [highs[0] - lows[0]]
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        if len(true_ranges) <= period:
            atr = sum(true_ranges) / len(true_ranges)
        else:
            atr = sum(true_ranges[:period]) / period
            for tr in true_ranges[period:]:
                atr = (atr * (period - 1) + tr) / period

        current_price = closes[-1] if closes else 1.0
        atr_pct = (atr / current_price * 100) if current_price else 0.0
        return {
            "atr": round(atr, 6),
            "atr_pct": round(atr_pct, 6),
            "suggested_sl_distance": round(atr * 1.5, 6),
            "suggested_sl_pct": round(atr_pct * 1.5, 6),
        }

    def calculate_bollinger_bands(
        self,
        closes: List[float],
        period: int = 20,
        num_std: float = 2.0,
    ) -> Dict[str, float]:
        """Bollinger Bands — volatility channel for squeeze/breakout detection."""
        if len(closes) < period:
            mid = sum(closes) / len(closes) if closes else 0.0
            return {"upper": mid, "middle": mid, "lower": mid, "bandwidth": 0.0, "percent_b": 0.5}

        window = closes[-period:]
        mid = sum(window) / period
        variance = sum((x - mid) ** 2 for x in window) / period
        std = variance ** 0.5
        upper = mid + num_std * std
        lower = mid - num_std * std
        bandwidth = ((upper - lower) / mid * 100) if mid else 0.0
        percent_b = ((closes[-1] - lower) / (upper - lower)) if (upper - lower) else 0.5

        return {
            "upper": round(upper, 6),
            "middle": round(mid, 6),
            "lower": round(lower, 6),
            "bandwidth": round(bandwidth, 6),
            "percent_b": round(percent_b, 6),
        }

    def calculate_vwap(self, candles: List[Dict]) -> float:
        """Volume-Weighted Average Price from candle data."""
        total_vp = 0.0
        total_vol = 0.0
        for c in candles:
            typical = (c["high"] + c["low"] + c["close"]) / 3
            vol = c.get("volume", 0.0)
            total_vp += typical * vol
            total_vol += vol
        return round(total_vp / total_vol, 6) if total_vol else 0.0

    def calculate_for_candles(self, candles: List[Dict]) -> Dict:
        if not candles:
            raise ValueError("candles cannot be empty")
        closes = [row["close"] for row in candles]
        highs = [row["high"] for row in candles]
        lows = [row["low"] for row in candles]
        ema7 = self.calculate_ema(closes, 7)[-1]
        ema25 = self.calculate_ema(closes, 25)[-1]
        ema99 = self.calculate_ema(closes, 99)[-1]
        rsi_values = self.calculate_rsi_series(closes, period=14)
        rsi_last = rsi_values[-1]
        rsi_divergence = self.detect_rsi_divergence(closes, rsi_values, lookback=30)
        return {
            "price": round(closes[-1], 6),
            "ema": {
                "7": round(ema7, 6),
                "25": round(ema25, 6),
                "99": round(ema99, 6),
            },
            "macd": self.calculate_macd(closes),
            "kdj": self.calculate_kdj(highs, lows, closes),
            "rsi": {
                "14": round(rsi_last, 6),
                "state": self.classify_rsi(rsi_last),
                "divergence": rsi_divergence,
            },
            "atr": self.calculate_atr(highs, lows, closes),
            "bollinger": self.calculate_bollinger_bands(closes),
            "vwap": self.calculate_vwap(candles),
        }
