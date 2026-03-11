"""Composite signal scoring system: aggregates multiple dimensions into a 0-100 bull/bear score."""

from typing import Dict

from ._base import FeatureBase


class SignalScoreMixin(FeatureBase):
    @classmethod
    def calculate_signal_score(
        cls,
        indicators_by_timeframe: Dict,
        orderbook_features: Dict,
        orderbook_dynamics: Dict,
        open_interest_trend: Dict,
        long_short_ratio: Dict,
        trade_flow: Dict,
        funding: Dict,
        basis: Dict,
    ) -> Dict:
        """Generate a composite bull/bear score from 0 (max bearish) to 100 (max bullish).

        Components:
        - Trend alignment (multi-TF EMA structure)
        - Momentum (MACD, RSI)
        - Orderbook pressure (imbalance, wall direction)
        - Trade flow (CVD, large trade direction)
        - Derivatives sentiment (OI, long/short ratio, funding)
        """
        components: Dict[str, Dict] = {}
        weights = {
            "trend": 0.25,
            "momentum": 0.20,
            "orderbook": 0.15,
            "trade_flow": 0.20,
            "derivatives": 0.20,
        }

        # ── Trend (multi-timeframe EMA alignment) ───────────────────────
        trend_score = 50.0
        trend_signals = []
        for tf in ("4h", "1h", "15m"):
            features = indicators_by_timeframe.get(tf, {}).get("features", {})
            trend = features.get("trend", "neutral")
            if trend == "bullish":
                trend_signals.append(1)
            elif trend == "bearish":
                trend_signals.append(-1)
            else:
                trend_signals.append(0)

        if trend_signals:
            tf_weights = [0.45, 0.35, 0.20]
            weighted = sum(s * w for s, w in zip(trend_signals, tf_weights[:len(trend_signals)]))
            trend_score = 50 + weighted * 50
        components["trend"] = {"score": round(trend_score, 1), "signals": trend_signals}

        # ── Momentum (MACD hist + RSI) ──────────────────────────────────
        momentum_score = 50.0
        primary_tf = "15m" if "15m" in indicators_by_timeframe else next(iter(indicators_by_timeframe), "")
        if primary_tf:
            metrics = indicators_by_timeframe[primary_tf]
            macd = metrics.get("macd", {})
            rsi = metrics.get("rsi", {})
            hist = float(macd.get("hist", 0))
            rsi_val = float(rsi.get("14", 50))

            macd_signal = 0.0
            if hist > 0:
                macd_signal = min(hist / max(abs(float(macd.get("dea", 1))), 0.01) * 50, 50)
            elif hist < 0:
                macd_signal = max(hist / max(abs(float(macd.get("dea", 1))), 0.01) * 50, -50)

            rsi_signal = (rsi_val - 50) * 0.8

            momentum_score = 50 + (macd_signal * 0.6 + rsi_signal * 0.4)
            momentum_score = max(0, min(100, momentum_score))

        components["momentum"] = {
            "score": round(momentum_score, 1),
            "macd_hist": float(macd.get("hist", 0)) if primary_tf else 0,
            "rsi": float(rsi.get("14", 50)) if primary_tf else 50,
        }

        # ── Orderbook pressure ──────────────────────────────────────────
        ob_score = 50.0
        imbalance = float(orderbook_features.get("imbalance", 0))
        ob_score = 50 + imbalance * 50
        spoof = str(orderbook_dynamics.get("spoofing_risk", "unknown"))
        if spoof == "high":
            ob_score = 50 + (ob_score - 50) * 0.3
        ob_score = max(0, min(100, ob_score))
        components["orderbook"] = {
            "score": round(ob_score, 1),
            "imbalance": round(imbalance, 4),
            "spoofing_risk": spoof,
        }

        # ── Trade flow (CVD + large trades) ─────────────────────────────
        tf_score = 50.0
        window_5m = trade_flow.get("windows", {}).get("5m", {})
        delta_quote = float(window_5m.get("delta_quote", 0))
        buy_quote = float(window_5m.get("buy_quote", 0))
        sell_quote = float(window_5m.get("sell_quote", 0))
        total = buy_quote + sell_quote
        if total > 0:
            flow_ratio = delta_quote / total
            tf_score = 50 + flow_ratio * 100
        large_dir = str(window_5m.get("large_trade_direction") or trade_flow.get("large_trade_direction", "balanced"))
        if large_dir == "buy_dominant":
            tf_score = min(100, tf_score + 8)
        elif large_dir == "sell_dominant":
            tf_score = max(0, tf_score - 8)
        tf_score = max(0, min(100, tf_score))
        components["trade_flow"] = {
            "score": round(tf_score, 1),
            "delta_quote": round(delta_quote, 2),
            "large_direction": large_dir,
        }

        # ── Derivatives sentiment ───────────────────────────────────────
        deriv_score = 50.0
        signals = []

        funding_rate = float(funding.get("funding_rate", 0))
        if funding_rate > 0.0003:
            signals.append(-1)
        elif funding_rate < -0.0003:
            signals.append(1)
        else:
            signals.append(0)

        oi_state = str(open_interest_trend.get("latest_state", "unknown"))
        if oi_state == "price_up_oi_up":
            signals.append(1)
        elif oi_state == "price_down_oi_up":
            signals.append(-1)
        elif oi_state == "price_up_oi_down":
            signals.append(0.5)
        elif oi_state == "price_down_oi_down":
            signals.append(-0.5)
        else:
            signals.append(0)

        crowding = str(long_short_ratio.get("overall_crowding", "balanced"))
        if crowding == "long_crowded":
            signals.append(-0.5)
        elif crowding == "short_crowded":
            signals.append(0.5)
        else:
            signals.append(0)

        basis_struct = str(basis.get("structure", "flat"))
        if basis_struct == "contango":
            signals.append(0.3)
        elif basis_struct == "backwardation":
            signals.append(-0.3)
        else:
            signals.append(0)

        if signals:
            avg_signal = sum(signals) / len(signals)
            deriv_score = 50 + avg_signal * 40
            deriv_score = max(0, min(100, deriv_score))

        components["derivatives"] = {
            "score": round(deriv_score, 1),
            "funding_rate": funding_rate,
            "oi_state": oi_state,
            "crowding": crowding,
        }

        # ── Composite ───────────────────────────────────────────────────
        composite = sum(
            components[key]["score"] * weights[key]
            for key in weights
            if key in components
        )
        composite = max(0, min(100, composite))

        if composite >= 65:
            bias = "bullish"
        elif composite <= 35:
            bias = "bearish"
        else:
            bias = "neutral"

        if composite >= 75:
            strength = "strong_bullish"
        elif composite >= 60:
            strength = "moderate_bullish"
        elif composite >= 40:
            strength = "neutral"
        elif composite >= 25:
            strength = "moderate_bearish"
        else:
            strength = "strong_bearish"

        return {
            "composite_score": round(composite, 1),
            "bias": bias,
            "strength": strength,
            "components": components,
            "weights": weights,
        }
