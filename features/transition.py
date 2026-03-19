"""Transition / change-rate feature extraction.

Computes velocity and regime-shift features from existing data:
- OI delta rates across periods
- basis change velocity
- funding regime classification
- CVD slope acceleration
- CVD divergence detection (price vs CVD new-high/low comparison)
"""

from typing import Dict, List, Optional, Sequence

from ._base import FeatureBase


class TransitionMixin(FeatureBase):

    @staticmethod
    def _detect_cvd_divergence(
        candles_5m: Sequence[Dict],
        lookback_bars: int = 12,
    ) -> Dict:
        """Detect hidden bullish/bearish CVD divergence using 5m kline taker data.

        Compares the most recent *lookback_bars* 5m candles:
        - price creates a new low but CVD does NOT → hidden bullish divergence
        - price creates a new high but CVD does NOT → hidden bearish divergence

        Strength is proportional to the gap between price and CVD extremes.
        Returns a dict with keys: type, strength, window_bars.
        """
        candles = [
            c for c in list(candles_5m)
            if float(c.get("taker_buy_base", 0)) > 0 or float(c.get("volume", 0)) > 0
        ]
        if len(candles) < lookback_bars + 2:
            return {"type": "no_divergence", "strength": "none", "window_bars": lookback_bars}

        window = candles[-lookback_bars:]

        # Build per-bar cumulative delta using taker buy fields
        deltas: List[float] = []
        for c in window:
            buy = float(c.get("taker_buy_base", 0))
            total = float(c.get("volume", 0))
            deltas.append(buy - (total - buy))

        cvd: List[float] = []
        running = 0.0
        for d in deltas:
            running += d
            cvd.append(running)

        closes = [float(c["close"]) for c in window]

        price_min_idx = closes.index(min(closes))
        price_max_idx = closes.index(max(closes))
        cvd_min_idx = cvd.index(min(cvd))
        cvd_max_idx = cvd.index(max(cvd))

        last_price = closes[-1]
        last_cvd = cvd[-1]

        div_type = "no_divergence"
        strength = "none"

        # Hidden bullish: price at/near new low but CVD not at new low
        if price_min_idx == len(closes) - 1:
            # Current close is the lowest in window
            if cvd_min_idx != len(cvd) - 1:
                # CVD did not confirm new low
                price_drop_pct = abs(closes[price_min_idx] - closes[0]) / max(abs(closes[0]), 1) * 100
                cvd_gap = abs(cvd[cvd_min_idx] - last_cvd) / (max(abs(min(cvd)), abs(max(cvd)), 0.001))
                div_type = "bullish_hidden_div"
                if cvd_gap > 0.3 and price_drop_pct > 0.3:
                    strength = "strong"
                elif cvd_gap > 0.1:
                    strength = "moderate"
                else:
                    strength = "weak"

        # Hidden bearish: price at/near new high but CVD not at new high
        elif price_max_idx == len(closes) - 1:
            if cvd_max_idx != len(cvd) - 1:
                price_rise_pct = abs(closes[price_max_idx] - closes[0]) / max(abs(closes[0]), 1) * 100
                cvd_gap = abs(cvd[cvd_max_idx] - last_cvd) / (max(abs(min(cvd)), abs(max(cvd)), 0.001))
                div_type = "bearish_hidden_div"
                if cvd_gap > 0.3 and price_rise_pct > 0.3:
                    strength = "strong"
                elif cvd_gap > 0.1:
                    strength = "moderate"
                else:
                    strength = "weak"

        return {
            "type": div_type,
            "strength": strength,
            "window_bars": lookback_bars,
        }

    @classmethod
    def extract_transition_features(
        cls,
        open_interest_trend: Dict,
        basis: Dict,
        funding: Dict,
        trade_flow: Dict,
        spot_perp: Dict,
        candles_5m: Optional[Sequence[Dict]] = None,
    ) -> Dict:
        result: Dict = {}

        # ── OI delta rates across periods ─────────────────────────────────
        oi_periods = open_interest_trend.get("periods", {})
        oi_deltas: Dict[str, Dict] = {}
        for period in ("5m", "15m", "1h"):
            p = oi_periods.get(period, {})
            series = p.get("series", [])
            delta_pct = float(p.get("delta_pct", 0))

            velocity = 0.0
            acceleration = 0.0
            if len(series) >= 3:
                oi_vals = [float(s.get("open_interest", 0)) for s in series]
                changes = [oi_vals[i] - oi_vals[i - 1] for i in range(1, len(oi_vals))]
                velocity = changes[-1] if changes else 0.0
                if len(changes) >= 2:
                    acceleration = changes[-1] - changes[-2]

            oi_deltas[period] = {
                "delta_pct": round(delta_pct, 4),
                "velocity": round(velocity, 2),
                "acceleration": round(acceleration, 2),
            }
        result["oi_rates"] = oi_deltas

        # ── Basis change velocity ─────────────────────────────────────────
        sp = spot_perp or {}
        basis_bps = float(sp.get("basis_bps", 0))
        basis_abs = float(basis.get("basis_abs", 0))

        windows = trade_flow.get("windows", {})
        w5 = windows.get("5m", {})
        w15 = windows.get("15m", {})
        perp_delta_5m = float(w5.get("delta_quote", 0))
        perp_delta_15m = float(w15.get("delta_quote", 0))

        if perp_delta_5m > 0 and basis_bps < -3:
            basis_regime = "perp_buying_but_discount"
        elif perp_delta_5m < 0 and basis_bps > 3:
            basis_regime = "perp_selling_but_premium"
        elif abs(basis_bps) > 8:
            basis_regime = "extreme_premium" if basis_bps > 0 else "extreme_discount"
        else:
            basis_regime = "normal"

        result["basis_dynamics"] = {
            "basis_bps": round(basis_bps, 2),
            "basis_abs": round(basis_abs, 2),
            "regime": basis_regime,
        }

        # ── Funding regime ────────────────────────────────────────────────
        fr = float(funding.get("funding_rate", 0))
        fr_pct = fr * 100

        if fr_pct > 0.01:
            funding_regime = "longs_pay"
        elif fr_pct < -0.01:
            funding_regime = "shorts_pay"
        else:
            funding_regime = "neutral"

        if abs(fr_pct) > 0.05:
            funding_intensity = "elevated"
        elif abs(fr_pct) > 0.1:
            funding_intensity = "extreme"
        else:
            funding_intensity = "normal"

        result["funding_dynamics"] = {
            "rate_pct": round(fr_pct, 4),
            "regime": funding_regime,
            "intensity": funding_intensity,
        }

        # ── CVD slope change ──────────────────────────────────────────────
        cvd_path = trade_flow.get("cvd_path", [])
        cvd_slope_5m = float(w5.get("cvd_slope_qty_per_minute", 0))
        cvd_slope_15m = float(w15.get("cvd_slope_qty_per_minute", 0))

        if cvd_slope_5m > 0 and cvd_slope_15m > 0:
            cvd_momentum = "accelerating_buy"
        elif cvd_slope_5m < 0 and cvd_slope_15m < 0:
            cvd_momentum = "accelerating_sell"
        elif cvd_slope_5m * cvd_slope_15m < 0:
            cvd_momentum = "diverging"
        else:
            cvd_momentum = "flat"

        result["cvd_dynamics"] = {
            "slope_5m": round(cvd_slope_5m, 4),
            "slope_15m": round(cvd_slope_15m, 4),
            "momentum": cvd_momentum,
        }

        # ── CVD divergence detection ──────────────────────────────────────────
        if candles_5m:
            div_1h = cls._detect_cvd_divergence(candles_5m, lookback_bars=12)   # ~1h window
            div_15m = cls._detect_cvd_divergence(candles_5m, lookback_bars=3)   # ~15m window
            result["cvd_divergence"] = {
                "1h_window": div_1h,
                "15m_window": div_15m,
            }
        else:
            result["cvd_divergence"] = None

        return result
