"""Transition / change-rate feature extraction.

Computes velocity and regime-shift features from existing data:
- OI delta rates across periods
- basis change velocity
- funding regime classification
- CVD slope acceleration
"""

from typing import Dict, List, Sequence

from ._base import FeatureBase


class TransitionMixin(FeatureBase):

    @staticmethod
    def extract_transition_features(
        open_interest_trend: Dict,
        basis: Dict,
        funding: Dict,
        trade_flow: Dict,
        spot_perp: Dict,
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

        return result
