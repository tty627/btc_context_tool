"""Spot vs perpetual comparison: CVD, basis, and flow interpretation."""

from typing import Dict, List, Sequence

from ._base import FeatureBase


class SpotPerpMixin(FeatureBase):
    @classmethod
    def extract_spot_perp_features(
        cls,
        spot_trades: Sequence[Dict],
        spot_ticker: Dict,
        perp_price: float,
        perp_funding_rate: float = 0.0,
    ) -> Dict:
        """Compare spot and perpetual market flow to distinguish genuine spot demand
        from self-reinforcing perp squeezes.

        Returns basis, spot CVD, per-window buy/sell breakdown, and a plain-language
        interpretation of which market is leading.
        """
        if not spot_ticker.get("available"):
            return {"available": False, "reason": "spot_ticker_unavailable"}

        spot_price = float(spot_ticker.get("last_price", 0))
        if spot_price <= 0:
            return {"available": False, "reason": "invalid_spot_price"}

        rows = sorted(list(spot_trades), key=lambda x: x.get("timestamp", 0))
        if not rows:
            return {"available": False, "reason": "no_spot_trades"}

        buy_quote = 0.0
        sell_quote = 0.0
        cvd_qty = 0.0
        for row in rows:
            qty = float(row.get("qty", 0))
            quote = float(row.get("quote_qty", 0))
            if row.get("aggressor_side") == "buy":
                buy_quote += quote
                cvd_qty += qty
            else:
                sell_quote += quote
                cvd_qty -= qty

        delta_quote = buy_quote - sell_quote

        end_ts = int(rows[-1]["timestamp"]) if rows else 0

        def _window_sum(minutes: int) -> Dict:
            cutoff = end_ts - minutes * 60 * 1000
            w = [r for r in rows if int(r.get("timestamp", 0)) >= cutoff]
            b, s = 0.0, 0.0
            for r in w:
                q = float(r.get("quote_qty", 0))
                if r.get("aggressor_side") == "buy":
                    b += q
                else:
                    s += q
            return {"buy_quote": round(b, 2), "sell_quote": round(s, 2), "delta_quote": round(b - s, 2)}

        basis_abs = perp_price - spot_price
        basis_bps = basis_abs / spot_price * 10000 if spot_price > 0 else 0.0

        if basis_bps > 8:
            basis_signal = "perp_premium_high"
        elif basis_bps > 3:
            basis_signal = "perp_premium_mild"
        elif basis_bps < -8:
            basis_signal = "perp_discount_high"
        elif basis_bps < -3:
            basis_signal = "perp_discount_mild"
        else:
            basis_signal = "near_parity"

        cvd_state = "positive" if delta_quote > 0 else "negative" if delta_quote < 0 else "flat"

        if cvd_state == "positive" and basis_bps > 3:
            interpretation = "spot_buying_drives_perp_premium"
        elif cvd_state == "positive" and basis_bps < -3:
            interpretation = "spot_buying_perp_lagging_or_suppressed"
        elif cvd_state == "negative" and basis_bps < -3:
            interpretation = "spot_selling_perp_discount_confirms"
        elif cvd_state == "negative" and basis_bps > 8:
            interpretation = "perp_self_driven_squeeze_no_spot_support"
        elif cvd_state == "flat":
            interpretation = "neutral_mixed"
        else:
            interpretation = "mixed"

        spot_vol_24h = float(spot_ticker.get("quote_volume", 0))

        return {
            "available": True,
            "spot_price": round(spot_price, 2),
            "perp_price": round(perp_price, 2),
            "basis_abs": round(basis_abs, 2),
            "basis_bps": round(basis_bps, 2),
            "basis_signal": basis_signal,
            "spot_cvd_qty": round(cvd_qty, 4),
            "spot_delta_quote": round(delta_quote, 2),
            "spot_buy_quote_total": round(buy_quote, 2),
            "spot_sell_quote_total": round(sell_quote, 2),
            "spot_5m": _window_sum(5),
            "spot_15m": _window_sum(15),
            "spot_vol_24h_quote": round(spot_vol_24h, 2),
            "cvd_state": cvd_state,
            "interpretation": interpretation,
        }
