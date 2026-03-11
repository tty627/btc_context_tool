"""Liquidation feature extraction mixin."""

from typing import Dict, List, Sequence

from ._base import FeatureBase


class LiquidationMixin(FeatureBase):
    @staticmethod
    def extract_liquidation_heatmap(
        current_price: float,
        recent_high: float,
        recent_low: float,
        force_orders: Sequence[Dict],
    ) -> Dict:
        rows = list(force_orders)
        if rows:
            bands: Dict[str, Dict] = {}
            band_width = max(current_price * 0.002, 10.0)
            for row in rows:
                price = row["price"]
                if price <= 0:
                    continue
                center = round(round(price / band_width) * band_width, 2)
                key = f"{center:.2f}"
                if key not in bands:
                    bands[key] = {
                        "zone_low": round(center - band_width / 2, 2),
                        "zone_high": round(center + band_width / 2, 2),
                        "buy_force_quote": 0.0,
                        "sell_force_quote": 0.0,
                    }
                if row["side"] == "buy":
                    bands[key]["buy_force_quote"] += row["quote_qty"]
                elif row["side"] == "sell":
                    bands[key]["sell_force_quote"] += row["quote_qty"]

            ranked = []
            for value in bands.values():
                total_quote = value["buy_force_quote"] + value["sell_force_quote"]
                value["total_force_quote"] = total_quote
                ranked.append(value)
            ranked.sort(key=lambda x: x["total_force_quote"], reverse=True)
            top = ranked[:6]
            max_quote = max((float(row["total_force_quote"]) for row in top), default=0.0)
            for row in top:
                row["buy_force_quote"] = round(row["buy_force_quote"], 6)
                row["sell_force_quote"] = round(row["sell_force_quote"], 6)
                row["total_force_quote"] = round(row["total_force_quote"], 6)
                ratio = 0.0 if max_quote == 0 else float(row["total_force_quote"]) / max_quote
                row["confidence"] = "high" if ratio >= 0.75 else "medium"
                row["assumption"] = "clustered_force_orders"
            return {
                "source": "force_orders",
                "confidence": "medium_high",
                "zones": top,
            }

        levels = [10, 25, 50]
        zones: List[Dict] = []
        for leverage in levels:
            long_liq = current_price * (1 - 1 / leverage)
            short_liq = current_price * (1 + 1 / leverage)
            width = current_price * 0.003
            zones.append(
                {
                    "name": f"long_{leverage}x",
                    "zone_low": round(long_liq - width, 2),
                    "zone_high": round(long_liq + width, 2),
                    "estimated_pressure": "long_liquidation",
                    "confidence": "low",
                    "assumption": "static_leverage_band",
                }
            )
            zones.append(
                {
                    "name": f"short_{leverage}x",
                    "zone_low": round(short_liq - width, 2),
                    "zone_high": round(short_liq + width, 2),
                    "estimated_pressure": "short_liquidation",
                    "confidence": "low",
                    "assumption": "static_leverage_band",
                }
            )

        if recent_low > 0:
            zones.append(
                {
                    "name": "recent_4h_low_sweep",
                    "zone_low": round(recent_low * 0.99, 2),
                    "zone_high": round(recent_low * 1.002, 2),
                    "estimated_pressure": "long_liquidation",
                    "confidence": "medium",
                    "assumption": "recent_range_liquidity_below",
                }
            )
        if recent_high > 0:
            zones.append(
                {
                    "name": "recent_4h_high_sweep",
                    "zone_low": round(recent_high * 0.998, 2),
                    "zone_high": round(recent_high * 1.01, 2),
                    "estimated_pressure": "short_liquidation",
                    "confidence": "medium",
                    "assumption": "recent_range_liquidity_above",
                }
            )
        return {
            "source": "model_estimate",
            "confidence": "low",
            "model_assumptions": [
                "static_leverage_bands",
                "recent_4h_range_sweep_levels",
            ],
            "zones": zones,
        }
