"""Deployment assessment feature extraction mixin."""

from typing import Dict, List, Sequence

from ._base import FeatureBase


class DeploymentMixin(FeatureBase):
    @staticmethod
    def _make_level(name: str, price: float, source: str, priority: int, current_price: float) -> Dict:
        role = "support" if price < current_price else "resistance" if price > current_price else "pivot"
        return {
            "name": name,
            "price": round(price, 6),
            "source": source,
            "priority": priority,
            "role": role,
        }

    @classmethod
    def _dedupe_levels(cls, levels: Sequence[Dict], current_price: float) -> List[Dict]:
        sorted_levels = sorted(levels, key=lambda row: (row["price"], row["priority"]))
        deduped: List[Dict] = []
        merge_gap = max(current_price * 0.00035, 12.0)
        for level in sorted_levels:
            if not deduped or abs(level["price"] - deduped[-1]["price"]) > merge_gap:
                deduped.append(level)
                continue
            if level["priority"] < deduped[-1]["priority"]:
                deduped[-1] = level
        return deduped

    @classmethod
    def _collect_reference_levels(
        cls,
        current_price: float,
        indicators_by_timeframe: Dict[str, Dict],
        recent_4h_range: Dict,
        volume_profile: Dict,
        liquidation_heatmap: Dict,
    ) -> List[Dict]:
        levels: List[Dict] = []
        high = float(recent_4h_range.get("high", 0.0))
        low = float(recent_4h_range.get("low", 0.0))
        if high > 0:
            levels.append(cls._make_level("4H High", high, "range", 1, current_price))
        if low > 0:
            levels.append(cls._make_level("4H Low", low, "range", 1, current_price))

        poc = float(volume_profile.get("poc_price", 0.0))
        if poc > 0:
            levels.append(cls._make_level("POC", poc, "profile", 1, current_price))
        for idx, price in enumerate(volume_profile.get("hvn_prices", []), start=1):
            if float(price) > 0:
                levels.append(cls._make_level(f"HVN{idx}", float(price), "profile", 2, current_price))
        for idx, price in enumerate(volume_profile.get("lvn_prices", []), start=1):
            if float(price) > 0:
                levels.append(cls._make_level(f"LVN{idx}", float(price), "profile", 3, current_price))

        for anchor in volume_profile.get("anchored_profiles", [])[:2]:
            vwap = float(anchor.get("anchored_vwap", 0.0))
            if vwap > 0:
                label = "AVWAP H" if anchor.get("anchor_type") == "recent_swing_high" else "AVWAP L"
                levels.append(cls._make_level(label, vwap, "anchored_vwap", 2, current_price))

        for tf in ("1h", "4h"):
            ema = indicators_by_timeframe.get(tf, {}).get("ema", {})
            ema25 = float(ema.get("25", 0.0))
            ema99 = float(ema.get("99", 0.0))
            if ema25 > 0:
                levels.append(cls._make_level(f"{tf} EMA25", ema25, "ema", 2, current_price))
            if ema99 > 0:
                levels.append(cls._make_level(f"{tf} EMA99", ema99, "ema", 3, current_price))

        for zone in liquidation_heatmap.get("zones", []):
            name = str(zone.get("name", ""))
            zone_low = float(zone.get("zone_low", 0.0))
            zone_high = float(zone.get("zone_high", 0.0))
            if zone_low <= 0 or zone_high <= 0:
                continue
            center = (zone_low + zone_high) / 2
            if "recent_4h" in name:
                levels.append(cls._make_level(name.replace("_", " "), center, "liquidity", 1, current_price))
            elif abs(center - current_price) / max(current_price, 1.0) <= 0.02:
                levels.append(cls._make_level(name.replace("_", " "), center, "liquidity", 3, current_price))

        return cls._dedupe_levels(levels, current_price)

    @staticmethod
    def _structure_score(trend_4h: str, trend_1h: str, trend_15m: str) -> float:
        if trend_4h == trend_1h == trend_15m and trend_4h in ("bullish", "bearish"):
            return 92.0
        if trend_4h == trend_1h and trend_4h in ("bullish", "bearish"):
            return 76.0
        if trend_4h in ("bullish", "bearish") and trend_1h == "neutral":
            return 62.0
        if trend_4h == "neutral" and trend_1h == trend_15m:
            return 55.0
        return 38.0

    @staticmethod
    def _score_from_distance(distance_bps: float) -> float:
        if distance_bps <= 15:
            return 92.0
        if distance_bps <= 35:
            return 76.0
        if distance_bps <= 60:
            return 58.0
        return 34.0

    @staticmethod
    def _score_from_spoof_risk(spoofing_risk: str) -> float:
        if spoofing_risk == "low":
            return 85.0
        if spoofing_risk == "medium":
            return 60.0
        if spoofing_risk == "high":
            return 30.0
        return 50.0

    @staticmethod
    def _flow_score(side: str, trade_flow: Dict, open_interest_trend: Dict) -> float:
        window = trade_flow.get("windows", {}).get("5m", {})
        delta_quote = float(window.get("delta_quote", 0.0))
        large_direction = str(window.get("large_trade_direction") or trade_flow.get("large_trade_direction") or "balanced")
        oi_state = str(open_interest_trend.get("latest_state", "unknown"))

        if side == "long":
            score = 40.0
            if delta_quote > 0:
                score += 18.0
            if large_direction == "buy_dominant":
                score += 18.0
            if oi_state in ("price_up_oi_up", "price_up_oi_down"):
                score += 16.0
            return min(score, 92.0)

        if side == "short":
            score = 40.0
            if delta_quote < 0:
                score += 18.0
            if large_direction == "sell_dominant":
                score += 18.0
            if oi_state in ("price_down_oi_up", "price_down_oi_down"):
                score += 16.0
            return min(score, 92.0)

        return 48.0

    @classmethod
    def extract_deployment_assessment(
        cls,
        current_price: float,
        indicators_by_timeframe: Dict[str, Dict],
        recent_4h_range: Dict,
        volume_profile: Dict,
        liquidation_heatmap: Dict,
        open_interest_trend: Dict,
        orderbook_dynamics: Dict,
        trade_flow: Dict,
        session_context: Dict,
    ) -> Dict:
        trend_4h = indicators_by_timeframe.get("4h", {}).get("features", {}).get("trend", "unknown")
        trend_1h = indicators_by_timeframe.get("1h", {}).get("features", {}).get("trend", "unknown")
        trend_15m = indicators_by_timeframe.get("15m", {}).get("features", {}).get("trend", "unknown")
        momentum_15m = indicators_by_timeframe.get("15m", {}).get("features", {}).get("momentum", "momentum_neutral")
        oi_state = str(open_interest_trend.get("latest_state", "unknown"))

        if trend_4h == "bullish" and trend_1h in ("bullish", "neutral"):
            primary_bias = "long"
        elif trend_4h == "bearish" and trend_1h in ("bearish", "neutral"):
            primary_bias = "short"
        else:
            primary_bias = "neutral"

        range_high = float(recent_4h_range.get("high", 0.0))
        range_low = float(recent_4h_range.get("low", 0.0))
        range_mid = (range_high + range_low) / 2 if range_high and range_low else current_price
        near_range_edge = abs(current_price - range_high) <= current_price * 0.002 or abs(current_price - range_low) <= current_price * 0.002
        if primary_bias == "long" and trend_15m == "bearish":
            transition_state = "bullish_pullback"
        elif primary_bias == "short" and trend_15m == "bullish":
            transition_state = "bearish_pullback"
        elif trend_15m == trend_1h == trend_4h and primary_bias != "neutral":
            transition_state = "trend_continuation"
        elif near_range_edge:
            transition_state = "breakout_watch"
        elif abs(current_price - range_mid) <= max(current_price * 0.0012, (range_high - range_low) * 0.15 if range_high > range_low else 0.0):
            transition_state = "range_rotation"
        else:
            transition_state = "mixed_transition"

        reference_levels = cls._collect_reference_levels(
            current_price=current_price,
            indicators_by_timeframe=indicators_by_timeframe,
            recent_4h_range=recent_4h_range,
            volume_profile=volume_profile,
            liquidation_heatmap=liquidation_heatmap,
        )
        support_levels = sorted([level for level in reference_levels if level["price"] < current_price], key=lambda row: row["price"], reverse=True)
        resistance_levels = sorted([level for level in reference_levels if level["price"] > current_price], key=lambda row: row["price"])
        zone_width = max(current_price * 0.0008, float(recent_4h_range.get("range_abs", 0.0)) * 0.04, 18.0)
        plan_zones: Dict[str, Dict] = {}
        distance_to_entry_bps = 999.0

        if primary_bias == "long" and support_levels and resistance_levels:
            entry_level = support_levels[0]["price"]
            invalidation_level = support_levels[1]["price"] if len(support_levels) > 1 else min(range_low or entry_level - zone_width, entry_level - zone_width)
            tp_level = resistance_levels[0]["price"]
            plan_zones = {
                "entry": {
                    "side": "long",
                    "zone_low": round(entry_level - zone_width * 0.35, 6),
                    "zone_high": round(entry_level + zone_width * 0.35, 6),
                },
                "invalidation": {
                    "side": "long",
                    "zone_low": round(invalidation_level - zone_width * 0.3, 6),
                    "zone_high": round(invalidation_level + zone_width * 0.15, 6),
                },
                "take_profit": {
                    "side": "long",
                    "zone_low": round(tp_level - zone_width * 0.25, 6),
                    "zone_high": round(tp_level + zone_width * 0.45, 6),
                },
            }
            distance_to_entry_bps = abs(current_price - entry_level) / max(current_price, 1.0) * 10000
        elif primary_bias == "short" and support_levels and resistance_levels:
            entry_level = resistance_levels[0]["price"]
            invalidation_level = resistance_levels[1]["price"] if len(resistance_levels) > 1 else max(range_high or entry_level + zone_width, entry_level + zone_width)
            tp_level = support_levels[0]["price"]
            plan_zones = {
                "entry": {
                    "side": "short",
                    "zone_low": round(entry_level - zone_width * 0.35, 6),
                    "zone_high": round(entry_level + zone_width * 0.35, 6),
                },
                "invalidation": {
                    "side": "short",
                    "zone_low": round(invalidation_level - zone_width * 0.15, 6),
                    "zone_high": round(invalidation_level + zone_width * 0.3, 6),
                },
                "take_profit": {
                    "side": "short",
                    "zone_low": round(tp_level - zone_width * 0.45, 6),
                    "zone_high": round(tp_level + zone_width * 0.25, 6),
                },
            }
            distance_to_entry_bps = abs(current_price - entry_level) / max(current_price, 1.0) * 10000

        structure_score = cls._structure_score(trend_4h, trend_1h, trend_15m)
        distance_score = cls._score_from_distance(distance_to_entry_bps) if plan_zones else 42.0
        spoof_score = cls._score_from_spoof_risk(str(orderbook_dynamics.get("spoofing_risk", "unknown")))
        flow_score = cls._flow_score(primary_bias, trade_flow, open_interest_trend)
        deployment_score_value = round(
            structure_score * 0.32
            + distance_score * 0.24
            + spoof_score * 0.14
            + flow_score * 0.2
            + (72.0 if session_context.get("funding_countdown_seconds", 0.0) > 1800 else 54.0) * 0.1,
            2,
        )
        if primary_bias == "neutral":
            deployment_score_value = min(deployment_score_value, 54.0)

        if deployment_score_value >= 76:
            deployment_score = "high"
        elif deployment_score_value >= 58:
            deployment_score = "medium"
        else:
            deployment_score = "low"

        state_tags = [
            transition_state,
            str(open_interest_trend.get("latest_interpretation", "unknown")),
            str(trade_flow.get("windows", {}).get("5m", {}).get("large_trade_direction", "unknown")),
            f"spoof:{orderbook_dynamics.get('spoofing_risk', 'unknown')}",
            f"deploy:{deployment_score}",
        ]

        liquidity_zones = []
        for zone in liquidation_heatmap.get("zones", []):
            zone_low = float(zone.get("zone_low", 0.0))
            zone_high = float(zone.get("zone_high", 0.0))
            if zone_low <= 0 or zone_high <= 0:
                continue
            if abs(((zone_low + zone_high) / 2) - current_price) / max(current_price, 1.0) <= 0.02 or "recent_4h" in str(zone.get("name", "")):
                liquidity_zones.append(
                    {
                        "name": str(zone.get("name", "zone")),
                        "zone_low": round(zone_low, 6),
                        "zone_high": round(zone_high, 6),
                        "estimated_pressure": zone.get("estimated_pressure"),
                    }
                )

        return {
            "primary_bias": primary_bias,
            "transition_state": transition_state,
            "execution_context": {
                "oi_state": oi_state,
                "oi_interpretation": open_interest_trend.get("latest_interpretation", "unknown"),
                "micro_flow_state": trade_flow.get("windows", {}).get("5m", {}).get("large_trade_direction", "unknown"),
                "momentum_15m": momentum_15m,
            },
            "deployment_score": deployment_score,
            "deployment_score_value": deployment_score_value,
            "distance_to_entry_bps": round(distance_to_entry_bps, 6) if plan_zones else None,
            "component_scores": {
                "structure_consistency": round(structure_score, 2),
                "distance_to_entry": round(distance_score, 2),
                "spoof_risk": round(spoof_score, 2),
                "flow_alignment": round(flow_score, 2),
            },
            "reference_levels": reference_levels,
            "liquidity_zones": liquidity_zones,
            "plan_zones": plan_zones,
            "state_tags": state_tags,
        }
