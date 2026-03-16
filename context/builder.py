from datetime import datetime, timezone
from typing import Dict


class MarketContextBuilder:
    def build(
        self,
        symbol: str,
        price: float,
        indicators_by_timeframe: Dict[str, Dict],
        account_positions: Dict,
        orderbook_features: Dict,
        orderbook_dynamics: Dict,
        open_interest: float,
        open_interest_trend: Dict,
        long_short_ratio: Dict,
        funding: Dict,
        basis: Dict,
        cross_exchange_funding: Dict,
        funding_spread: Dict,
        options_iv: Dict,
        stats_24h: Dict,
        recent_4h_range: Dict,
        volume_change: Dict,
        volume_profile: Dict,
        trade_flow: Dict,
        liquidation_heatmap: Dict,
        external_drivers: Dict | None = None,
        chart_files: Dict[str, str] | None = None,
    ) -> Dict:
        market_structure = {
            timeframe: metrics.get("features", {}).get("trend", "unknown")
            for timeframe, metrics in indicators_by_timeframe.items()
        }
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "price": round(price, 6),
            "market_structure": market_structure,
            "timeframes": indicators_by_timeframe,
            "account_positions": account_positions,
            "orderbook": orderbook_features,
            "orderbook_dynamics": orderbook_dynamics,
            "open_interest": round(open_interest, 6),
            "open_interest_trend": open_interest_trend,
            "long_short_ratio": long_short_ratio,
            "funding": funding,
            "basis": basis,
            "cross_exchange_funding": cross_exchange_funding,
            "funding_spread": funding_spread,
            "options_iv": options_iv,
            "stats_24h": stats_24h,
            "recent_4h_range": recent_4h_range,
            "volume_change": volume_change,
            "volume_profile": volume_profile,
            "trade_flow": trade_flow,
            "liquidation_heatmap": liquidation_heatmap,
            "external_drivers": external_drivers or {},
            "chart_files": chart_files or {},
        }
