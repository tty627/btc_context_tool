"""Trade flow feature extraction: CVD, delta, large trades, aggressor layers."""

from typing import Any, Dict, List, Sequence, Tuple

from ._base import FeatureBase


class TradeFlowMixin(FeatureBase):
    @staticmethod
    def _summarize_trade_rows(rows: Sequence[Dict], large_trade_threshold: float) -> Dict:
        trade_rows = list(rows)
        buy_quote = 0.0
        sell_quote = 0.0
        delta_qty = 0.0
        delta_quote = 0.0
        cvd_qty = 0.0
        large_buy_quote = 0.0
        large_sell_quote = 0.0

        for row in trade_rows:
            qty = float(row.get("qty", 0.0))
            quote_qty = float(row.get("quote_qty", 0.0))
            if row.get("aggressor_side") == "buy":
                buy_quote += quote_qty
                delta_qty += qty
                delta_quote += quote_qty
                cvd_qty += qty
                if quote_qty >= large_trade_threshold:
                    large_buy_quote += quote_qty
            else:
                sell_quote += quote_qty
                delta_qty -= qty
                delta_quote -= quote_qty
                cvd_qty -= qty
                if quote_qty >= large_trade_threshold:
                    large_sell_quote += quote_qty

        large_trade_direction = "balanced"
        if large_buy_quote > large_sell_quote * 1.1:
            large_trade_direction = "buy_dominant"
        elif large_sell_quote > large_buy_quote * 1.1:
            large_trade_direction = "sell_dominant"

        return {
            "trade_count": len(trade_rows),
            "buy_quote": round(buy_quote, 6),
            "sell_quote": round(sell_quote, 6),
            "delta_qty": round(delta_qty, 6),
            "delta_quote": round(delta_quote, 6),
            "cvd_qty": round(cvd_qty, 6),
            "large_buy_quote": round(large_buy_quote, 6),
            "large_sell_quote": round(large_sell_quote, 6),
            "large_trade_direction": large_trade_direction,
        }

    @classmethod
    def extract_trade_flow_features(cls, trades: Sequence[Dict], large_trade_quantile: float = 0.9) -> Dict:
        rows = sorted(list(trades), key=lambda row: int(row.get("timestamp", 0)))
        if not rows:
            return {
                "trade_count": 0,
                "buy_quote": 0.0,
                "sell_quote": 0.0,
                "delta_qty": 0.0,
                "delta_quote": 0.0,
                "cvd_qty": 0.0,
                "large_trade_threshold_quote": 0.0,
                "large_buy_quote": 0.0,
                "large_sell_quote": 0.0,
                "large_trade_direction": "neutral",
                "coverage_seconds": 0.0,
                "coverage_minutes": 0.0,
                "windows": {},
                "aggressor_layers": {},
                "large_trade_clusters": [],
                "absorption_zones": [],
                "cvd_path": [],
            }

        threshold = cls._quantile([float(row.get("quote_qty", 0.0)) for row in rows], large_trade_quantile)
        summary = cls._summarize_trade_rows(rows, threshold)
        start_ts = int(rows[0].get("timestamp", 0))
        end_ts = int(rows[-1].get("timestamp", 0))
        coverage_seconds = max(0.0, (end_ts - start_ts) / 1000)
        coverage_minutes = coverage_seconds / 60

        windows: Dict[str, Dict] = {}
        for label in ("1m", "5m", "15m", "30m"):
            seconds = cls._window_seconds(label)
            window_rows = [row for row in rows if int(row.get("timestamp", 0)) >= end_ts - seconds * 1000]
            window_summary = cls._summarize_trade_rows(window_rows, threshold)
            if window_rows:
                window_coverage = max(0.0, (int(window_rows[-1].get("timestamp", 0)) - int(window_rows[0].get("timestamp", 0))) / 1000)
            else:
                window_coverage = 0.0
            delta_quote = float(window_summary.get("delta_quote", 0.0))
            delta_qty = float(window_summary.get("delta_qty", 0.0))
            window_summary.update(
                {
                    "coverage_seconds": round(window_coverage, 6),
                    "coverage_ratio": round(min(1.0, cls._safe_div(window_coverage, seconds)), 6),
                    "delta_quote_per_minute": round(cls._safe_div(delta_quote, max(window_coverage / 60, 0.001)), 6),
                    "cvd_slope_qty_per_minute": round(cls._safe_div(delta_qty, max(window_coverage / 60, 0.001)), 6),
                }
            )
            windows[label] = window_summary

        quote_sizes = [float(row.get("quote_qty", 0.0)) for row in rows]
        q50 = cls._quantile(quote_sizes, 0.5)
        q80 = cls._quantile(quote_sizes, 0.8)
        q95 = cls._quantile(quote_sizes, 0.95)
        aggressor_layers: Dict[str, Dict] = {}
        layer_defs = (
            ("small", lambda value: value <= q50),
            ("medium", lambda value: q50 < value <= q80),
            ("large", lambda value: q80 < value <= q95),
            ("block", lambda value: value > q95),
        )
        for label, matcher in layer_defs:
            layer_rows = [row for row in rows if matcher(float(row.get("quote_qty", 0.0)))]
            aggressor_layers[label] = cls._summarize_trade_rows(layer_rows, threshold)

        large_rows = [row for row in rows if float(row.get("quote_qty", 0.0)) >= threshold]
        large_trade_clusters: List[Dict] = []
        absorption_zones: List[Dict] = []
        if large_rows:
            prices = sorted(float(row.get("price", 0.0)) for row in large_rows)
            mid_price = prices[len(prices) // 2]
            price_range = max(prices) - min(prices)
            bin_size = max(mid_price * 0.0005, price_range / 12 if price_range > 0 else 0.0, 25.0)
            clusters: Dict[float, Dict[str, Any]] = {}
            for row in large_rows:
                price = float(row.get("price", 0.0))
                center = round(round(price / bin_size) * bin_size, 2)
                cluster = clusters.setdefault(
                    center,
                    {
                        "center_price": center,
                        "zone_low": round(center - bin_size / 2, 2),
                        "zone_high": round(center + bin_size / 2, 2),
                        "trade_count": 0,
                        "total_quote": 0.0,
                        "buy_quote": 0.0,
                        "sell_quote": 0.0,
                        "min_price": price,
                        "max_price": price,
                    },
                )
                quote_qty = float(row.get("quote_qty", 0.0))
                cluster["trade_count"] += 1
                cluster["total_quote"] += quote_qty
                cluster["min_price"] = min(float(cluster["min_price"]), price)
                cluster["max_price"] = max(float(cluster["max_price"]), price)
                if row.get("aggressor_side") == "buy":
                    cluster["buy_quote"] += quote_qty
                else:
                    cluster["sell_quote"] += quote_qty

            ranked_clusters = sorted(clusters.values(), key=lambda item: item["total_quote"], reverse=True)
            cluster_totals = [float(item["total_quote"]) for item in ranked_clusters]
            cluster_threshold = cls._quantile(cluster_totals, 0.75)
            for cluster in ranked_clusters[:6]:
                total_quote = float(cluster["total_quote"])
                buy_quote = float(cluster["buy_quote"])
                sell_quote = float(cluster["sell_quote"])
                imbalance = cls._safe_div(abs(buy_quote - sell_quote), total_quote)
                price_span = float(cluster["max_price"]) - float(cluster["min_price"])
                if buy_quote > sell_quote * 1.5 and price_span <= bin_size * 0.4:
                    absorption_side = "ask_absorption"
                elif sell_quote > buy_quote * 1.5 and price_span <= bin_size * 0.4:
                    absorption_side = "bid_absorption"
                elif imbalance <= 0.2 and price_span <= bin_size * 0.5:
                    absorption_side = "two_way_absorption"
                else:
                    absorption_side = "none"
                dominant_side = "buy" if buy_quote > sell_quote else "sell" if sell_quote > buy_quote else "balanced"
                cluster_output = {
                    "center_price": round(float(cluster["center_price"]), 2),
                    "zone_low": round(float(cluster["zone_low"]), 2),
                    "zone_high": round(float(cluster["zone_high"]), 2),
                    "trade_count": int(cluster["trade_count"]),
                    "total_quote": round(total_quote, 6),
                    "buy_quote": round(buy_quote, 6),
                    "sell_quote": round(sell_quote, 6),
                    "dominant_side": dominant_side,
                }
                large_trade_clusters.append(cluster_output)
                if total_quote >= cluster_threshold and absorption_side != "none":
                    absorption_zones.append(
                        {
                            **cluster_output,
                            "absorption_side": absorption_side,
                            "imbalance_ratio": round(imbalance, 6),
                        }
                    )

        cvd_qty = 0.0
        delta_quote = 0.0
        cvd_path: List[Dict] = []
        step = max(1, len(rows) // 120)
        for idx, row in enumerate(rows, start=1):
            qty = float(row.get("qty", 0.0))
            quote_qty = float(row.get("quote_qty", 0.0))
            if row.get("aggressor_side") == "buy":
                cvd_qty += qty
                delta_quote += quote_qty
            else:
                cvd_qty -= qty
                delta_quote -= quote_qty
            if idx == len(rows) or idx % step == 0:
                cvd_path.append(
                    {
                        "timestamp": int(row.get("timestamp", 0)),
                        "price": round(float(row.get("price", 0.0)), 6),
                        "cvd_qty": round(cvd_qty, 6),
                        "delta_quote": round(delta_quote, 6),
                    }
                )

        return {
            "trade_count": len(rows),
            "from_timestamp": start_ts,
            "to_timestamp": end_ts,
            "coverage_seconds": round(coverage_seconds, 6),
            "coverage_minutes": round(coverage_minutes, 6),
            "trades_per_minute": round(cls._safe_div(len(rows), max(coverage_minutes, 0.001)), 6),
            **summary,
            "large_trade_threshold_quote": round(threshold, 6),
            "windows": windows,
            "aggressor_layers": aggressor_layers,
            "large_trade_clusters": large_trade_clusters,
            "absorption_zones": absorption_zones,
            "cvd_path": cvd_path,
        }

    @staticmethod
    def extract_kline_flow(klines_1m: Sequence[Dict], windows_bars: Tuple[int, ...] = (30, 60)) -> Dict:
        """Compute cumulative delta from 1m klines using taker buy volume fields.

        Requires klines with taker_buy_base / taker_buy_quote / quote_volume fields.
        Returns per-window buy/sell/delta breakdown and CVD trend direction.
        """
        candles = [
            c for c in sorted(list(klines_1m), key=lambda x: x.get("open_time", 0))
            if float(c.get("volume", 0)) > 0
        ]
        if not candles or "taker_buy_base" not in (candles[0] if candles else {}):
            return {"available": False, "reason": "missing_taker_buy_fields"}

        windows: Dict[str, Dict] = {}
        for bars in windows_bars:
            recent = candles[-bars:]
            if not recent:
                continue
            taker_buy_base = sum(float(c.get("taker_buy_base", 0)) for c in recent)
            taker_sell_base = sum(float(c.get("volume", 0)) - float(c.get("taker_buy_base", 0)) for c in recent)
            taker_buy_quote = sum(float(c.get("taker_buy_quote", 0)) for c in recent)
            taker_sell_quote = sum(float(c.get("quote_volume", 0)) - float(c.get("taker_buy_quote", 0)) for c in recent)
            delta_qty = taker_buy_base - taker_sell_base
            delta_quote = taker_buy_quote - taker_sell_quote

            running = 0.0
            cvd_series: List[float] = []
            for c in recent:
                b = float(c.get("taker_buy_base", 0))
                s = float(c.get("volume", 0)) - b
                running += b - s
                cvd_series.append(running)

            if len(cvd_series) >= 4:
                mid = len(cvd_series) // 2
                avg_first = sum(cvd_series[:mid]) / mid
                avg_second = sum(cvd_series[mid:]) / len(cvd_series[mid:])
                if avg_second > avg_first * 1.05:
                    cvd_trend = "rising"
                elif avg_second < avg_first * 0.95:
                    cvd_trend = "falling"
                else:
                    cvd_trend = "flat"
            else:
                cvd_trend = "unknown"

            label = f"{bars}m" if bars < 60 else f"{bars // 60}h"
            windows[label] = {
                "bars": len(recent),
                "buy_base": round(taker_buy_base, 4),
                "sell_base": round(taker_sell_base, 4),
                "delta_qty": round(delta_qty, 4),
                "buy_quote": round(taker_buy_quote, 2),
                "sell_quote": round(taker_sell_quote, 2),
                "delta_quote": round(delta_quote, 2),
                "direction": "buy" if delta_qty > 0 else "sell" if delta_qty < 0 else "flat",
                "cvd_trend": cvd_trend,
            }

        return {"available": True, "windows": windows}

    @classmethod
    def extract_price_level_delta(
        cls,
        trades: Sequence[Dict],
        window_minutes: int = 20,
        bin_size: float = 100.0,
    ) -> Dict:
        """Compute per-price-bin buy/sell delta (footprint-style) from recent agg_trades.

        Detects stacked imbalance (3+ consecutive bins same side) and absorption zones.
        Note: agg_trades typically cover only the most recent few minutes of high-activity
        markets; actual_coverage_minutes reflects true data span.
        """
        rows = sorted(list(trades), key=lambda x: int(x.get("timestamp", 0)))
        if not rows:
            return {"available": False, "reason": "no_trades"}

        end_ts = int(rows[-1]["timestamp"])
        cutoff_ts = end_ts - window_minutes * 60 * 1000
        window_rows = [r for r in rows if int(r.get("timestamp", 0)) >= cutoff_ts]
        if not window_rows:
            return {"available": False, "reason": "no_trades_in_window"}

        actual_coverage_minutes = max(0.0, (end_ts - int(window_rows[0]["timestamp"])) / 60000)

        prices = [float(r.get("price", 0)) for r in window_rows if float(r.get("price", 0)) > 0]
        if not prices:
            return {"available": False, "reason": "no_valid_prices"}
        mid_price = sorted(prices)[len(prices) // 2]
        actual_bin = max(bin_size, mid_price * 0.0015)

        bins: Dict[float, Dict] = {}
        for row in window_rows:
            price = float(row.get("price", 0))
            if price <= 0:
                continue
            center = round(round(price / actual_bin) * actual_bin, 2)
            if center not in bins:
                bins[center] = {"price": center, "buy": 0.0, "sell": 0.0}
            q = float(row.get("quote_qty", 0))
            if row.get("aggressor_side") == "buy":
                bins[center]["buy"] += q
            else:
                bins[center]["sell"] += q

        sorted_bins: List[Dict] = []
        for b in sorted(bins.values(), key=lambda x: x["price"]):
            total = b["buy"] + b["sell"]
            if total <= 0:
                continue
            delta = b["buy"] - b["sell"]
            imb = delta / total
            sig = "buy_imbalance" if imb >= 0.5 else "sell_imbalance" if imb <= -0.5 else "balanced"
            sorted_bins.append({
                "price": b["price"],
                "buy_quote": round(b["buy"], 2),
                "sell_quote": round(b["sell"], 2),
                "delta_quote": round(delta, 2),
                "imbalance": round(imb, 3),
                "signal": sig,
                "total_quote": round(total, 2),
            })

        # Stacked imbalance: 3+ consecutive bins same direction
        stacked: List[Dict] = []
        i = 0
        while i < len(sorted_bins):
            sig = sorted_bins[i]["signal"]
            if sig in ("buy_imbalance", "sell_imbalance"):
                j = i
                while j < len(sorted_bins) and sorted_bins[j]["signal"] == sig:
                    j += 1
                count = j - i
                if count >= 3:
                    stacked.append({
                        "direction": "buy" if sig == "buy_imbalance" else "sell",
                        "count": count,
                        "from_price": sorted_bins[i]["price"],
                        "to_price": sorted_bins[j - 1]["price"],
                    })
                i = j
            else:
                i += 1

        # Absorption: high volume but near-balanced (both sides fighting at same level)
        max_vol = max((b["total_quote"] for b in sorted_bins), default=0.0)
        vol_threshold = max_vol * 0.35
        absorption: List[Dict] = [
            {"price": b["price"], "total_quote": b["total_quote"], "imbalance": b["imbalance"]}
            for b in sorted_bins
            if b["total_quote"] >= vol_threshold and abs(b["imbalance"]) < 0.2
        ]

        top_buy = sorted(
            [b for b in sorted_bins if b["signal"] == "buy_imbalance"],
            key=lambda x: abs(x["imbalance"]), reverse=True,
        )[:3]
        top_sell = sorted(
            [b for b in sorted_bins if b["signal"] == "sell_imbalance"],
            key=lambda x: abs(x["imbalance"]), reverse=True,
        )[:3]

        return {
            "available": True,
            "window_minutes": window_minutes,
            "actual_coverage_minutes": round(actual_coverage_minutes, 1),
            "bin_size": round(actual_bin, 2),
            "total_bins": len(sorted_bins),
            "stacked_imbalance": stacked,
            "absorption_zones": absorption,
            "top_buy_imbalance": [b["price"] for b in top_buy],
            "top_sell_imbalance": [b["price"] for b in top_sell],
            "all_bins": sorted_bins,
        }

    @staticmethod
    def extract_key_level_flows(
        trades: Sequence[Dict],
        key_levels: Sequence[Dict],
        radius_pct: float = 0.0015,
    ) -> List[Dict]:
        """Summarise taker buy/sell around each key level.

        Args:
            key_levels: list of {"name": str, "price": float}.
            radius_pct: price radius as fraction (default 0.15 %).
        """
        if not trades or not key_levels:
            return []
        rows = list(trades)
        results: List[Dict] = []
        for kl in key_levels:
            price = float(kl.get("price", 0))
            if price <= 0:
                continue
            radius = price * radius_pct
            lo, hi = price - radius, price + radius
            buy_q, sell_q = 0.0, 0.0
            for r in rows:
                rp = float(r.get("price", 0))
                if lo <= rp <= hi:
                    q = float(r.get("quote_qty", 0))
                    if r.get("aggressor_side") == "buy":
                        buy_q += q
                    else:
                        sell_q += q
            if buy_q + sell_q < 1:
                continue
            net = buy_q - sell_q
            if buy_q + sell_q > 0 and abs(net) / (buy_q + sell_q) < 0.15:
                tag = "absorbed"
            elif net > 0:
                tag = "buy_dominant"
            else:
                tag = "sell_dominant"
            results.append({
                "name": kl.get("name", "?"),
                "price": round(price, 1),
                "buy": round(buy_q, 0),
                "sell": round(sell_q, 0),
                "net": round(net, 0),
                "tag": tag,
            })
        return results

    @staticmethod
    def extract_key_level_tests(
        candles_15m: Sequence[Dict],
        key_levels: Sequence[Dict],
        lookback: int = 48,
        radius_pct: float = 0.0015,
    ) -> Dict[str, Dict]:
        """Count how many times price has tested each key level over the last
        *lookback* 15m candles (~12 hours), and measure average post-test bounce.

        A "test" occurs when a candle's high or low comes within *radius_pct* of
        the level price.  The bounce after each test is the maximum close-distance
        from the level over the next 3 candles.

        Returns:
            dict keyed by "{name}@{price}" → {tests, first_test_min_ago, avg_bounce_pct}
        """
        import time as _time

        if not candles_15m or not key_levels:
            return {}

        candles = list(candles_15m)[-lookback:]
        now_ms = int(_time.time() * 1000)
        bar_ms = 15 * 60 * 1000  # 15 minutes in ms

        result: Dict[str, Dict] = {}
        for kl in key_levels:
            price = float(kl.get("price", 0))
            if price <= 0:
                continue
            name = str(kl.get("name", "?"))
            radius = price * radius_pct

            test_indices: List[int] = []
            for i, c in enumerate(candles):
                h = float(c.get("high", 0))
                l_ = float(c.get("low", 0))
                if (l_ - radius) <= price <= (h + radius):
                    test_indices.append(i)

            if not test_indices:
                key = f"{name}@{round(price, 1)}"
                result[key] = {"tests_12h": 0, "first_test_min_ago": None, "avg_bounce_pct": None}
                continue

            # First test time
            first_idx = test_indices[0]
            # Estimate time: work backwards from now
            bars_ago_first = len(candles) - 1 - first_idx
            first_test_min_ago = round(bars_ago_first * 15)

            # Average bounce: after each test, max abs close deviation in next 3 bars
            bounces: List[float] = []
            for idx in test_indices:
                next_bars = candles[idx + 1: idx + 4]
                if next_bars:
                    bounce = max(
                        abs(float(b.get("close", price)) - price) / price * 100
                        for b in next_bars
                    )
                    bounces.append(bounce)
            avg_bounce = round(sum(bounces) / len(bounces), 3) if bounces else 0.0

            key = f"{name}@{round(price, 1)}"
            result[key] = {
                "tests_12h": len(test_indices),
                "first_test_min_ago": first_test_min_ago,
                "avg_bounce_pct": avg_bounce,
            }

        return result
