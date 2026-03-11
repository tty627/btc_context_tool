from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple


class FeatureExtractor:
    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _price_precision(price: float) -> int:
        absolute = abs(price)
        if absolute >= 1000:
            return 1
        if absolute >= 1:
            return 3
        return 6

    @classmethod
    def _price_key(cls, price: float) -> float:
        return round(float(price), cls._price_precision(price))

    @staticmethod
    def _window_seconds(label: str) -> int:
        mapping = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "1h": 3600}
        return mapping.get(label, 0)

    @staticmethod
    def _snapshot_timestamp_ms(snapshot: Dict) -> int:
        event_time_ms = snapshot.get("event_time_ms")
        if isinstance(event_time_ms, int):
            return event_time_ms

        raw = snapshot.get("event_time")
        if isinstance(raw, str):
            try:
                return int(datetime.fromisoformat(raw).timestamp() * 1000)
            except ValueError:
                return 0
        return 0

    @staticmethod
    def _quantile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        if q <= 0:
            return min(values)
        if q >= 1:
            return max(values)
        sorted_values = sorted(values)
        index = int((len(sorted_values) - 1) * q)
        return float(sorted_values[index])

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

    @staticmethod
    def extract_orderbook_features(orderbook: Dict) -> Dict:
        bids: List[Dict] = orderbook["bids"]
        asks: List[Dict] = orderbook["asks"]
        bid_volume = sum(row["qty"] for row in bids)
        ask_volume = sum(row["qty"] for row in asks)
        total = bid_volume + ask_volume
        imbalance = 0.0 if total == 0 else (bid_volume - ask_volume) / total
        strongest_bid = max(bids, key=lambda x: x["qty"]) if bids else {"price": 0.0, "qty": 0.0}
        strongest_ask = max(asks, key=lambda x: x["qty"]) if asks else {"price": 0.0, "qty": 0.0}
        return {
            "bid_volume": round(bid_volume, 6),
            "ask_volume": round(ask_volume, 6),
            "imbalance": round(imbalance, 6),
            "bid_wall": strongest_bid,
            "ask_wall": strongest_ask,
        }

    @staticmethod
    def classify_trend(ema: Dict[str, float]) -> str:
        ema7 = ema["7"]
        ema25 = ema["25"]
        ema99 = ema["99"]
        if ema7 > ema25 > ema99:
            return "bullish"
        if ema7 < ema25 < ema99:
            return "bearish"
        return "neutral"

    @staticmethod
    def classify_momentum(macd: Dict[str, float]) -> str:
        if macd["dif"] > macd["dea"] and macd["hist"] > 0:
            return "momentum_up"
        if macd["dif"] < macd["dea"] and macd["hist"] < 0:
            return "momentum_down"
        return "momentum_neutral"

    @staticmethod
    def classify_kdj(kdj: Dict[str, float]) -> str:
        k = kdj["k"]
        if k > 80:
            return "overbought"
        if k < 20:
            return "oversold"
        return "neutral"

    @classmethod
    def extract_timeframe_features(cls, indicators_by_timeframe: Dict[str, Dict]) -> Dict[str, Dict]:
        output = {}
        for timeframe, metrics in indicators_by_timeframe.items():
            rsi = metrics.get("rsi", {})
            output[timeframe] = {
                "trend": cls.classify_trend(metrics["ema"]),
                "momentum": cls.classify_momentum(metrics["macd"]),
                "kdj_state": cls.classify_kdj(metrics["kdj"]),
                "rsi_state": rsi.get("state", "unknown"),
                "rsi_divergence": rsi.get("divergence", "none"),
            }
        return output

    @staticmethod
    def extract_recent_4h_range(candles_15m: Sequence[Dict]) -> Dict:
        window = list(candles_15m[-16:])
        if not window:
            return {
                "high": 0.0,
                "low": 0.0,
                "range_abs": 0.0,
                "range_pct": 0.0,
                "from_open_time": 0,
                "to_close_time": 0,
                "high_sweep_zone": {"zone_low": 0.0, "zone_high": 0.0},
                "low_sweep_zone": {"zone_low": 0.0, "zone_high": 0.0},
            }
        recent_high = max(row["high"] for row in window)
        recent_low = min(row["low"] for row in window)
        range_abs = recent_high - recent_low
        range_pct = 0.0 if recent_low <= 0 else range_abs / recent_low * 100
        return {
            "high": round(recent_high, 6),
            "low": round(recent_low, 6),
            "range_abs": round(range_abs, 6),
            "range_pct": round(range_pct, 6),
            "from_open_time": int(window[0]["open_time"]),
            "to_close_time": int(window[-1]["close_time"]),
            "high_sweep_zone": {
                "zone_low": round(recent_high * 0.998, 6),
                "zone_high": round(recent_high * 1.01, 6),
            },
            "low_sweep_zone": {
                "zone_low": round(recent_low * 0.99, 6),
                "zone_high": round(recent_low * 1.002, 6),
            },
        }

    @staticmethod
    def extract_volume_change(klines_by_timeframe: Dict[str, Sequence[Dict]]) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        for timeframe, candles in klines_by_timeframe.items():
            rows = list(candles)
            if not rows:
                result[timeframe] = {
                    "last_volume": 0.0,
                    "prev_volume": 0.0,
                    "delta_volume": 0.0,
                    "delta_pct": 0.0,
                    "avg20_volume": 0.0,
                    "vs_avg20_pct": 0.0,
                }
                continue

            last_volume = rows[-1]["volume"]
            prev_volume = rows[-2]["volume"] if len(rows) > 1 else 0.0
            delta_volume = last_volume - prev_volume
            delta_pct = 0.0 if prev_volume <= 0 else delta_volume / prev_volume * 100
            lookback = rows[-20:]
            avg20 = sum(row["volume"] for row in lookback) / len(lookback)
            vs_avg20_pct = 0.0 if avg20 <= 0 else (last_volume - avg20) / avg20 * 100

            result[timeframe] = {
                "last_volume": round(last_volume, 6),
                "prev_volume": round(prev_volume, 6),
                "delta_volume": round(delta_volume, 6),
                "delta_pct": round(delta_pct, 6),
                "avg20_volume": round(avg20, 6),
                "vs_avg20_pct": round(vs_avg20_pct, 6),
            }
        return result

    @staticmethod
    def _trend_from_delta_pct(delta_pct: float) -> str:
        if delta_pct > 2:
            return "increasing"
        if delta_pct < -2:
            return "decreasing"
        return "flat"

    @staticmethod
    def _label_price_oi_state(price_delta_pct: float, oi_delta_pct: float) -> Tuple[str, str]:
        if abs(price_delta_pct) < 0.03 and abs(oi_delta_pct) < 0.1:
            return "flat", "low_energy"
        if price_delta_pct >= 0 and oi_delta_pct >= 0:
            return "price_up_oi_up", "new_longs"
        if price_delta_pct >= 0 and oi_delta_pct < 0:
            return "price_up_oi_down", "short_covering"
        if price_delta_pct < 0 and oi_delta_pct >= 0:
            return "price_down_oi_up", "new_shorts"
        return "price_down_oi_down", "long_unwind"

    @staticmethod
    def _align_price_points(history: Sequence[Dict], candles: Sequence[Dict]) -> List[Dict]:
        oi_rows = sorted(list(history), key=lambda x: x.get("timestamp", 0))
        candle_rows = sorted(list(candles), key=lambda x: x.get("close_time", 0))
        if not oi_rows:
            return []

        aligned: List[Dict] = []
        candle_idx = 0
        last_price = 0.0
        for row in oi_rows:
            timestamp = int(row.get("timestamp", 0))
            while candle_idx + 1 < len(candle_rows) and int(candle_rows[candle_idx]["close_time"]) <= timestamp:
                candle_idx += 1

            if candle_rows:
                while candle_idx > 0 and int(candle_rows[candle_idx]["close_time"]) > timestamp:
                    candle_idx -= 1
                candle = candle_rows[candle_idx]
                last_price = float(candle.get("close", last_price))

            aligned.append(
                {
                    "timestamp": timestamp,
                    "open_interest": float(row.get("sum_open_interest", 0.0)),
                    "open_interest_value": float(row.get("sum_open_interest_value", 0.0)),
                    "price": last_price,
                }
            )
        return aligned

    @classmethod
    def _summarize_open_interest_period(
        cls,
        current_open_interest: float,
        history: Sequence[Dict],
        candles: Sequence[Dict],
    ) -> Dict:
        rows = sorted(list(history), key=lambda x: x.get("timestamp", 0))
        if not rows:
            return {
                "current": round(current_open_interest, 6),
                "history_points": 0,
                "delta_abs": 0.0,
                "delta_pct": 0.0,
                "vs_avg_pct": 0.0,
                "trend": "unknown",
                "latest_state": "unknown",
                "latest_interpretation": "unknown",
                "series": [],
            }

        first_oi = float(rows[0].get("sum_open_interest", 0.0))
        values = [float(row.get("sum_open_interest", 0.0)) for row in rows if float(row.get("sum_open_interest", 0.0)) > 0]
        avg_oi = sum(values) / len(values) if values else 0.0
        delta_abs = current_open_interest - first_oi
        delta_pct = 0.0 if first_oi <= 0 else delta_abs / first_oi * 100
        vs_avg_pct = 0.0 if avg_oi <= 0 else (current_open_interest - avg_oi) / avg_oi * 100
        trend = cls._trend_from_delta_pct(delta_pct)

        aligned = cls._align_price_points(rows, candles)
        series: List[Dict] = []
        latest_state = "unknown"
        latest_interpretation = "unknown"
        for idx, point in enumerate(aligned):
            if idx == 0:
                price_delta_pct = 0.0
                oi_delta_period_pct = 0.0
            else:
                prev = aligned[idx - 1]
                price_delta_pct = cls._safe_div(point["price"] - prev["price"], prev["price"]) * 100
                oi_delta_period_pct = cls._safe_div(
                    point["open_interest"] - prev["open_interest"],
                    prev["open_interest"],
                ) * 100
            state, interpretation = cls._label_price_oi_state(price_delta_pct, oi_delta_period_pct)
            latest_state = state
            latest_interpretation = interpretation
            series.append(
                {
                    "timestamp": point["timestamp"],
                    "price": round(point["price"], 6),
                    "open_interest": round(point["open_interest"], 6),
                    "open_interest_value": round(point["open_interest_value"], 6),
                    "price_delta_pct": round(price_delta_pct, 6),
                    "oi_delta_pct": round(oi_delta_period_pct, 6),
                    "state": state,
                    "interpretation": interpretation,
                }
            )

        return {
            "current": round(current_open_interest, 6),
            "history_points": len(rows),
            "delta_abs": round(delta_abs, 6),
            "delta_pct": round(delta_pct, 6),
            "vs_avg_pct": round(vs_avg_pct, 6),
            "trend": trend,
            "latest_state": latest_state,
            "latest_interpretation": latest_interpretation,
            "series": series,
        }

    @classmethod
    def extract_open_interest_trend(
        cls,
        current_open_interest: float,
        history_by_period: Dict[str, Sequence[Dict]],
        price_candles_by_period: Dict[str, Sequence[Dict]],
        trade_flow: Dict,
        volume_change: Dict[str, Dict],
        summary_period: str = "5m",
    ) -> Dict:
        periods: Dict[str, Dict] = {}
        for period, history in history_by_period.items():
            periods[period] = cls._summarize_open_interest_period(
                current_open_interest=current_open_interest,
                history=history,
                candles=price_candles_by_period.get(period, []),
            )

        summary = periods.get(summary_period) or next(iter(periods.values()), None)
        if summary is None:
            return {
                "current": round(current_open_interest, 6),
                "history_points": 0,
                "delta_abs": 0.0,
                "delta_pct": 0.0,
                "vs_avg_pct": 0.0,
                "trend": "unknown",
                "summary_period": summary_period,
                "periods": {},
                "composite_signal": "unknown",
                "volume_oi_cvd_state": {
                    "volume_state": "unknown",
                    "oi_state": "unknown",
                    "cvd_state": "unknown",
                    "price_oi_state": "unknown",
                },
            }

        window_5m = trade_flow.get("windows", {}).get("5m", {})
        cvd_delta_quote = float(window_5m.get("delta_quote", 0.0))
        if cvd_delta_quote > 0:
            cvd_state = "positive"
        elif cvd_delta_quote < 0:
            cvd_state = "negative"
        else:
            cvd_state = "flat"

        volume_metrics = volume_change.get("15m", {})
        volume_vs_avg = float(volume_metrics.get("vs_avg20_pct", 0.0))
        if volume_vs_avg >= 15:
            volume_state = "expanding"
        elif volume_vs_avg <= -15:
            volume_state = "contracting"
        else:
            volume_state = "normal"

        latest_state = str(summary.get("latest_state", "unknown"))
        if latest_state == "price_up_oi_up" and cvd_state == "positive":
            composite_signal = "trend_up_confirmed_by_oi_and_cvd"
        elif latest_state == "price_down_oi_up" and cvd_state == "negative":
            composite_signal = "trend_down_confirmed_by_oi_and_cvd"
        elif latest_state == "price_up_oi_down":
            composite_signal = "short_covering_rally"
        elif latest_state == "price_down_oi_down":
            composite_signal = "long_unwind_drop"
        elif latest_state == "price_up_oi_up":
            composite_signal = "price_up_but_cvd_not_confirming"
        elif latest_state == "price_down_oi_up":
            composite_signal = "price_down_but_cvd_not_confirming"
        else:
            composite_signal = "mixed"

        return {
            "current": summary["current"],
            "history_points": summary["history_points"],
            "delta_abs": summary["delta_abs"],
            "delta_pct": summary["delta_pct"],
            "vs_avg_pct": summary["vs_avg_pct"],
            "trend": summary["trend"],
            "summary_period": summary_period,
            "latest_state": summary.get("latest_state", "unknown"),
            "latest_interpretation": summary.get("latest_interpretation", "unknown"),
            "periods": periods,
            "composite_signal": composite_signal,
            "volume_oi_cvd_state": {
                "volume_state": volume_state,
                "oi_state": summary["trend"],
                "cvd_state": cvd_state,
                "price_oi_state": latest_state,
            },
        }

    @staticmethod
    def _extract_ratio_snapshot(rows: Sequence[Dict]) -> Dict:
        series = sorted(list(rows), key=lambda x: x.get("timestamp", 0))
        if not series:
            return {
                "latest_ratio": 0.0,
                "avg_ratio": 0.0,
                "delta_pct": 0.0,
                "long_account": 0.0,
                "short_account": 0.0,
                "crowding": "unknown",
                "history": [],
            }

        ratios = [float(row.get("long_short_ratio", 0.0)) for row in series]
        latest = ratios[-1]
        first = ratios[0]
        avg_ratio = sum(ratios) / len(ratios)
        delta_pct = 0.0 if first <= 0 else (latest - first) / first * 100
        latest_row = series[-1]
        long_account = float(latest_row.get("long_account", 0.0))
        short_account = float(latest_row.get("short_account", 0.0))

        if latest >= 1.3:
            crowding = "long_crowded"
        elif 0 < latest <= 0.77:
            crowding = "short_crowded"
        else:
            crowding = "balanced"

        history = [
            {
                "timestamp": int(row.get("timestamp", 0)),
                "ratio": round(float(row.get("long_short_ratio", 0.0)), 6),
                "long_account": round(float(row.get("long_account", 0.0)), 6),
                "short_account": round(float(row.get("short_account", 0.0)), 6),
            }
            for row in series
        ]

        return {
            "latest_ratio": round(latest, 6),
            "avg_ratio": round(avg_ratio, 6),
            "delta_pct": round(delta_pct, 6),
            "long_account": round(long_account, 6),
            "short_account": round(short_account, 6),
            "crowding": crowding,
            "history": history,
        }

    @classmethod
    def extract_long_short_ratio(
        cls,
        global_account_ratio: Sequence[Dict],
        top_trader_ratio: Sequence[Dict],
    ) -> Dict:
        global_snapshot = cls._extract_ratio_snapshot(global_account_ratio)
        top_snapshot = cls._extract_ratio_snapshot(top_trader_ratio)

        signals = [global_snapshot.get("crowding"), top_snapshot.get("crowding")]
        if "long_crowded" in signals:
            overall = "long_crowded"
        elif "short_crowded" in signals:
            overall = "short_crowded"
        elif "unknown" in signals and signals.count("unknown") == len(signals):
            overall = "unknown"
        else:
            overall = "balanced"

        return {
            "global_account": global_snapshot,
            "top_trader_position": top_snapshot,
            "overall_crowding": overall,
        }

    @classmethod
    def _collect_wall_runs(cls, snapshots: Sequence[Dict], top_n: int = 4) -> List[Dict]:
        rows = sorted(list(snapshots), key=cls._snapshot_timestamp_ms)
        active: Dict[str, Dict[float, Dict]] = {"bid": {}, "ask": {}}
        completed: List[Dict] = []

        for snapshot in rows:
            timestamp = cls._snapshot_timestamp_ms(snapshot)
            for side, levels in (("bid", snapshot.get("bids", [])), ("ask", snapshot.get("asks", []))):
                top_levels = sorted(levels, key=lambda row: float(row.get("qty", 0.0)), reverse=True)[:top_n]
                present_prices: set[float] = set()
                for rank, level in enumerate(top_levels, start=1):
                    price = cls._price_key(float(level.get("price", 0.0)))
                    qty = float(level.get("qty", 0.0))
                    present_prices.add(price)
                    current = active[side].get(price)
                    if current is None:
                        active[side][price] = {
                            "side": side,
                            "price": price,
                            "start_ms": timestamp,
                            "end_ms": timestamp,
                            "snapshots": 1,
                            "sum_qty": qty,
                            "max_qty": qty,
                            "best_rank": rank,
                        }
                    else:
                        current["end_ms"] = timestamp
                        current["snapshots"] += 1
                        current["sum_qty"] += qty
                        current["max_qty"] = max(float(current["max_qty"]), qty)
                        current["best_rank"] = min(int(current["best_rank"]), rank)

                for price in list(active[side].keys()):
                    if price in present_prices:
                        continue
                    current = active[side].pop(price)
                    completed.append(current)

        for side in ("bid", "ask"):
            for current in active[side].values():
                completed.append(current)

        total_snapshots = max(1, len(rows))
        output: List[Dict] = []
        for item in completed:
            lifetime_seconds = max(0.0, (int(item["end_ms"]) - int(item["start_ms"])) / 1000)
            output.append(
                {
                    "side": item["side"],
                    "price": round(float(item["price"]), cls._price_precision(float(item["price"]))),
                    "lifetime_seconds": round(lifetime_seconds, 6),
                    "snapshots": int(item["snapshots"]),
                    "persistence_ratio": round(cls._safe_div(float(item["snapshots"]), total_snapshots), 6),
                    "avg_qty": round(cls._safe_div(float(item["sum_qty"]), float(item["snapshots"])), 6),
                    "max_qty": round(float(item["max_qty"]), 6),
                    "best_rank": int(item["best_rank"]),
                    "start_ms": int(item["start_ms"]),
                    "end_ms": int(item["end_ms"]),
                }
            )

        output.sort(key=lambda row: (row["lifetime_seconds"], row["persistence_ratio"], row["max_qty"]), reverse=True)
        return output[:8]

    @classmethod
    def _top_level_activity(cls, activity: Dict[float, Dict]) -> List[Dict]:
        ranked = sorted(
            activity.values(),
            key=lambda row: (row["added_qty"] + row["cancelled_qty"], row["presence_ratio"], row["max_qty"]),
            reverse=True,
        )
        output: List[Dict] = []
        for row in ranked[:6]:
            output.append(
                {
                    "price": round(float(row["price"]), cls._price_precision(float(row["price"]))),
                    "added_qty": round(float(row["added_qty"]), 6),
                    "cancelled_qty": round(float(row["cancelled_qty"]), 6),
                    "net_qty": round(float(row["added_qty"]) - float(row["cancelled_qty"]), 6),
                    "avg_qty": round(float(row["avg_qty"]), 6),
                    "max_qty": round(float(row["max_qty"]), 6),
                    "updates": int(row["updates"]),
                    "presence_ratio": round(float(row["presence_ratio"]), 6),
                    "best_rank": int(row["best_rank"]),
                }
            )
        return output

    @classmethod
    def extract_orderbook_dynamics(cls, snapshots: Sequence[Dict], trades: Sequence[Dict] | None = None) -> Dict:
        rows = sorted(list(snapshots), key=cls._snapshot_timestamp_ms)
        if len(rows) < 2:
            return {
                "snapshot_count": len(rows),
                "sample_duration_seconds": 0.0,
                "avg_bid_volume_change": 0.0,
                "avg_ask_volume_change": 0.0,
                "bid_add_rate_per_second": 0.0,
                "bid_cancel_rate_per_second": 0.0,
                "ask_add_rate_per_second": 0.0,
                "ask_cancel_rate_per_second": 0.0,
                "best_bid_change_count": 0,
                "best_ask_change_count": 0,
                "best_bid_change_per_minute": 0.0,
                "best_ask_change_per_minute": 0.0,
                "avg_wall_lifetime_seconds": 0.0,
                "max_wall_lifetime_seconds": 0.0,
                "wall_pull_events": 0,
                "wall_add_events": 0,
                "wall_absorption_events": 0,
                "wall_sweep_events": 0,
                "wall_pull_without_trade_events": 0,
                "passive_absorption_quote": 0.0,
                "aggressive_sweep_quote": 0.0,
                "spoofing_risk": "unknown",
                "wall_behavior": "unknown",
                "top_bid_level_activity": [],
                "top_ask_level_activity": [],
                "persistent_walls": [],
                "series": [],
            }

        duration_seconds = max(0.001, (cls._snapshot_timestamp_ms(rows[-1]) - cls._snapshot_timestamp_ms(rows[0])) / 1000)
        series: List[Dict] = []
        level_activity: Dict[str, Dict[float, Dict[str, Any]]] = {"bid": {}, "ask": {}}
        trade_rows = sorted(list(trades or []), key=lambda row: int(row.get("timestamp", 0)))

        for snapshot in rows:
            bids = snapshot.get("bids", [])
            asks = snapshot.get("asks", [])
            bid_volume = sum(float(row.get("qty", 0.0)) for row in bids)
            ask_volume = sum(float(row.get("qty", 0.0)) for row in asks)
            best_bid = float(bids[0].get("price", 0.0)) if bids else 0.0
            best_ask = float(asks[0].get("price", 0.0)) if asks else 0.0
            total = bid_volume + ask_volume
            imbalance = 0.0 if total == 0 else (bid_volume - ask_volume) / total
            bid_wall = max(bids, key=lambda row: float(row.get("qty", 0.0))) if bids else {"price": 0.0, "qty": 0.0}
            ask_wall = max(asks, key=lambda row: float(row.get("qty", 0.0))) if asks else {"price": 0.0, "qty": 0.0}
            timestamp = cls._snapshot_timestamp_ms(snapshot)

            series.append(
                {
                    "timestamp": timestamp,
                    "best_bid": round(best_bid, cls._price_precision(best_bid)),
                    "best_ask": round(best_ask, cls._price_precision(best_ask)),
                    "spread": round(max(0.0, best_ask - best_bid), 6),
                    "bid_volume": round(bid_volume, 6),
                    "ask_volume": round(ask_volume, 6),
                    "imbalance": round(imbalance, 6),
                    "bid_wall_price": round(float(bid_wall.get("price", 0.0)), cls._price_precision(float(bid_wall.get("price", 0.0)))),
                    "bid_wall_qty": round(float(bid_wall.get("qty", 0.0)), 6),
                    "ask_wall_price": round(float(ask_wall.get("price", 0.0)), cls._price_precision(float(ask_wall.get("price", 0.0)))),
                    "ask_wall_qty": round(float(ask_wall.get("qty", 0.0)), 6),
                }
            )

            for side, levels in (("bid", bids), ("ask", asks)):
                for rank, level in enumerate(levels, start=1):
                    price = cls._price_key(float(level.get("price", 0.0)))
                    qty = float(level.get("qty", 0.0))
                    record = level_activity[side].setdefault(
                        price,
                        {
                            "price": price,
                            "added_qty": 0.0,
                            "cancelled_qty": 0.0,
                            "sum_qty": 0.0,
                            "max_qty": 0.0,
                            "presence_count": 0,
                            "updates": 0,
                            "best_rank": rank,
                        },
                    )
                    record["sum_qty"] += qty
                    record["max_qty"] = max(float(record["max_qty"]), qty)
                    record["presence_count"] += 1
                    record["best_rank"] = min(int(record["best_rank"]), rank)

        bid_changes: List[float] = []
        ask_changes: List[float] = []
        wall_pull_events = 0
        wall_add_events = 0
        best_bid_change_count = 0
        best_ask_change_count = 0
        spread_change_count = 0
        mid_price_change_count = 0
        passive_absorption_quote = 0.0
        aggressive_sweep_quote = 0.0
        pull_without_trade_quote = 0.0
        wall_absorption_events = 0
        wall_sweep_events = 0
        wall_pull_without_trade_events = 0
        bid_added = 0.0
        bid_cancelled = 0.0
        ask_added = 0.0
        ask_cancelled = 0.0

        trade_idx = 0
        for before, after in zip(rows[:-1], rows[1:]):
            before_bids = before.get("bids", [])
            before_asks = before.get("asks", [])
            after_bids = after.get("bids", [])
            after_asks = after.get("asks", [])
            before_bid_volume = sum(float(row.get("qty", 0.0)) for row in before_bids)
            after_bid_volume = sum(float(row.get("qty", 0.0)) for row in after_bids)
            before_ask_volume = sum(float(row.get("qty", 0.0)) for row in before_asks)
            after_ask_volume = sum(float(row.get("qty", 0.0)) for row in after_asks)
            bid_changes.append(after_bid_volume - before_bid_volume)
            ask_changes.append(after_ask_volume - before_ask_volume)

            before_best_bid = float(before_bids[0].get("price", 0.0)) if before_bids else 0.0
            after_best_bid = float(after_bids[0].get("price", 0.0)) if after_bids else 0.0
            before_best_ask = float(before_asks[0].get("price", 0.0)) if before_asks else 0.0
            after_best_ask = float(after_asks[0].get("price", 0.0)) if after_asks else 0.0
            before_spread = max(0.0, before_best_ask - before_best_bid)
            after_spread = max(0.0, after_best_ask - after_best_bid)
            before_mid = (before_best_bid + before_best_ask) / 2 if before_best_bid and before_best_ask else 0.0
            after_mid = (after_best_bid + after_best_ask) / 2 if after_best_bid and after_best_ask else 0.0
            if after_best_bid != before_best_bid:
                best_bid_change_count += 1
            if after_best_ask != before_best_ask:
                best_ask_change_count += 1
            if after_spread != before_spread:
                spread_change_count += 1
            if after_mid != before_mid:
                mid_price_change_count += 1

            for side, before_levels, after_levels in (("bid", before_bids, after_bids), ("ask", before_asks, after_asks)):
                before_map = {cls._price_key(float(row.get("price", 0.0))): float(row.get("qty", 0.0)) for row in before_levels}
                after_map = {cls._price_key(float(row.get("price", 0.0))): float(row.get("qty", 0.0)) for row in after_levels}
                before_top = {
                    cls._price_key(float(row.get("price", 0.0)))
                    for row in sorted(before_levels, key=lambda level: float(level.get("qty", 0.0)), reverse=True)[:6]
                }
                after_top = {
                    cls._price_key(float(row.get("price", 0.0)))
                    for row in sorted(after_levels, key=lambda level: float(level.get("qty", 0.0)), reverse=True)[:6]
                }
                wall_pull_events += len(before_top - after_top)
                wall_add_events += len(after_top - before_top)

                for price in set(before_map) | set(after_map):
                    before_qty = before_map.get(price, 0.0)
                    after_qty = after_map.get(price, 0.0)
                    delta = after_qty - before_qty
                    record = level_activity[side].setdefault(
                        price,
                        {
                            "price": price,
                            "added_qty": 0.0,
                            "cancelled_qty": 0.0,
                            "sum_qty": 0.0,
                            "max_qty": 0.0,
                            "presence_count": 0,
                            "updates": 0,
                            "best_rank": 999,
                        },
                    )
                    if delta > 0:
                        record["added_qty"] += delta
                        if side == "bid":
                            bid_added += delta
                        else:
                            ask_added += delta
                    elif delta < 0:
                        record["cancelled_qty"] += abs(delta)
                        if side == "bid":
                            bid_cancelled += abs(delta)
                        else:
                            ask_cancelled += abs(delta)
                    if delta != 0:
                        record["updates"] += 1

            before_ts = cls._snapshot_timestamp_ms(before)
            after_ts = cls._snapshot_timestamp_ms(after)
            interval_trades: List[Dict] = []
            while trade_idx < len(trade_rows) and int(trade_rows[trade_idx].get("timestamp", 0)) < before_ts:
                trade_idx += 1
            scan_idx = trade_idx
            while scan_idx < len(trade_rows):
                timestamp = int(trade_rows[scan_idx].get("timestamp", 0))
                if timestamp >= after_ts:
                    break
                interval_trades.append(trade_rows[scan_idx])
                scan_idx += 1

            for side, before_levels, after_levels in (("bid", before_bids, after_bids), ("ask", before_asks, after_asks)):
                if not before_levels:
                    continue
                strongest_before = max(before_levels, key=lambda row: float(row.get("qty", 0.0)))
                wall_price = float(strongest_before.get("price", 0.0))
                wall_qty = float(strongest_before.get("qty", 0.0))
                if wall_price <= 0 or wall_qty <= 0:
                    continue
                after_map = {cls._price_key(float(row.get("price", 0.0))): float(row.get("qty", 0.0)) for row in after_levels}
                after_qty = after_map.get(cls._price_key(wall_price), 0.0)
                tolerance = max(wall_price * 0.00015, 5.0)
                expected_aggressor = "sell" if side == "bid" else "buy"
                hit_quote = sum(
                    float(trade.get("quote_qty", 0.0))
                    for trade in interval_trades
                    if trade.get("aggressor_side") == expected_aggressor
                    and abs(float(trade.get("price", 0.0)) - wall_price) <= tolerance
                )
                wall_quote = wall_price * wall_qty
                if hit_quote >= wall_quote * 0.1 and after_qty >= wall_qty * 0.6:
                    passive_absorption_quote += hit_quote
                    wall_absorption_events += 1
                elif hit_quote >= wall_quote * 0.25 and after_qty <= wall_qty * 0.25:
                    aggressive_sweep_quote += hit_quote
                    wall_sweep_events += 1
                elif after_qty <= wall_qty * 0.25:
                    pull_without_trade_quote += wall_quote
                    wall_pull_without_trade_events += 1

        persistent_walls = cls._collect_wall_runs(rows)
        avg_wall_lifetime = cls._safe_div(
            sum(float(row["lifetime_seconds"]) for row in persistent_walls),
            len(persistent_walls),
        )
        max_wall_lifetime = max((float(row["lifetime_seconds"]) for row in persistent_walls), default=0.0)
        pull_pressure = wall_pull_without_trade_events / max(1, wall_absorption_events + wall_sweep_events + wall_pull_without_trade_events)
        if pull_pressure >= 0.5 and avg_wall_lifetime <= duration_seconds * 0.2:
            spoofing_risk = "high"
        elif pull_pressure >= 0.25:
            spoofing_risk = "medium"
        else:
            spoofing_risk = "low"

        if passive_absorption_quote > aggressive_sweep_quote * 1.2 and passive_absorption_quote > pull_without_trade_quote * 0.5:
            wall_behavior = "absorbing"
        elif pull_without_trade_quote > max(passive_absorption_quote, aggressive_sweep_quote):
            wall_behavior = "pulling_or_spoofing"
        elif aggressive_sweep_quote > passive_absorption_quote * 1.2:
            wall_behavior = "aggressive_sweeps"
        else:
            wall_behavior = "mixed"

        total_snapshots = max(1, len(rows))
        for side in ("bid", "ask"):
            for record in level_activity[side].values():
                record["avg_qty"] = cls._safe_div(float(record["sum_qty"]), float(record["presence_count"]) or 1.0)
                record["presence_ratio"] = cls._safe_div(float(record["presence_count"]), total_snapshots)

        return {
            "snapshot_count": len(rows),
            "sample_duration_seconds": round(duration_seconds, 6),
            "avg_bid_volume_change": round(cls._safe_div(sum(bid_changes), len(bid_changes)), 6),
            "avg_ask_volume_change": round(cls._safe_div(sum(ask_changes), len(ask_changes)), 6),
            "bid_add_rate_per_second": round(cls._safe_div(bid_added, duration_seconds), 6),
            "bid_cancel_rate_per_second": round(cls._safe_div(bid_cancelled, duration_seconds), 6),
            "ask_add_rate_per_second": round(cls._safe_div(ask_added, duration_seconds), 6),
            "ask_cancel_rate_per_second": round(cls._safe_div(ask_cancelled, duration_seconds), 6),
            "best_bid_change_count": best_bid_change_count,
            "best_ask_change_count": best_ask_change_count,
            "best_bid_change_per_minute": round(cls._safe_div(best_bid_change_count * 60, duration_seconds), 6),
            "best_ask_change_per_minute": round(cls._safe_div(best_ask_change_count * 60, duration_seconds), 6),
            "spread_change_count": spread_change_count,
            "mid_price_change_count": mid_price_change_count,
            "avg_wall_lifetime_seconds": round(avg_wall_lifetime, 6),
            "max_wall_lifetime_seconds": round(max_wall_lifetime, 6),
            "wall_pull_events": wall_pull_events,
            "wall_add_events": wall_add_events,
            "wall_absorption_events": wall_absorption_events,
            "wall_sweep_events": wall_sweep_events,
            "wall_pull_without_trade_events": wall_pull_without_trade_events,
            "passive_absorption_quote": round(passive_absorption_quote, 6),
            "aggressive_sweep_quote": round(aggressive_sweep_quote, 6),
            "pull_without_trade_quote": round(pull_without_trade_quote, 6),
            "spoofing_risk": spoofing_risk,
            "wall_behavior": wall_behavior,
            "top_bid_level_activity": cls._top_level_activity(level_activity["bid"]),
            "top_ask_level_activity": cls._top_level_activity(level_activity["ask"]),
            "persistent_walls": persistent_walls,
            "series": series,
        }

    @staticmethod
    def extract_volume_profile(candles: Sequence[Dict], bins: int = 24, window: int = 72) -> Dict:
        rows = list(candles[-window:]) if window > 0 else list(candles)
        if not rows:
            return {
                "window_size": 0,
                "bins": bins,
                "price_low": 0.0,
                "price_high": 0.0,
                "poc_price": 0.0,
                "hvn_prices": [],
                "lvn_prices": [],
            }

        low = min(float(row["low"]) for row in rows)
        high = max(float(row["high"]) for row in rows)
        if high <= low or bins <= 0:
            midpoint = (high + low) / 2 if (high + low) > 0 else 0.0
            return {
                "window_size": len(rows),
                "bins": max(1, bins),
                "price_low": round(low, 6),
                "price_high": round(high, 6),
                "poc_price": round(midpoint, 6),
                "hvn_prices": [round(midpoint, 6)],
                "lvn_prices": [round(midpoint, 6)],
            }

        bin_size = (high - low) / bins
        volume_bins = [0.0 for _ in range(bins)]
        centers = [low + (idx + 0.5) * bin_size for idx in range(bins)]

        for row in rows:
            typical_price = (float(row["high"]) + float(row["low"]) + float(row["close"])) / 3
            raw_idx = int((typical_price - low) / bin_size)
            idx = min(bins - 1, max(0, raw_idx))
            volume_bins[idx] += float(row["volume"])

        poc_idx = max(range(bins), key=lambda idx: volume_bins[idx])
        ranked = sorted(range(bins), key=lambda idx: volume_bins[idx], reverse=True)
        hvn_idx = ranked[: min(3, bins)]
        non_zero_idx = [idx for idx in range(bins) if volume_bins[idx] > 0]
        lvn_ranked = sorted(non_zero_idx, key=lambda idx: volume_bins[idx])
        lvn_idx = lvn_ranked[: min(3, len(lvn_ranked))]

        return {
            "window_size": len(rows),
            "bins": bins,
            "price_low": round(low, 6),
            "price_high": round(high, 6),
            "bin_size": round(bin_size, 6),
            "poc_price": round(centers[poc_idx], 6),
            "hvn_prices": [round(centers[idx], 6) for idx in hvn_idx],
            "lvn_prices": [round(centers[idx], 6) for idx in lvn_idx],
        }

    @classmethod
    def extract_session_profiles(cls, candles: Sequence[Dict], bins: int = 24) -> Dict:
        rows = list(candles)
        if not rows:
            return {
                "session_clock": "UTC_8H",
                "current_session": "unknown",
                "profiles": {},
            }

        session_labels = ("asia", "europe", "us")
        groups: Dict[str, List[Dict]] = {label: [] for label in session_labels}
        for row in rows[-96:]:
            open_time = int(row.get("open_time", 0))
            hour = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).hour
            if 0 <= hour < 8:
                groups["asia"].append(row)
            elif 8 <= hour < 16:
                groups["europe"].append(row)
            else:
                groups["us"].append(row)

        latest_hour = datetime.fromtimestamp(int(rows[-1].get("open_time", 0)) / 1000, tz=timezone.utc).hour
        if 0 <= latest_hour < 8:
            current_session = "asia"
        elif 8 <= latest_hour < 16:
            current_session = "europe"
        else:
            current_session = "us"

        profiles: Dict[str, Dict] = {}
        for label, session_rows in groups.items():
            profile = cls.extract_volume_profile(session_rows, bins=bins, window=0)
            profile["from_open_time"] = int(session_rows[0]["open_time"]) if session_rows else 0
            profile["to_close_time"] = int(session_rows[-1]["close_time"]) if session_rows else 0
            profile["high"] = round(max((float(row["high"]) for row in session_rows), default=0.0), 6)
            profile["low"] = round(min((float(row["low"]) for row in session_rows), default=0.0), 6)
            profiles[label] = profile

        return {
            "session_clock": "UTC_8H",
            "current_session": current_session,
            "profiles": profiles,
        }

    @staticmethod
    def _anchored_vwap(candles: Sequence[Dict]) -> float:
        rows = list(candles)
        if not rows:
            return 0.0
        total_quote = 0.0
        total_volume = 0.0
        for row in rows:
            typical_price = (float(row["high"]) + float(row["low"]) + float(row["close"])) / 3
            volume = float(row["volume"])
            total_quote += typical_price * volume
            total_volume += volume
        return 0.0 if total_volume == 0 else total_quote / total_volume

    @staticmethod
    def _recent_swings(candles: Sequence[Dict], pivot: int = 2, lookback: int = 72) -> List[Tuple[str, int]]:
        rows = list(candles[-lookback:])
        if len(rows) < pivot * 2 + 1:
            return []

        swings: List[Tuple[str, int]] = []
        offset = len(candles) - len(rows)
        for idx in range(pivot, len(rows) - pivot):
            center = rows[idx]
            highs = [float(row["high"]) for row in rows[idx - pivot : idx + pivot + 1]]
            lows = [float(row["low"]) for row in rows[idx - pivot : idx + pivot + 1]]
            if float(center["high"]) >= max(highs):
                swings.append(("recent_swing_high", offset + idx))
            if float(center["low"]) <= min(lows):
                swings.append(("recent_swing_low", offset + idx))
        return swings

    @classmethod
    def extract_anchored_profiles(cls, candles: Sequence[Dict], bins: int = 24) -> List[Dict]:
        rows = list(candles)
        if not rows:
            return []

        anchors = cls._recent_swings(rows)
        deduped: List[Tuple[str, int]] = []
        seen_types: set[str] = set()
        for anchor_type, idx in reversed(anchors):
            if anchor_type in seen_types:
                continue
            deduped.append((anchor_type, idx))
            seen_types.add(anchor_type)
            if len(deduped) >= 2:
                break
        deduped.reverse()

        output: List[Dict] = []
        current_price = float(rows[-1].get("close", 0.0))
        for anchor_type, idx in deduped:
            anchor_rows = rows[idx:]
            anchor_row = rows[idx]
            profile = cls.extract_volume_profile(anchor_rows, bins=bins, window=0)
            vwap = cls._anchored_vwap(anchor_rows)
            distance_bps = cls._safe_div(current_price - vwap, vwap) * 10000 if vwap > 0 else 0.0
            output.append(
                {
                    "anchor_type": anchor_type,
                    "anchor_open_time": int(anchor_row.get("open_time", 0)),
                    "anchor_price": round(float(anchor_row.get("high" if "high" in anchor_type else "low", anchor_row.get("close", 0.0))), 6),
                    "window_size": len(anchor_rows),
                    "anchored_vwap": round(vwap, 6),
                    "distance_to_vwap_bps": round(distance_bps, 6),
                    "poc_price": profile.get("poc_price", 0.0),
                    "hvn_prices": profile.get("hvn_prices", []),
                    "lvn_prices": profile.get("lvn_prices", []),
                }
            )
        return output

    @staticmethod
    def _parse_iso_timestamp(value: str) -> int:
        try:
            return int(datetime.fromisoformat(value).timestamp() * 1000)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _session_name(timestamp_ms: int) -> str:
        hour = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).hour
        if 0 <= hour < 8:
            return "asia"
        if 8 <= hour < 16:
            return "europe"
        return "us"

    @classmethod
    def extract_session_context(cls, candles: Sequence[Dict], funding: Dict, stats_24h: Dict) -> Dict:
        rows = list(candles)
        if not rows:
            return {
                "current_session": "unknown",
                "session_high": 0.0,
                "session_low": 0.0,
                "day_high": round(float(stats_24h.get("high_price", 0.0)), 6),
                "day_low": round(float(stats_24h.get("low_price", 0.0)), 6),
                "funding_countdown_seconds": 0.0,
                "funding_countdown_label": "unknown",
            }

        current_session = cls._session_name(int(rows[-1].get("open_time", 0)))
        session_rows: List[Dict] = []
        for row in reversed(rows):
            if cls._session_name(int(row.get("open_time", 0))) != current_session:
                break
            session_rows.append(row)
        session_rows.reverse()

        day_window = rows[-96:] if len(rows) >= 96 else rows
        session_high = max((float(row.get("high", 0.0)) for row in session_rows), default=0.0)
        session_low = min((float(row.get("low", 0.0)) for row in session_rows), default=0.0)
        day_high = float(stats_24h.get("high_price") or max((float(row.get("high", 0.0)) for row in day_window), default=0.0))
        day_low = float(stats_24h.get("low_price") or min((float(row.get("low", 0.0)) for row in day_window), default=0.0))
        latest_close_ms = int(rows[-1].get("close_time", 0))
        next_funding_ms = cls._parse_iso_timestamp(str(funding.get("next_funding_time") or ""))
        countdown_seconds = max(0.0, (next_funding_ms - latest_close_ms) / 1000) if next_funding_ms else 0.0
        if countdown_seconds >= 3600:
            funding_countdown_label = f"{int(countdown_seconds // 3600)}h {int((countdown_seconds % 3600) // 60)}m"
        elif countdown_seconds >= 60:
            funding_countdown_label = f"{int(countdown_seconds // 60)}m {int(countdown_seconds % 60)}s"
        else:
            funding_countdown_label = f"{int(countdown_seconds)}s"

        return {
            "current_session": current_session,
            "session_high": round(session_high, 6),
            "session_low": round(session_low, 6),
            "session_open_time": int(session_rows[0].get("open_time", 0)) if session_rows else 0,
            "session_close_time": int(session_rows[-1].get("close_time", 0)) if session_rows else 0,
            "day_high": round(day_high, 6),
            "day_low": round(day_low, 6),
            "funding_countdown_seconds": round(countdown_seconds, 6),
            "funding_countdown_label": funding_countdown_label,
        }

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

    @staticmethod
    def extract_basis(funding: Dict) -> Dict:
        mark_price = float(funding.get("mark_price", 0.0))
        index_price = float(funding.get("index_price", 0.0))
        if index_price <= 0:
            return {
                "basis_abs": 0.0,
                "basis_bps": 0.0,
                "structure": "unknown",
            }

        basis_abs = mark_price - index_price
        basis_bps = basis_abs / index_price * 10000
        if basis_abs > 0:
            structure = "contango"
        elif basis_abs < 0:
            structure = "backwardation"
        else:
            structure = "flat"

        return {
            "basis_abs": round(basis_abs, 6),
            "basis_bps": round(basis_bps, 6),
            "structure": structure,
        }

    @staticmethod
    def extract_options_iv_placeholder() -> Dict:
        return {
            "available": False,
            "atm_iv": None,
            "skew": None,
            "reason": "options_iv_data_not_integrated",
        }

    @staticmethod
    def extract_cross_exchange_funding_spread(cross_exchange_funding: Dict) -> Dict:
        rates: List[Dict] = []
        for exchange, payload in (cross_exchange_funding or {}).items():
            if payload.get("available"):
                rates.append(
                    {
                        "exchange": exchange,
                        "funding_rate": float(payload.get("funding_rate", 0.0)),
                    }
                )

        if len(rates) < 2:
            return {
                "available_count": len(rates),
                "max_spread_bps": 0.0,
                "highest_exchange": "unknown",
                "lowest_exchange": "unknown",
                "signal": "insufficient_data",
            }

        highest = max(rates, key=lambda x: x["funding_rate"])
        lowest = min(rates, key=lambda x: x["funding_rate"])
        spread_bps = (highest["funding_rate"] - lowest["funding_rate"]) * 10000

        if spread_bps >= 3:
            signal = "significant_spread"
        elif spread_bps >= 1:
            signal = "moderate_spread"
        else:
            signal = "tight_spread"

        return {
            "available_count": len(rates),
            "max_spread_bps": round(spread_bps, 6),
            "highest_exchange": highest["exchange"],
            "lowest_exchange": lowest["exchange"],
            "signal": signal,
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
        for label in ("1m", "5m", "15m"):
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
