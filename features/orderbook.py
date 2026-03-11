"""Orderbook feature extraction mixin."""

from typing import Any, Dict, List, Sequence

from ._base import FeatureBase


class OrderbookMixin(FeatureBase):
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
