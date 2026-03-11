"""Derivatives feature extraction mixin."""

from typing import Dict, List, Sequence, Tuple

from ._base import FeatureBase


class DerivativesMixin(FeatureBase):
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
