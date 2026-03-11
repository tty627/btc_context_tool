from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, List, Sequence

_DEFAULT_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class KlineChartGenerator:
    SESSION_COLORS = {
        "asia": "#f4f1de",
        "europe": "#e9f5db",
        "us": "#fdf0d5",
    }

    LEVEL_COLORS = {
        "range": "#c65d07",
        "profile": "#457b9d",
        "anchored_vwap": "#8d6a9f",
        "ema": "#6a994e",
        "liquidity": "#b56576",
    }

    def __init__(self, output_dir: Path, bars_by_timeframe: Dict[str, int]) -> None:
        self.output_dir = Path(output_dir)
        self.bars_by_timeframe = bars_by_timeframe

    @staticmethod
    def _ema(values: Sequence[float], period: int) -> List[float]:
        if not values:
            return []
        alpha = 2 / (period + 1)
        result = [values[0]]
        for value in values[1:]:
            result.append(alpha * value + (1 - alpha) * result[-1])
        return result

    @staticmethod
    def _linear_trend(values: Sequence[float]) -> List[float]:
        n = len(values)
        if n <= 1:
            return list(values)
        sum_x = n * (n - 1) / 2
        sum_y = float(sum(values))
        sum_xx = (n - 1) * n * (2 * n - 1) / 6
        sum_xy = sum(i * value for i, value in enumerate(values))
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return [values[0]] * n
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        return [slope * i + intercept for i in range(n)]

    @staticmethod
    def _format_time(open_time_ms: int) -> str:
        dt = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
        return dt.strftime("%m-%d %H:%M")

    @staticmethod
    def _timestamp_to_x(timestamp_ms: int, candles: Sequence[Dict]) -> float:
        if not candles:
            return 0.0
        start_ms = int(candles[0]["open_time"])
        end_ms = int(candles[-1]["close_time"])
        if end_ms <= start_ms:
            return float(len(candles) - 1)
        ratio = (timestamp_ms - start_ms) / (end_ms - start_ms)
        ratio = max(0.0, min(1.0, ratio))
        return ratio * (len(candles) - 1)

    @staticmethod
    def _session_name(open_time_ms: int) -> str:
        hour = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).hour
        if 0 <= hour < 8:
            return "asia"
        if 8 <= hour < 16:
            return "europe"
        return "us"

    @staticmethod
    def _price_precision(price: float) -> int:
        absolute = abs(price)
        if absolute >= 1000:
            return 1
        if absolute >= 1:
            return 3
        return 6

    def generate(
        self,
        symbol: str,
        klines_by_timeframe: Dict[str, Sequence[Dict]],
        context: Dict | None = None,
    ) -> Dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        chart_files: Dict[str, str] = {}
        for timeframe in self.bars_by_timeframe:
            candles = list(klines_by_timeframe.get(timeframe, []))
            if not candles:
                continue
            bars = min(len(candles), self.bars_by_timeframe[timeframe])
            window = candles[-bars:]
            output_path = self.output_dir / f"{symbol}_{timeframe}.png"
            self._plot_single(symbol, timeframe, window, output_path, context or {})
            chart_files[timeframe] = str(output_path.resolve())
        return chart_files

    @staticmethod
    def _chart_spec(timeframe: str) -> Dict:
        if timeframe == "4h":
            return {
                "mode": "structure",
                "title": "Structure",
                "oi_period": "1h",
                "show_micro": False,
                "show_plan": False,
                "label_limit": 6,
                "show_trade_clusters": False,
            }
        if timeframe == "15m":
            return {
                "mode": "transition",
                "title": "Transition",
                "oi_period": "5m",
                "show_micro": False,
                "show_plan": False,
                "label_limit": 6,
                "show_trade_clusters": False,
            }
        return {
            "mode": "execution",
            "title": "Execution",
            "oi_period": "5m",
            "show_micro": True,
            "show_plan": False,
            "label_limit": 5,
            "show_trade_clusters": True,
        }

    @staticmethod
    def _append_reference_level(levels: List[Dict], name: str, price: float, source: str, priority: int) -> None:
        try:
            level_price = float(price)
        except (TypeError, ValueError):
            return
        if level_price <= 0:
            return
        levels.append(
            {
                "name": name,
                "price": level_price,
                "source": source,
                "priority": priority,
            }
        )

    def _collect_reference_levels(self, context: Dict) -> List[Dict]:
        levels: List[Dict] = []
        recent_4h_range = context.get("recent_4h_range", {})
        self._append_reference_level(levels, "4H High", recent_4h_range.get("high"), "range", 1)
        self._append_reference_level(levels, "4H Low", recent_4h_range.get("low"), "range", 1)

        volume_profile = context.get("volume_profile", {})
        self._append_reference_level(levels, "POC", volume_profile.get("poc_price"), "profile", 1)
        for idx, price in enumerate(volume_profile.get("hvn_prices", [])[:3], start=1):
            self._append_reference_level(levels, f"HVN{idx}", price, "profile", 2)
        for idx, price in enumerate(volume_profile.get("lvn_prices", [])[:3], start=1):
            self._append_reference_level(levels, f"LVN{idx}", price, "profile", 3)

        anchor_names = {
            "recent_swing_high": "AVWAP-H",
            "recent_swing_low": "AVWAP-L",
        }
        for idx, anchor in enumerate(volume_profile.get("anchored_profiles", []), start=1):
            if not isinstance(anchor, dict):
                continue
            anchor_name = anchor_names.get(anchor.get("anchor_type"), f"AVWAP{idx}")
            self._append_reference_level(
                levels,
                anchor_name,
                anchor.get("anchored_vwap"),
                "anchored_vwap",
                2,
            )

        timeframes = context.get("timeframes", {})
        for tf, priority in (("1h", 2), ("4h", 3)):
            ema = timeframes.get(tf, {}).get("ema", {})
            if not isinstance(ema, dict):
                continue
            self._append_reference_level(levels, f"{tf.upper()} EMA25", ema.get("25"), "ema", priority)
            self._append_reference_level(levels, f"{tf.upper()} EMA99", ema.get("99"), "ema", 3)
        return levels

    def _add_session_backgrounds(self, ax, candles: Sequence[Dict]) -> None:
        if not candles:
            return
        spans: List[tuple[int, int, str]] = []
        current_label = self._session_name(int(candles[0]["open_time"]))
        start_idx = 0
        for idx, row in enumerate(candles[1:], start=1):
            label = self._session_name(int(row["open_time"]))
            if label == current_label:
                continue
            spans.append((start_idx, idx - 1, current_label))
            start_idx = idx
            current_label = label
        spans.append((start_idx, len(candles) - 1, current_label))

        for start_idx, end_idx, label in spans:
            ax.axvspan(start_idx - 0.5, end_idx + 0.5, color=self.SESSION_COLORS[label], alpha=0.2, zorder=0)

    def _select_labelled_levels(
        self,
        levels: Sequence[Dict],
        current_price: float,
        visible_low: float,
        visible_high: float,
        limit: int,
    ) -> set[str]:
        span = max(1.0, visible_high - visible_low)
        min_gap = max(span * 0.035, current_price * 0.0012, 18.0)
        sorted_levels = sorted(
            [level for level in levels if visible_low <= float(level["price"]) <= visible_high],
            key=lambda level: (int(level.get("priority", 3)), abs(float(level["price"]) - current_price)),
        )
        labels: set[str] = set()
        placed_prices: List[float] = []
        for level in sorted_levels:
            price = float(level["price"])
            if any(abs(price - placed) < min_gap for placed in placed_prices):
                continue
            labels.add(str(level["name"]))
            placed_prices.append(price)
            if len(labels) >= limit:
                break
        return labels

    def _draw_reference_levels(
        self,
        ax,
        levels: Sequence[Dict],
        current_price: float,
        visible_low: float,
        visible_high: float,
        label_limit: int,
    ) -> None:
        visible_levels = [level for level in levels if visible_low <= float(level["price"]) <= visible_high]
        labels = self._select_labelled_levels(visible_levels, current_price, visible_low, visible_high, label_limit)
        for level in visible_levels:
            price = float(level["price"])
            source = str(level.get("source", "profile"))
            color = self.LEVEL_COLORS.get(source, "#5c677d")
            priority = int(level.get("priority", 3))
            linestyle = "-" if priority == 1 else "--" if priority == 2 else ":"
            linewidth = 1.1 if priority == 1 else 0.9 if priority == 2 else 0.7
            alpha = 0.85 if priority == 1 else 0.55 if priority == 2 else 0.35
            ax.axhline(price, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
            if str(level["name"]) in labels:
                ax.text(
                    1.006,
                    price,
                    str(level["name"]),
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=7 if priority < 3 else 6,
                    color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.2},
                )

    def _draw_plan_zones(self, ax, candles: Sequence[Dict], deployment_context: Dict) -> None:
        plan = deployment_context.get("plan_zones", {})
        if not plan:
            return
        zone_styles = {
            "entry": ("#4caf50", 0.12, "Entry"),
            "invalidation": ("#d62828", 0.11, "Invalid"),
            "take_profit": ("#457b9d", 0.1, "TP"),
        }
        label_x = len(candles) + 0.3
        for key, (color, alpha, label) in zone_styles.items():
            zone = plan.get(key, {})
            zone_low = float(zone.get("zone_low", 0.0))
            zone_high = float(zone.get("zone_high", 0.0))
            if zone_low <= 0 or zone_high <= zone_low:
                continue
            ax.axhspan(zone_low, zone_high, color=color, alpha=alpha, zorder=0.2)
            ax.text(
                label_x,
                (zone_low + zone_high) / 2,
                label,
                fontsize=7,
                color=color,
                ha="left",
                va="center",
                clip_on=False,
            )

    def _draw_trade_clusters(self, ax, trade_flow: Dict, current_price: float, candles_len: int) -> None:
        clusters = trade_flow.get("large_trade_clusters", [])[:3]
        absorption_zones = trade_flow.get("absorption_zones", [])
        absorption_lookup = {
            (round(float(zone.get("zone_low", 0.0)), 2), round(float(zone.get("zone_high", 0.0)), 2)): zone
            for zone in absorption_zones
        }
        label_x = candles_len + 0.3
        for idx, cluster in enumerate(clusters, start=1):
            zone_low = float(cluster.get("zone_low", 0.0))
            zone_high = float(cluster.get("zone_high", 0.0))
            if zone_low <= 0 or zone_high <= 0:
                continue
            if abs(((zone_low + zone_high) / 2) - current_price) / max(current_price, 1.0) > 0.015:
                continue
            color = "#5c677d"
            alpha = 0.12
            label = f"Cluster {idx}"
            key = (round(zone_low, 2), round(zone_high, 2))
            absorption = absorption_lookup.get(key)
            if absorption:
                color = "#2a9d8f"
                alpha = 0.18
                label = f"Absorption {idx}"
            ax.axhspan(zone_low, zone_high, color=color, alpha=alpha, zorder=0.15)
            ax.text(
                label_x,
                (zone_low + zone_high) / 2,
                label,
                fontsize=6.5,
                color=color,
                ha="left",
                va="center",
                clip_on=False,
            )

    def _draw_liquidity_zones(self, ax, context: Dict, current_price: float) -> None:
        recent_4h_range = context.get("recent_4h_range", {})
        for zone_key in ("high_sweep_zone", "low_sweep_zone"):
            zone = recent_4h_range.get(zone_key, {})
            if not isinstance(zone, dict):
                continue
            zone_low = float(zone.get("zone_low", 0.0))
            zone_high = float(zone.get("zone_high", 0.0))
            if zone_low <= 0 or zone_high <= zone_low:
                continue
            ax.axhspan(zone_low, zone_high, color="#dce3ea", alpha=0.08, zorder=0.08)

        for zone in context.get("liquidation_heatmap", {}).get("zones", []):
            zone_low = float(zone.get("zone_low", 0.0))
            zone_high = float(zone.get("zone_high", 0.0))
            if zone_low <= 0 or zone_high <= 0:
                continue
            if abs(((zone_low + zone_high) / 2) - current_price) / max(current_price, 1.0) > 0.02:
                continue
            ax.axhspan(zone_low, zone_high, color="#d2d8e0", alpha=0.08, zorder=0.1)

    def _plot_line_series(self, ax, candles: Sequence[Dict], points: Sequence[Dict], value_key: str, color: str, label: str) -> None:
        if not points:
            return
        x_values = [self._timestamp_to_x(int(point.get("timestamp", 0)), candles) for point in points]
        y_values = [float(point.get(value_key, 0.0)) for point in points]
        ax.plot(x_values, y_values, color=color, linewidth=1.2, label=label)

    def _add_badges(self, ax, badges: Sequence[tuple[str, str]]) -> None:
        x_cursor = 0.01
        for text, color in badges:
            ax.text(
                x_cursor,
                0.985,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#1f2933",
                bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
            )
            x_cursor += min(0.22, 0.011 * len(text) + 0.05)

    def _plot_price_panel(self, ax, candles: Sequence[Dict], timeframe: str, context: Dict, spec: Dict) -> None:
        opens = [row["open"] for row in candles]
        highs = [row["high"] for row in candles]
        lows = [row["low"] for row in candles]
        closes = [row["close"] for row in candles]
        x_values = list(range(len(candles)))

        self._add_session_backgrounds(ax, candles)

        body_floor = max((max(highs) - min(lows)) * 0.001, 0.5)
        for idx, row in enumerate(candles):
            open_price = row["open"]
            close_price = row["close"]
            color = "#0f9d58" if close_price >= open_price else "#d24c36"
            ax.vlines(idx, row["low"], row["high"], color=color, linewidth=1.0, alpha=0.95)
            lower = min(open_price, close_price)
            height = max(abs(close_price - open_price), body_floor)
            ax.add_patch(
                Rectangle(
                    (idx - 0.31, lower),
                    0.62,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.7,
                    alpha=0.88,
                )
            )

        ema7 = self._ema(closes, 7)
        ema25 = self._ema(closes, 25)
        ema99 = self._ema(closes, 99)
        ax.plot(x_values, ema7, color="#d97706", linewidth=1.2, label="EMA7")
        ax.plot(x_values, ema25, color="#1f6f8b", linewidth=1.2, label="EMA25")
        ax.plot(x_values, ema99, color="#3d5a40", linewidth=1.2, label="EMA99")
        current_price = float(context.get("price", closes[-1]))
        visible_low = min(lows)
        visible_high = max(highs)
        reference_levels = self._collect_reference_levels(context)
        self._draw_liquidity_zones(ax, context, current_price)
        if spec["show_trade_clusters"]:
            self._draw_trade_clusters(ax, context.get("trade_flow", {}), current_price, len(candles))
        self._draw_reference_levels(
            ax=ax,
            levels=reference_levels,
            current_price=current_price,
            visible_low=visible_low,
            visible_high=visible_high,
            label_limit=int(spec["label_limit"]),
        )

        if spec["mode"] in ("transition", "execution"):
            session_context = context.get("session_context", {})
            session_high = float(session_context.get("session_high", 0.0))
            session_low = float(session_context.get("session_low", 0.0))
            if session_high > 0:
                ax.axhline(session_high, color="#9c6644", linestyle=":", linewidth=0.8, alpha=0.55)
            if session_low > 0:
                ax.axhline(session_low, color="#718355", linestyle=":", linewidth=0.8, alpha=0.55)

        bar_state = context.get("timeframes", {}).get(timeframe, {}).get("bar_state", {})
        if bar_state and not bar_state.get("bar_closed", True):
            ax.axvspan(len(candles) - 1.5, len(candles) - 0.5, color="#ffe066", alpha=0.12)

        ax.set_title(f"{context.get('symbol', 'BTCUSDT')} {timeframe} {spec['title']} Chart", fontsize=12, pad=10)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
        ax.legend(loc="upper left", fontsize=7, ncol=4)
        ax.set_xlim(-1, len(candles) + 4)

    def _plot_derivatives_panel(self, ax, candles: Sequence[Dict], context: Dict, spec: Dict) -> None:
        open_interest = context.get("open_interest_trend", {})
        oi_period = spec["oi_period"]
        oi_points = open_interest.get("periods", {}).get(oi_period, {}).get("series", [])
        self._plot_line_series(ax, candles, oi_points, "open_interest", "#174b63", f"OI {oi_period}")
        ax.set_ylabel("OI", fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)

        ratio_history = context.get("long_short_ratio", {}).get("global_account", {}).get("history", [])
        ax_ratio = ax.twinx()
        self._plot_line_series(ax_ratio, candles, ratio_history, "ratio", "#b56576", "Global L/S")
        ax_ratio.set_ylabel("L/S", fontsize=8, color="#b56576")
        ax_ratio.tick_params(axis="y", labelsize=8, colors="#b56576")

        lines_left, labels_left = ax.get_legend_handles_labels()
        lines_right, labels_right = ax_ratio.get_legend_handles_labels()
        if lines_left or lines_right:
            ax.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right", fontsize=7)

    def _plot_micro_panel(self, ax, candles: Sequence[Dict], context: Dict) -> None:
        trade_flow = context.get("trade_flow", {})
        orderbook_dynamics = context.get("orderbook_dynamics", {})
        self._plot_line_series(ax, candles, trade_flow.get("cvd_path", []), "cvd_qty", "#2a9d8f", "CVD")
        ax.set_ylabel("CVD", fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)

        ax_imbalance = ax.twinx()
        self._plot_line_series(ax_imbalance, candles, orderbook_dynamics.get("series", []), "imbalance", "#c65d07", "OB Imb")
        ax_imbalance.set_ylabel("OB", fontsize=8, color="#c65d07")
        ax_imbalance.tick_params(axis="y", labelsize=8, colors="#c65d07")

        lines_left, labels_left = ax.get_legend_handles_labels()
        lines_right, labels_right = ax_imbalance.get_legend_handles_labels()
        if lines_left or lines_right:
            ax.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right", fontsize=7)

    def _plot_volume_panel(self, ax, candles: Sequence[Dict]) -> None:
        for idx, row in enumerate(candles):
            color = "#0f9d58" if row["close"] >= row["open"] else "#d24c36"
            ax.bar(idx, row["volume"], color=color, width=0.66, alpha=0.58)
        ax.set_ylabel("Vol", fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)

        tick_count = min(8, len(candles))
        step = max(1, len(candles) // max(1, tick_count - 1))
        ticks = list(range(0, len(candles), step))
        if ticks[-1] != len(candles) - 1:
            ticks.append(len(candles) - 1)
        labels = [self._format_time(int(candles[idx]["open_time"])) for idx in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)

    def _plot_single(
        self,
        symbol: str,
        timeframe: str,
        candles: Sequence[Dict],
        output_path: Path,
        context: Dict,
    ) -> None:
        spec = self._chart_spec(timeframe)
        panel_count = 4 if spec["show_micro"] else 3
        height_ratios = [5, 1.4, 1.1, 1.0] if spec["show_micro"] else [5, 1.4, 1.0]
        fig, axes = plt.subplots(
            panel_count,
            1,
            figsize=(15, 9 if spec["show_micro"] else 8),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        axes_list = list(axes) if panel_count > 1 else [axes]
        fig.patch.set_facecolor("#f8f5f0")
        for axis in axes_list:
            axis.set_facecolor("#fffdf8")

        self._plot_price_panel(axes_list[0], candles, timeframe, context, spec)
        self._plot_derivatives_panel(axes_list[1], candles, context, spec)
        if spec["show_micro"]:
            self._plot_micro_panel(axes_list[2], candles, context)
            self._plot_volume_panel(axes_list[3], candles)
        else:
            self._plot_volume_panel(axes_list[2], candles)

        plt.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
