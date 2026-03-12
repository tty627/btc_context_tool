"""BTC execution-grade chart generator.

Design principles
─────────────────
• Three-tier level hierarchy: primary (thick solid) → secondary (thin dashed)
  → tertiary (dotted, dim).  Same object = same colour across all charts.
• Zone-box mechanism: levels within 0.2 % of each other are merged into a
  shaded rectangle with a single label to prevent right-axis clutter.
• Execution overlay (15m / 5m only): position management lines
  (Entry / BE / SL / T0 / T1 / T2) and action lines
  (Hold above / Reduce near / Add above / Invalidate below).
• OI-state badge and Spot/Perp state badge on every price panel.
• Colour semantics are defined once in _CLR and _TIER at module level.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_DEFAULT_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ─── colour semantics (same object = same colour in every chart) ────────────
_CLR = {
    # reference levels
    "range":         "#c65d07",   # orange  — 4H High / 4H Low
    "profile":       "#1d5c8b",   # blue    — POC / HVN / LVN
    "anchored_vwap": "#7b4fa6",   # purple  — AVWAP
    "ema":           "#3a7d44",   # green   — EMA25 / EMA99
    "liquidity":     "#b56576",   # coral   — liquidity sweep zones
    "session":       "#9c6644",   # tan     — session high / low
    # position management lines
    "entry":         "#2ecc40",   # bright green
    "be":            "#0074d9",   # blue
    "sl":            "#ff4136",   # red
    "t0":            "#e6ac00",   # amber   — reduce / breakeven zone
    "t1":            "#5dbbdb",   # sky     — structure target
    "t2":            "#01c853",   # lime    — extension target
    # execution action lines
    "hold":          "#2ecc40",
    "reduce":        "#ff851b",
    "add":           "#0074d9",
    "invalidate":    "#ff4136",
    # micro structure
    "cluster":       "#5c677d",
    "absorption":    "#2a9d8f",
}

# Tier rendering styles: (linestyle, linewidth, alpha, label_fontsize)
_TIER: Dict[int, Tuple[str, float, float, float]] = {
    1: ("-",  1.7, 0.90, 8.0),   # primary   — thick solid, bright
    2: ("--", 1.0, 0.58, 7.0),   # secondary — thin dashed
    3: (":",  0.6, 0.28, 6.0),   # tertiary  — dotted, dim
}

# OI state → (badge label, background colour)
_OI_STATE: Dict[str, Tuple[str, str]] = {
    "price_up_oi_up":     ("OI: new longs ↑",      "#c8f7c5"),
    "price_up_oi_down":   ("OI: short covering ↑",  "#d6eaf8"),
    "price_down_oi_up":   ("OI: new shorts ↓",      "#fde3e3"),
    "price_down_oi_down": ("OI: long unwind ↓",      "#fff3cd"),
    "flat":               ("OI: flat",               "#ececec"),
}


class KlineChartGenerator:
    SESSION_COLORS = {
        "asia":   "#f4f1de",
        "europe": "#e9f5db",
        "us":     "#fdf0d5",
    }
    # Keep for external callers that reference LEVEL_COLORS
    LEVEL_COLORS = {k: _CLR[k] for k in ("range", "profile", "anchored_vwap", "ema", "liquidity")}

    def __init__(self, output_dir: Path, bars_by_timeframe: Dict[str, int]) -> None:
        self.output_dir = Path(output_dir)
        self.bars_by_timeframe = bars_by_timeframe

    # ─── maths helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ema(values: Sequence[float], period: int) -> List[float]:
        if not values:
            return []
        alpha = 2 / (period + 1)
        result = [values[0]]
        for v in values[1:]:
            result.append(alpha * v + (1 - alpha) * result[-1])
        return result

    @staticmethod
    def _linear_trend(values: Sequence[float]) -> List[float]:
        n = len(values)
        if n <= 1:
            return list(values)
        sx = n * (n - 1) / 2
        sy = float(sum(values))
        sxx = (n - 1) * n * (2 * n - 1) / 6
        sxy = sum(i * v for i, v in enumerate(values))
        d = n * sxx - sx * sx
        if d == 0:
            return [values[0]] * n
        m = (n * sxy - sx * sy) / d
        b = (sy - m * sx) / n
        return [m * i + b for i in range(n)]

    @staticmethod
    def _format_time(open_time_ms: int) -> str:
        return datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")

    @staticmethod
    def _timestamp_to_x(ts_ms: int, candles: Sequence[Dict]) -> float:
        if not candles:
            return 0.0
        s = int(candles[0]["open_time"])
        e = int(candles[-1]["close_time"])
        if e <= s:
            return float(len(candles) - 1)
        r = (ts_ms - s) / (e - s)
        return max(0.0, min(1.0, r)) * (len(candles) - 1)

    @staticmethod
    def _session_name(ts_ms: int) -> str:
        h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
        if h < 8:
            return "asia"
        if h < 16:
            return "europe"
        return "us"

    @staticmethod
    def _price_precision(price: float) -> int:
        a = abs(price)
        if a >= 1000:
            return 1
        if a >= 1:
            return 3
        return 6

    # ─── level collection ────────────────────────────────────────────────────

    @staticmethod
    def _append_level(levels: List[Dict], name: str, price, source: str, priority: int) -> None:
        try:
            p = float(price)
        except (TypeError, ValueError):
            return
        if p > 0:
            levels.append({"name": name, "price": p, "source": source, "priority": priority})

    def _collect_reference_levels(self, context: Dict) -> List[Dict]:
        levels: List[Dict] = []
        r4h = context.get("recent_4h_range", {})
        self._append_level(levels, "4H High", r4h.get("high"), "range", 1)
        self._append_level(levels, "4H Low",  r4h.get("low"),  "range", 1)

        vp = context.get("volume_profile", {})
        self._append_level(levels, "POC", vp.get("poc_price"), "profile", 1)
        for i, p in enumerate(vp.get("hvn_prices", [])[:3], 1):
            self._append_level(levels, f"HVN{i}", p, "profile", 2)
        for i, p in enumerate(vp.get("lvn_prices", [])[:3], 1):
            self._append_level(levels, f"LVN{i}", p, "profile", 3)

        anchor_names = {"recent_swing_high": "AVWAP-H", "recent_swing_low": "AVWAP-L"}
        for i, a in enumerate(vp.get("anchored_profiles", []), 1):
            if isinstance(a, dict):
                self._append_level(
                    levels,
                    anchor_names.get(a.get("anchor_type"), f"AVWAP{i}"),
                    a.get("anchored_vwap"),
                    "anchored_vwap",
                    2,
                )

        tfs = context.get("timeframes", {})
        for tf, pri in (("1h", 2), ("4h", 3)):
            ema = tfs.get(tf, {}).get("ema", {})
            if isinstance(ema, dict):
                self._append_level(levels, f"{tf.upper()} EMA25", ema.get("25"), "ema", pri)
                self._append_level(levels, f"{tf.upper()} EMA99", ema.get("99"), "ema", 3)
        return levels

    # ─── zone-box builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_zone_boxes(
        levels: List[Dict],
        current_price: float,
        min_gap_pct: float = 0.002,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Group levels within `min_gap_pct` of each other into zone boxes.

        Returns (standalone_levels, zone_boxes).
        """
        if not levels:
            return [], []
        gap = max(current_price * min_gap_pct, 10.0)
        sorted_lvls = sorted(levels, key=lambda x: float(x["price"]))
        groups: List[List[Dict]] = []
        grp = [sorted_lvls[0]]
        for lv in sorted_lvls[1:]:
            if float(lv["price"]) - float(grp[-1]["price"]) <= gap:
                grp.append(lv)
            else:
                groups.append(grp)
                grp = [lv]
        groups.append(grp)

        standalone: List[Dict] = []
        zones: List[Dict] = []
        for g in groups:
            if len(g) == 1:
                standalone.append(g[0])
            else:
                lo = float(g[0]["price"])
                hi = float(g[-1]["price"])
                best = min(g, key=lambda x: int(x.get("priority", 3)))
                pad = max((hi - lo) * 0.15, 2.0)
                zones.append({
                    "zone_low":  lo - pad,
                    "zone_high": hi + pad,
                    "center":    (lo + hi) / 2,
                    "label":     " / ".join(lv["name"] for lv in g),
                    "source":    best.get("source", "profile"),
                    "priority":  int(best.get("priority", 2)),
                    "members":   g,
                })
        return standalone, zones

    # ─── execution overlay helpers ───────────────────────────────────────────

    @staticmethod
    def _get_position_lines(context: Dict) -> List[Dict]:
        """Extract Entry / BE / SL / T0 / T1 / T2 from account_positions + position_sizing."""
        lines: List[Dict] = []
        acc = context.get("account_positions", {})
        sym = acc.get("symbol_position") if acc.get("available") else None
        if not sym or abs(float(sym.get("position_amt", 0) or 0)) == 0:
            return lines

        side   = sym.get("side", "long")
        entry  = float(sym.get("entry_price", 0) or 0)
        liq    = float(sym.get("liquidation_price", 0) or 0)

        if entry > 0:
            lines.append({"label": "Entry", "price": entry, "style": "entry"})
            lines.append({"label": "BE",    "price": entry, "style": "be"})
        if liq > 0:
            lines.append({"label": "Liq",   "price": liq,   "style": "sl"})

        sizing = context.get("position_sizing", {})
        if sizing and sizing.get("available"):
            ref = sizing.get("reference_levels", {}).get(side, {})
            sl  = float(ref.get("stop_loss", 0) or 0)
            tp1 = float(ref.get("tp1", 0) or 0)
            tp2 = float(ref.get("tp2", 0) or 0)
            atr = float(sizing.get("atr", 0) or 0)
            if sl > 0:
                lines.append({"label": "SL",  "price": sl,  "style": "sl"})
            # T0 = entry ± 1× ATR (conservative first target / reduce zone)
            if entry > 0 and atr > 0:
                t0 = entry + atr if side == "long" else entry - atr
                lines.append({"label": "T0",  "price": t0,  "style": "t0"})
            if tp1 > 0:
                lines.append({"label": "T1",  "price": tp1, "style": "t1"})
            if tp2 > 0:
                lines.append({"label": "T2",  "price": tp2, "style": "t2"})
        return lines

    @staticmethod
    def _get_action_lines(
        levels: List[Dict],
        current_price: float,
    ) -> List[Dict]:
        """Derive Hold / Reduce / Add / Invalidate action lines from key levels."""
        if not levels or current_price <= 0:
            return []
        pri2 = [lv for lv in levels if int(lv.get("priority", 3)) <= 2]
        supports    = sorted([lv for lv in pri2 if float(lv["price"]) < current_price],
                             key=lambda x: float(x["price"]), reverse=True)
        resistances = sorted([lv for lv in pri2 if float(lv["price"]) > current_price],
                             key=lambda x: float(x["price"]))
        actions: List[Dict] = []
        if supports:
            p = float(supports[0]["price"])
            actions.append({"label": f"Hold > {p:.0f}", "price": p, "style": "hold"})
            if len(supports) >= 2:
                p2 = float(supports[1]["price"])
                actions.append({"label": f"Invalidate < {p2:.0f}", "price": p2, "style": "invalidate"})
        if resistances:
            p = float(resistances[0]["price"])
            actions.append({"label": f"Reduce @ {p:.0f}", "price": p, "style": "reduce"})
            if len(resistances) >= 2:
                p2 = float(resistances[1]["price"])
                actions.append({"label": f"Add > {p2:.0f} accepted", "price": p2, "style": "add"})
        return actions

    def _draw_execution_overlay(
        self,
        ax,
        context: Dict,
        levels: List[Dict],
        current_price: float,
        timeframe: str,
    ) -> None:
        """Draw position management lines on all charts; action lines on 15m / 5m."""
        # Position lines (Entry / BE / SL / T0 / T1 / T2)
        pos_lines = self._get_position_lines(context)
        for pl in pos_lines:
            style = pl["style"]
            color = _CLR.get(style, "#888888")
            lw = 1.9 if style in ("sl", "entry") else 1.3
            ls = "--" if style in ("be", "t0", "t1", "t2") else "-"
            ax.axhline(pl["price"], color=color, linewidth=lw, linestyle=ls, alpha=0.88, zorder=5)
            ax.text(
                1.006, pl["price"], pl["label"],
                transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.5, color=color, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.90, "pad": 0.3},
                zorder=6,
            )

        # Action lines (15m / 5m only)
        if timeframe not in ("15m", "5m"):
            return
        for al in self._get_action_lines(levels, current_price):
            style = al["style"]
            color = _CLR.get(style, "#888888")
            ax.axhline(al["price"], color=color, linewidth=1.2, linestyle="-.", alpha=0.68, zorder=4)
            ax.text(
                0.01, al["price"], al["label"],
                transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=6.5, color=color, fontstyle="italic",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.60, "pad": 0.2},
                zorder=5,
            )

    # ─── badge helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_oi_badge(context: Dict, oi_period: str) -> Optional[Tuple[str, str]]:
        oi = context.get("open_interest_trend", {})
        state = oi.get("periods", {}).get(oi_period, oi).get("latest_state", "")
        return _OI_STATE.get(state)

    @staticmethod
    def _get_spot_perp_badge(context: Dict) -> Optional[Tuple[str, str]]:
        sp = context.get("spot_perp", {})
        if not sp.get("available"):
            return None
        bps   = float(sp.get("basis_bps", 0) or 0)
        cvd   = sp.get("cvd_state", "")
        p_delta = float(context.get("trade_flow", {}).get("delta_quote", 0) or 0)
        if bps > 8:
            return f"S/P: perp-led ↑  basis={bps:+.1f}bps", "#fde3e3"
        if bps < -8:
            return f"S/P: perp-led ↓  basis={bps:+.1f}bps", "#fde3e3"
        if cvd == "positive" and p_delta < 0:
            return f"S/P: spot-led divergent  basis={bps:+.1f}bps", "#fff3cd"
        if cvd == "positive":
            return f"S/P: spot-led  basis={bps:+.1f}bps", "#c8f7c5"
        if cvd == "negative" and p_delta < 0:
            return f"S/P: perp sell-led  basis={bps:+.1f}bps", "#fde3e3"
        return f"S/P: neutral  basis={bps:+.1f}bps", "#ececec"

    def _add_badges(self, ax, badges: Sequence[Tuple[str, str]]) -> None:
        x = 0.01
        for text, color in badges:
            ax.text(
                x, 0.985, text,
                transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color="#1f2933",
                bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.80, "pad": 0.3},
            )
            x += min(0.35, 0.011 * len(text) + 0.05)

    # ─── reference level drawing with zone boxes ─────────────────────────────

    def _draw_reference_levels(
        self,
        ax,
        levels: Sequence[Dict],
        current_price: float,
        visible_low: float,
        visible_high: float,
        label_limit: int,
    ) -> None:
        in_view = [lv for lv in levels
                   if visible_low * 0.998 <= float(lv["price"]) <= visible_high * 1.002]
        if not in_view:
            return

        standalone, zone_boxes = self._build_zone_boxes(in_view, current_price)

        # 1. Draw zone boxes (shaded rectangle + single merged label)
        placed: List[float] = []
        for zb in zone_boxes:
            c = zb["center"]
            if not (visible_low <= c <= visible_high):
                continue
            src   = zb.get("source", "profile")
            color = _CLR.get(src, "#dce3ea")
            ax.axhspan(zb["zone_low"], zb["zone_high"], color=color, alpha=0.09, zorder=0.5)
            # thin centre reference line
            pri = int(zb.get("priority", 2))
            ls, lw, alpha, _ = _TIER.get(pri, _TIER[2])
            ax.axhline(c, color=color, linestyle=ls, linewidth=lw * 0.75, alpha=alpha * 0.65, zorder=1)
            ax.text(
                1.006, c, zb["label"],
                transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.5, color=color, alpha=0.85,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.2},
            )
            placed.append(c)

        # 2. Draw standalone levels with anti-collision
        span    = max(1.0, visible_high - visible_low)
        min_gap = max(span * 0.025, current_price * 0.001, 12.0)
        sorted_sa = sorted(
            [lv for lv in standalone if visible_low <= float(lv["price"]) <= visible_high],
            key=lambda x: (int(x.get("priority", 3)), abs(float(x["price"]) - current_price)),
        )
        labelled = 0
        for lv in sorted_sa:
            price = float(lv["price"])
            src   = str(lv.get("source", "profile"))
            color = _CLR.get(src, "#5c677d")
            pri   = int(lv.get("priority", 3))
            ls, lw, alpha, fsize = _TIER.get(pri, _TIER[3])
            ax.axhline(price, color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=2)
            too_close = any(abs(price - p) < min_gap for p in placed)
            if not too_close and labelled < label_limit:
                ax.text(
                    1.006, price, str(lv["name"]),
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=fsize, color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.2},
                )
                placed.append(price)
                labelled += 1

    # ─── session backgrounds ─────────────────────────────────────────────────

    def _add_session_backgrounds(self, ax, candles: Sequence[Dict]) -> None:
        if not candles:
            return
        spans: List[tuple] = []
        cur   = self._session_name(int(candles[0]["open_time"]))
        start = 0
        for i, row in enumerate(candles[1:], 1):
            lbl = self._session_name(int(row["open_time"]))
            if lbl != cur:
                spans.append((start, i - 1, cur))
                start, cur = i, lbl
        spans.append((start, len(candles) - 1, cur))
        for s, e, lbl in spans:
            ax.axvspan(s - 0.5, e + 0.5, color=self.SESSION_COLORS[lbl], alpha=0.18, zorder=0)

    # ─── micro-structure zones (5m execution chart) ──────────────────────────

    def _draw_micro_zones(
        self,
        ax,
        trade_flow: Dict,
        current_price: float,
        candles_len: int,
        visible_low: float,
        visible_high: float,
    ) -> None:
        """Draw price-level delta absorption zones and large trade clusters."""
        # Price-level delta absorption zones
        pld = trade_flow.get("price_level_delta", {})
        if pld.get("available"):
            bin_sz = float(pld.get("bin_size", 100))
            for z in pld.get("absorption_zones", []):
                zp = float(z.get("price", 0))
                if zp <= 0 or not (visible_low <= zp <= visible_high):
                    continue
                ax.axhspan(zp - bin_sz / 2, zp + bin_sz / 2,
                           color=_CLR["absorption"], alpha=0.14, zorder=0.6)
                ax.text(candles_len + 0.3, zp, "Absorb",
                        fontsize=6, color=_CLR["absorption"],
                        ha="left", va="center", clip_on=False)

        # Large trade clusters (within 2 % of current price)
        for idx, c in enumerate(trade_flow.get("large_trade_clusters", [])[:3], 1):
            zl = float(c.get("zone_low",  0))
            zh = float(c.get("zone_high", 0))
            if zl <= 0 or zh <= 0:
                continue
            center = (zl + zh) / 2
            if abs(center - current_price) / max(current_price, 1) > 0.02:
                continue
            if not (visible_low <= center <= visible_high):
                continue
            buy  = float(c.get("buy_quote",  0))
            sell = float(c.get("sell_quote", 0))
            dom  = "B" if buy > sell else "S"
            color = "#0f9d58" if dom == "B" else "#d24c36"
            ax.axhspan(zl, zh, color=color, alpha=0.08, zorder=0.4)
            ax.text(candles_len + 0.3, center, f"Clus{idx} {dom}",
                    fontsize=6, color=color, ha="left", va="center", clip_on=False)

    # ─── chart spec ──────────────────────────────────────────────────────────

    @staticmethod
    def _chart_spec(timeframe: str) -> Dict:
        if timeframe == "4h":
            return {"mode": "structure",   "title": "4H Structure",
                    "oi_period": "1h",  "show_micro": False,
                    "label_limit": 6,   "show_trade_clusters": False, "show_exec": True}
        if timeframe == "1h":
            return {"mode": "decision",    "title": "1H Decision",
                    "oi_period": "15m", "show_micro": False,
                    "label_limit": 7,   "show_trade_clusters": False, "show_exec": True}
        if timeframe == "15m":
            return {"mode": "transition",  "title": "15m Transition",
                    "oi_period": "5m",  "show_micro": False,
                    "label_limit": 7,   "show_trade_clusters": False, "show_exec": True}
        # 5m
        return     {"mode": "execution",   "title": "5m Execution",
                    "oi_period": "5m",  "show_micro": True,
                    "label_limit": 5,   "show_trade_clusters": True,  "show_exec": True}

    # ─── line series helper ──────────────────────────────────────────────────

    def _plot_line_series(
        self, ax, candles, points, value_key: str, color: str, label: str,
    ) -> None:
        if not points:
            return
        xs = [self._timestamp_to_x(int(p.get("timestamp", 0)), candles) for p in points]
        ys = [float(p.get(value_key, 0)) for p in points]
        ax.plot(xs, ys, color=color, linewidth=1.2, label=label)

    # ─── panels ──────────────────────────────────────────────────────────────

    def _plot_price_panel(
        self, ax, candles: Sequence[Dict], timeframe: str, context: Dict, spec: Dict,
    ) -> None:
        opens  = [r["open"]  for r in candles]
        highs  = [r["high"]  for r in candles]
        lows   = [r["low"]   for r in candles]
        closes = [r["close"] for r in candles]
        xs     = list(range(len(candles)))

        self._add_session_backgrounds(ax, candles)

        # Candlesticks
        body_floor = max((max(highs) - min(lows)) * 0.001, 0.5)
        for i, row in enumerate(candles):
            o, c = row["open"], row["close"]
            col  = "#0f9d58" if c >= o else "#d24c36"
            ax.vlines(i, row["low"], row["high"], color=col, linewidth=1.0, alpha=0.95)
            ax.add_patch(Rectangle(
                (i - 0.31, min(o, c)), 0.62, max(abs(c - o), body_floor),
                facecolor=col, edgecolor=col, linewidth=0.7, alpha=0.88,
            ))

        # EMA lines — EMA7 dimmed (background); EMA25/99 are primary guides
        ema7  = self._ema(closes, 7)
        ema25 = self._ema(closes, 25)
        ema99 = self._ema(closes, 99)
        ax.plot(xs, ema7,  color="#d97706", linewidth=0.85, alpha=0.55,
                label="EMA7", linestyle="--")
        ax.plot(xs, ema25, color=_CLR["ema"], linewidth=1.35, label="EMA25")
        ax.plot(xs, ema99, color="#1f6f8b", linewidth=1.35, label="EMA99")

        current_price = float(context.get("price", closes[-1]))
        vis_low, vis_high = min(lows), max(highs)
        pad = (vis_high - vis_low) * 0.05
        vlo, vhi = vis_low - pad, vis_high + pad

        # Liquidity sweep zones (very faint background)
        for zk in ("high_sweep_zone", "low_sweep_zone"):
            z = context.get("recent_4h_range", {}).get(zk, {})
            if isinstance(z, dict):
                zl, zh = float(z.get("zone_low", 0)), float(z.get("zone_high", 0))
                if 0 < zl < zh:
                    ax.axhspan(zl, zh, color="#dce3ea", alpha=0.07, zorder=0.08)

        # Micro-structure zones (5m only)
        if spec["show_trade_clusters"]:
            self._draw_micro_zones(
                ax, context.get("trade_flow", {}), current_price, len(candles), vlo, vhi,
            )

        # Reference levels with zone boxes
        ref_levels = self._collect_reference_levels(context)
        self._draw_reference_levels(ax, ref_levels, current_price, vlo, vhi, int(spec["label_limit"]))

        # Session high / low (transition+ charts)
        if spec["mode"] in ("transition", "execution", "decision"):
            sc = context.get("session_context", {})
            for k, color in (("session_high", _CLR["session"]), ("session_low", _CLR["session"])):
                v = float(sc.get(k, 0) or 0)
                if v > 0:
                    ax.axhline(v, color=color, linestyle=":", linewidth=0.8, alpha=0.50)

        # Execution overlay (position + action lines)
        if spec.get("show_exec"):
            self._draw_execution_overlay(ax, context, ref_levels, current_price, timeframe)

        # Badges
        badges: List[Tuple[str, str]] = []
        oi_badge = self._get_oi_badge(context, spec["oi_period"])
        if oi_badge:
            badges.append(oi_badge)
        if timeframe in ("1h", "15m", "5m"):
            sp_badge = self._get_spot_perp_badge(context)
            if sp_badge:
                badges.append(sp_badge)
        badges.append((f"${current_price:,.1f}", "#e8f4f8"))
        self._add_badges(ax, badges)

        ax.set_title(
            f"{context.get('symbol', 'BTCUSDT')} {spec['title']}", fontsize=11, pad=8,
        )
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.38)
        ax.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.55)
        ax.set_xlim(-1, len(candles) + 6)

    def _plot_derivatives_panel(
        self, ax, candles: Sequence[Dict], context: Dict, spec: Dict,
    ) -> None:
        oi_period = spec["oi_period"]
        oi_pts = (context.get("open_interest_trend", {})
                  .get("periods", {}).get(oi_period, {}).get("series", []))
        self._plot_line_series(ax, candles, oi_pts, "open_interest", "#174b63", f"OI {oi_period}")
        ax.set_ylabel("OI", fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)

        ratio_hist = (context.get("long_short_ratio", {})
                      .get("global_account", {}).get("history", []))
        ax2 = ax.twinx()
        self._plot_line_series(ax2, candles, ratio_hist, "ratio", "#b56576", "Global L/S")
        ax2.set_ylabel("L/S", fontsize=8, color="#b56576")
        ax2.tick_params(axis="y", labelsize=8, colors="#b56576")
        ll, rl = ax.get_legend_handles_labels()
        lr, rr = ax2.get_legend_handles_labels()
        if ll or lr:
            ax.legend(ll + lr, rl + rr, loc="upper right", fontsize=7)

    @staticmethod
    def _plot_delta_panel(ax, candles: Sequence[Dict]) -> None:
        """Per-bar taker buy/sell delta bars + running CVD overlay."""
        deltas: List[float] = []
        for c in candles:
            buy   = float(c.get("taker_buy_base", 0))
            total = float(c.get("volume", 0))
            deltas.append(buy - (total - buy))
        if not any(d != 0 for d in deltas):
            ax.set_ylabel("Δ", fontsize=8)
            return
        colors = ["#0f9d58" if d >= 0 else "#d24c36" for d in deltas]
        ax.bar(range(len(deltas)), deltas, color=colors, width=0.72, alpha=0.72)
        ax.axhline(0, color="#888", linewidth=0.7, alpha=0.5)
        ax.set_ylabel("Δ BTC", fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
        running, cvd = 0.0, []
        for d in deltas:
            running += d
            cvd.append(running)
        ax2 = ax.twinx()
        ax2.plot(range(len(cvd)), cvd, color=_CLR["anchored_vwap"], linewidth=1.3, label="CVD")
        ax2.set_ylabel("CVD", fontsize=8, color=_CLR["anchored_vwap"])
        ax2.tick_params(axis="y", labelsize=7, colors=_CLR["anchored_vwap"])
        ll, rl = ax.get_legend_handles_labels()
        lr, rr = ax2.get_legend_handles_labels()
        if lr:
            ax.legend(ll + lr, rl + rr, loc="upper right", fontsize=7)

    def _plot_micro_panel(self, ax, candles: Sequence[Dict], context: Dict) -> None:
        tf = context.get("trade_flow", {})
        od = context.get("orderbook_dynamics", {})
        self._plot_line_series(ax, candles, tf.get("cvd_path", []),      "cvd_qty",   "#2a9d8f", "CVD")
        ax.set_ylabel("CVD", fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
        ax2 = ax.twinx()
        self._plot_line_series(ax2, candles, od.get("series", []), "imbalance", "#c65d07", "OB Imb")
        ax2.set_ylabel("OB", fontsize=8, color="#c65d07")
        ax2.tick_params(axis="y", labelsize=8, colors="#c65d07")
        ll, rl = ax.get_legend_handles_labels()
        lr, rr = ax2.get_legend_handles_labels()
        if ll or lr:
            ax.legend(ll + lr, rl + rr, loc="upper right", fontsize=7)

    def _plot_volume_panel(self, ax, candles: Sequence[Dict]) -> None:
        for i, row in enumerate(candles):
            col = "#0f9d58" if row["close"] >= row["open"] else "#d24c36"
            ax.bar(i, row["volume"], color=col, width=0.66, alpha=0.55)
        ax.set_ylabel("Vol", fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.30)
        tc   = min(8, len(candles))
        step = max(1, len(candles) // max(1, tc - 1))
        ticks = list(range(0, len(candles), step))
        if ticks[-1] != len(candles) - 1:
            ticks.append(len(candles) - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [self._format_time(int(candles[i]["open_time"])) for i in ticks], fontsize=8,
        )

    # ─── single chart ────────────────────────────────────────────────────────

    def _plot_single(
        self,
        symbol: str,
        timeframe: str,
        candles: Sequence[Dict],
        output_path: Path,
        context: Dict,
    ) -> None:
        spec        = self._chart_spec(timeframe)
        panel_count = 4 if spec["show_micro"] else 3
        ratios      = [5, 1.4, 1.1, 1.0] if spec["show_micro"] else [5, 1.4, 1.0]
        fig, axes = plt.subplots(
            panel_count, 1,
            figsize=(15, 9 if spec["show_micro"] else 8),
            sharex=True,
            gridspec_kw={"height_ratios": ratios},
        )
        axes_list = list(axes) if panel_count > 1 else [axes]
        fig.patch.set_facecolor("#f8f5f0")
        for a in axes_list:
            a.set_facecolor("#fffdf8")

        self._plot_price_panel(axes_list[0], candles, timeframe, context, spec)
        self._plot_derivatives_panel(axes_list[1], candles, context, spec)
        if spec["show_micro"]:
            has_taker = any(float(c.get("taker_buy_base", 0)) > 0 for c in candles)
            if has_taker:
                self._plot_delta_panel(axes_list[2], candles)
            else:
                self._plot_micro_panel(axes_list[2], candles, context)
            self._plot_volume_panel(axes_list[3], candles)
        else:
            self._plot_volume_panel(axes_list[2], candles)

        plt.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # ─── public entry points ─────────────────────────────────────────────────

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
            bars   = min(len(candles), self.bars_by_timeframe[timeframe])
            window = candles[-bars:]
            path   = self.output_dir / f"{symbol}_{timeframe}.png"
            self._plot_single(symbol, timeframe, window, path, context or {})
            chart_files[timeframe] = str(path.resolve())
        return chart_files

    def generate_spot_perp_chart(
        self,
        symbol: str,
        perp_klines: Sequence[Dict],
        spot_klines: Sequence[Dict],
        output_path: Path,
    ) -> str | None:
        """Spot vs perp comparison chart with basis bps and state badge."""
        if not perp_klines or not spot_klines:
            return None

        perp_sorted = sorted(perp_klines, key=lambda x: x["open_time"])[-120:]
        spot_lookup = {c["open_time"]: float(c["close"]) for c in spot_klines}
        aligned     = [c for c in perp_sorted if c["open_time"] in spot_lookup] or perp_sorted

        spot_closes = [spot_lookup.get(c["open_time"], 0.0) for c in aligned]
        basis_bps   = [
            (c["close"] - s) / s * 10000 if s > 0 else 0.0
            for c, s in zip(aligned, spot_closes)
        ]

        fig, axes = plt.subplots(
            3, 1, figsize=(15, 9), sharex=True,
            gridspec_kw={"height_ratios": [5, 1.5, 1.0]},
        )
        fig.patch.set_facecolor("#f8f5f0")
        for a in axes:
            a.set_facecolor("#fffdf8")

        ax_p      = axes[0]
        body_floor = max(
            (max(c["high"] for c in aligned) - min(c["low"] for c in aligned)) * 0.001, 0.5
        )
        for i, row in enumerate(aligned):
            col = "#0f9d58" if row["close"] >= row["open"] else "#d24c36"
            ax_p.vlines(i, row["low"], row["high"], color=col, linewidth=1.0, alpha=0.8)
            ax_p.add_patch(Rectangle(
                (i - 0.3, min(row["open"], row["close"])), 0.6,
                max(abs(row["close"] - row["open"]), body_floor),
                facecolor=col, edgecolor=col, linewidth=0.7, alpha=0.85,
            ))
        valid = [(i, s) for i, s in enumerate(spot_closes) if s > 0]
        if valid:
            xs, ys = zip(*valid)
            ax_p.plot(list(xs), list(ys), color="#457b9d", linewidth=1.6,
                      linestyle="--", label="Spot", alpha=0.9)

        # Spot/perp state badge (last 10 bars average)
        recent_avg = sum(basis_bps[-10:]) / max(1, len(basis_bps[-10:]))
        if recent_avg > 8:
            bt, bc = "perp premium high",  "#fde3e3"
        elif recent_avg > 3:
            bt, bc = "perp mild premium",  "#fff3cd"
        elif recent_avg < -8:
            bt, bc = "perp discount high", "#fde3e3"
        elif recent_avg < -3:
            bt, bc = "perp mild discount", "#fff3cd"
        else:
            bt, bc = "near parity",        "#c8f7c5"
        self._add_badges(ax_p, [
            (bt, bc),
            (f"basis(avg10) {recent_avg:+.1f}bps", "#e8f4f8"),
        ])

        ax_p.set_title(f"{symbol} Spot vs Perp (1H)", fontsize=12, pad=10)
        ax_p.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
        ax_p.legend(loc="upper left", fontsize=8)
        ax_p.set_xlim(-1, len(aligned) + 2)

        ax_b = axes[1]
        ax_b.bar(range(len(basis_bps)),
                 basis_bps,
                 color=["#0f9d58" if b >= 0 else "#d24c36" for b in basis_bps],
                 width=0.72, alpha=0.70)
        ax_b.axhline(0, color="#888", linewidth=0.8)
        ax_b.set_ylabel("Basis bps", fontsize=8)
        ax_b.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)

        ax_v = axes[2]
        for i, row in enumerate(aligned):
            col = "#0f9d58" if row["close"] >= row["open"] else "#d24c36"
            ax_v.bar(i, row.get("volume", 0), color=col, width=0.66, alpha=0.55)
        ax_v.set_ylabel("Perp Vol", fontsize=8)
        ax_v.grid(True, linestyle=":", linewidth=0.5, alpha=0.30)
        tc   = min(8, len(aligned))
        step = max(1, len(aligned) // max(1, tc - 1))
        ticks = list(range(0, len(aligned), step))
        if ticks and ticks[-1] != len(aligned) - 1:
            ticks.append(len(aligned) - 1)
        ax_v.set_xticks(ticks)
        ax_v.set_xticklabels(
            [self._format_time(int(aligned[i]["open_time"])) for i in ticks], fontsize=8,
        )

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return str(output_path.resolve())
