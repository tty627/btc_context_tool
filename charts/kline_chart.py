"""BTC structural-context chart generator.

Design philosophy
─────────────────
Charts are structural-context aids for a downstream AI, NOT decision panels.
• Show WHERE price is (structure, zones, key levels).
• Do NOT show WHAT to do (no action labels, no OI/S-P badges, no trade advice).
• Keep text labels minimal: object name + price tier only (POC, HVN2, 4H High …).
• The AI reads btc_context.json / btc_prompt.txt for indicator values and flow data;
  charts supply the spatial / visual context that text cannot replicate.

Three-tier level hierarchy
──────────────────────────
• Primary   (tier 1): thick solid line, bright — 4H range, POC
• Secondary (tier 2): thin dashed — HVN, AVWAP
• Tertiary  (tier 3): dotted, dim — LVN, minor pivots

Zone-box mechanism
──────────────────
• Nearby levels (within 0.2 % / $10) are merged into a shaded rectangle
  with a single merged label.  Reduces right-axis clutter.

position_overlay switch (default False)
────────────────────────────────────────
• False (default): clean structural chart, no position context drawn.
• True: draws Entry / BE / SL / Liq / T0 / T1 / T2 from account_positions
  and position_sizing.  Useful for manual review; disabled by default so
  that position anchoring does not influence AI structural reading.
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
    # micro structure zones (5m)
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

    def _collect_reference_levels(self, context: Dict, mode: str = "all") -> List[Dict]:
        """Collect reference levels filtered by chart mode.

        mode controls which level groups are included:
          structure  — 4H range, POC, daily anchors, EMA99
          decision   — structure + EMA25, Session H/L
          transition — POC, AVWAP, EMA25, HVN1, Session H/L
          execution  — AVWAP, Session H/L, HVN1
          all        — everything (legacy)
        """
        levels: List[Dict] = []
        r4h = context.get("recent_4h_range", {})
        vp = context.get("volume_profile", {})
        da = context.get("daily_anchors", {})
        sc = context.get("session_context", {})
        tfs = context.get("timeframes", {})

        include_range = mode in ("structure", "decision", "all")
        include_daily = mode in ("structure", "decision", "all")
        include_poc = mode != "execution"
        include_hvn = mode in ("transition", "execution", "all")
        include_lvn = mode == "all"
        include_avwap = mode in ("transition", "execution", "all")
        include_ema25 = mode in ("decision", "transition", "all")
        include_ema99 = mode in ("structure", "decision", "all")
        include_session = mode in ("decision", "transition", "execution", "all")

        if include_range:
            self._append_level(levels, "4H High", r4h.get("high"), "range", 1)
            self._append_level(levels, "4H Low",  r4h.get("low"),  "range", 1)

        if include_daily and da.get("available"):
            self._append_level(levels, "D High", da.get("prev_day_high"), "range", 1)
            self._append_level(levels, "D Low",  da.get("prev_day_low"),  "range", 1)
            self._append_level(levels, "W VWAP", da.get("weekly_vwap"),   "anchored_vwap", 2)
            self._append_level(levels, "W High", da.get("week_high"),     "range", 2)
            self._append_level(levels, "W Low",  da.get("week_low"),      "range", 2)
            self._append_level(levels, "M Open", da.get("month_open"),    "range", 2)

        if include_poc:
            self._append_level(levels, "POC", vp.get("poc_price"), "profile", 1)

        if include_hvn:
            for i, p in enumerate(vp.get("hvn_prices", [])[:1], 1):
                self._append_level(levels, f"HVN{i}", p, "profile", 2)

        if include_lvn:
            for i, p in enumerate(vp.get("lvn_prices", [])[:3], 1):
                self._append_level(levels, f"LVN{i}", p, "profile", 3)

        if include_avwap:
            anchor_names = {"recent_swing_high": "AVWAP-H", "recent_swing_low": "AVWAP-L"}
            for i, a in enumerate(vp.get("anchored_profiles", []), 1):
                if isinstance(a, dict):
                    self._append_level(
                        levels,
                        anchor_names.get(a.get("anchor_type"), f"AVWAP{i}"),
                        a.get("anchored_vwap"),
                        "anchored_vwap",
                        2 if mode in ("transition",) else 1,
                    )

        if include_ema25:
            for tf_name in ("1h",):
                ema = tfs.get(tf_name, {}).get("ema", {})
                if isinstance(ema, dict):
                    self._append_level(levels, f"{tf_name.upper()} EMA25", ema.get("25"), "ema", 2)

        if include_ema99:
            for tf_name in ("4h",):
                ema = tfs.get(tf_name, {}).get("ema", {})
                if isinstance(ema, dict):
                    self._append_level(levels, f"{tf_name.upper()} EMA99", ema.get("99"), "ema", 2)

        if include_session:
            sh = sc.get("session_high", 0)
            sl = sc.get("session_low", 0)
            if float(sh or 0) > 0:
                self._append_level(levels, "Sess H", sh, "session", 2)
            if float(sl or 0) > 0:
                self._append_level(levels, "Sess L", sl, "session", 2)

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

    def _draw_position_lines(self, ax, context: Dict) -> None:
        """Draw Entry / BE / SL / Liq / T0 / T1 / T2 lines (only when position_overlay=True)."""
        for pl in self._get_position_lines(context):
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
        _SESSION_LABELS = {"asia": "Asia", "europe": "Europe", "us": "US"}
        for s, e, lbl in spans:
            ax.axvspan(s - 0.5, e + 0.5, color=self.SESSION_COLORS[lbl], alpha=0.18, zorder=0)
            open_time = self._format_time(int(candles[s]["open_time"]))
            display = _SESSION_LABELS.get(lbl, lbl)
            ax.text(
                s + 0.5, 1.0, f" {display} {open_time}",
                transform=ax.get_xaxis_transform(),
                fontsize=6.5, color="#5c4b3a", alpha=0.70,
                ha="left", va="top",
            )

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
            return {
                "mode": "structure", "title": "4H Structure",
                "panels": ["price", "volume"],
                "oi_period": "1h", "show_micro": False,
                "label_limit": 5, "show_trade_clusters": False,
                "show_ema7": False, "show_events": False,
            }
        if timeframe == "1h":
            return {
                "mode": "decision", "title": "1H Decision",
                "panels": ["price", "derivatives", "volume"],
                "oi_period": "15m", "show_micro": False,
                "label_limit": 6, "show_trade_clusters": False,
                "show_ema7": False, "show_events": False,
            }
        if timeframe == "15m":
            return {
                "mode": "transition", "title": "15m Transition",
                "panels": ["price", "delta", "volume"],
                "oi_period": "5m", "show_micro": False,
                "label_limit": 6, "show_trade_clusters": False,
                "show_ema7": True, "show_events": True,
            }
        return {
            "mode": "execution", "title": "5m Execution Context",
            "panels": ["price", "delta", "micro", "volume"],
            "oi_period": "5m", "show_micro": True,
            "label_limit": 5, "show_trade_clusters": True,
            "show_ema7": True, "show_events": True,
        }

    # ─── event detection engine ─────────────────────────────────────────────

    def _detect_events(
        self,
        candles: Sequence[Dict],
        ref_levels: List[Dict],
        context: Dict,
        max_events: int = 8,
    ) -> List[Dict]:
        """Detect structural events on the visible candle window.

        Returns list of {"type", "x", "price", "label"} dicts, newest first,
        capped at max_events.
        """
        events: List[Dict] = []

        # --- break / reclaim of tier-1 levels ---
        tier1 = [lv for lv in ref_levels if int(lv.get("priority", 3)) <= 1]
        for lv in tier1:
            lp = float(lv["price"])
            name = str(lv.get("name", "?"))
            for i in range(1, len(candles)):
                prev_c = candles[i - 1]["close"]
                cur_c = candles[i]["close"]
                if prev_c >= lp > cur_c:
                    events.append({
                        "type": "break", "x": i, "price": candles[i]["low"],
                        "label": f"break {name}",
                    })
                elif prev_c <= lp < cur_c:
                    events.append({
                        "type": "reclaim", "x": i, "price": candles[i]["high"],
                        "label": f"reclaim {name}",
                    })

        # --- volume spike (> 2x MA20) ---
        if len(candles) >= 20:
            vols = [c["volume"] for c in candles]
            vol_ma = self._ema(vols, 20)
            for i, (v, m) in enumerate(zip(vols, vol_ma)):
                if m > 0 and v / m >= 2.0:
                    events.append({
                        "type": "vol_spike", "x": i, "price": candles[i]["high"],
                        "label": "vol spike",
                    })

        # --- CVD divergence (price new high but CVD not, or vice versa) ---
        if len(candles) >= 10:
            closes = [c["close"] for c in candles]
            deltas = []
            for c in candles:
                buy = float(c.get("taker_buy_base", 0))
                total = float(c.get("volume", 0))
                deltas.append(buy - (total - buy))
            if any(d != 0 for d in deltas):
                cvd = []
                running = 0.0
                for d in deltas:
                    running += d
                    cvd.append(running)
                lookback = min(20, len(candles) // 3)
                for i in range(lookback, len(candles)):
                    window_p = closes[i - lookback:i + 1]
                    window_c = cvd[i - lookback:i + 1]
                    p_max_idx = window_p.index(max(window_p))
                    c_max_idx = window_c.index(max(window_c))
                    if (closes[i] == max(window_p) and p_max_idx != c_max_idx
                            and cvd[i] < max(window_c) * 0.95):
                        events.append({
                            "type": "cvd_div", "x": i, "price": candles[i]["high"],
                            "label": "CVD div",
                        })

        events.sort(key=lambda e: e["x"], reverse=True)
        seen_x: set = set()
        deduped: List[Dict] = []
        for ev in events:
            if ev["x"] not in seen_x:
                deduped.append(ev)
                seen_x.add(ev["x"])
            if len(deduped) >= max_events:
                break
        return deduped

    def _draw_events(self, ax, events: List[Dict]) -> None:
        _EVENT_STYLE = {
            "break":     {"marker": "v", "color": "#d24c36", "va": "top"},
            "reclaim":   {"marker": "^", "color": "#0f9d58", "va": "bottom"},
            "vol_spike": {"marker": "D", "color": "#1d5c8b", "va": "bottom"},
            "cvd_div":   {"marker": "s", "color": "#7b4fa6", "va": "bottom"},
        }
        for ev in events:
            s = _EVENT_STYLE.get(ev["type"], _EVENT_STYLE["break"])
            ax.plot(ev["x"], ev["price"], marker=s["marker"], color=s["color"],
                    markersize=6, alpha=0.85, zorder=8)
            ax.annotate(
                ev["label"], (ev["x"], ev["price"]),
                textcoords="offset points",
                xytext=(0, -10 if s["va"] == "top" else 10),
                fontsize=6, color=s["color"], ha="center", va=s["va"],
                alpha=0.85,
            )

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
        position_overlay: bool = False,
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

        mode = spec.get("mode", "all")
        if spec.get("show_ema7", True):
            ema7 = self._ema(closes, 7)
            ax.plot(xs, ema7, color="#d97706", linewidth=0.85, alpha=0.55,
                    label="EMA7", linestyle="--")
        if mode in ("decision", "transition", "all"):
            ema25 = self._ema(closes, 25)
            ax.plot(xs, ema25, color=_CLR["ema"], linewidth=1.35, label="EMA25")
        if mode in ("structure", "decision", "all"):
            ema99 = self._ema(closes, 99)
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

        ref_levels = self._collect_reference_levels(context, mode=mode)
        self._draw_reference_levels(ax, ref_levels, current_price, vlo, vhi, int(spec["label_limit"]))

        if spec.get("show_events", False):
            events = self._detect_events(candles, ref_levels, context)
            self._draw_events(ax, events)

        if position_overlay:
            self._draw_position_lines(ax, context)

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

        # OI flush detection: 3+ consecutive drops with cumulative > 2%
        if len(oi_pts) >= 4:
            oi_vals = [float(p.get("open_interest", 0)) for p in oi_pts]
            oi_xs = [self._timestamp_to_x(int(p.get("timestamp", 0)), candles) for p in oi_pts]
            streak = 0
            streak_start = 0
            for k in range(1, len(oi_vals)):
                if oi_vals[k] < oi_vals[k - 1]:
                    if streak == 0:
                        streak_start = k - 1
                    streak += 1
                else:
                    if streak >= 3:
                        cum = (oi_vals[streak_start] - oi_vals[k - 1]) / max(oi_vals[streak_start], 1)
                        if cum > 0.02:
                            mid_x = oi_xs[(streak_start + k - 1) // 2]
                            mid_y = oi_vals[(streak_start + k - 1) // 2]
                            ax.annotate(
                                "OI flush", (mid_x, mid_y),
                                fontsize=7, color="#c65d07", fontweight="bold",
                                ha="center", va="bottom",
                            )
                    streak = 0

        ratio_hist = (context.get("long_short_ratio", {})
                      .get("global_account", {}).get("history", []))
        ax2 = ax.twinx()
        self._plot_line_series(ax2, candles, ratio_hist, "ratio", "#b56576", "Global L/S")
        ax2.set_ylabel("L/S", fontsize=8, color="#b56576")
        ax2.tick_params(axis="y", labelsize=8, colors="#b56576")

        # L/S crowded detection
        if len(ratio_hist) >= 5:
            ratios = [float(r.get("ratio", 1.0)) for r in ratio_hist]
            r_mean = sum(ratios) / len(ratios)
            r_var = sum((x - r_mean) ** 2 for x in ratios) / len(ratios)
            r_std = r_var ** 0.5
            if r_std > 0:
                for k, rv in enumerate(ratios):
                    z = (rv - r_mean) / r_std
                    if abs(z) >= 1.5:
                        rx = self._timestamp_to_x(int(ratio_hist[k].get("timestamp", 0)), candles)
                        ax2.annotate(
                            "crowded", (rx, rv),
                            fontsize=6.5, color="#b56576", alpha=0.85,
                            ha="center", va="bottom",
                        )

        # Funding rate line
        funding = context.get("funding", {})
        fr = funding.get("funding_rate")
        if fr is not None:
            fr_pct = float(fr) * 100
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("outward", 45))
            ax3.axhline(fr_pct, color="#7b4fa6", linewidth=1.2, linestyle="--",
                        alpha=0.7, label=f"FR {fr_pct:+.4f}%")
            ax3.axhline(0, color="#7b4fa6", linewidth=0.4, alpha=0.3)
            ax3.set_ylabel("FR%", fontsize=7, color="#7b4fa6")
            ax3.tick_params(axis="y", labelsize=7, colors="#7b4fa6")

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
        volumes = [row["volume"] for row in candles]
        for i, row in enumerate(candles):
            col = "#0f9d58" if row["close"] >= row["open"] else "#d24c36"
            ax.bar(i, row["volume"], color=col, width=0.66, alpha=0.55)

        if len(volumes) >= 20:
            ma20 = self._ema(volumes, 20)
            ax.plot(range(len(ma20)), ma20, color="#d97706", linewidth=1.1,
                    linestyle="--", alpha=0.75, label="Vol MA20")
            for i, (v, m) in enumerate(zip(volumes, ma20)):
                if m <= 0:
                    continue
                ratio = v / m
                if ratio >= 1.5:
                    ax.text(i, v, f"{ratio:.1f}x", fontsize=6, ha="center",
                            va="bottom", color="#0f9d58", alpha=0.85)
                elif ratio <= 0.5:
                    ax.text(i, v, f"{ratio:.1f}x", fontsize=6, ha="center",
                            va="bottom", color="#d24c36", alpha=0.70)
            ax.legend(loc="upper right", fontsize=7, framealpha=0.5)

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
        position_overlay: bool = False,
    ) -> None:
        spec = self._chart_spec(timeframe)
        panels = spec.get("panels", ["price", "volume"])

        _PANEL_RATIOS = {"price": 5, "derivatives": 1.4, "delta": 1.1, "micro": 1.0, "volume": 1.0}
        ratios = [_PANEL_RATIOS.get(p, 1.0) for p in panels]
        panel_count = len(panels)
        fig_h = 6 + 1.2 * (panel_count - 1)
        fig, axes = plt.subplots(
            panel_count, 1,
            figsize=(15, fig_h),
            sharex=True,
            gridspec_kw={"height_ratios": ratios},
        )
        axes_list = list(axes) if panel_count > 1 else [axes]
        fig.patch.set_facecolor("#f8f5f0")
        for a in axes_list:
            a.set_facecolor("#fffdf8")

        for ax, panel_name in zip(axes_list, panels):
            if panel_name == "price":
                self._plot_price_panel(
                    ax, candles, timeframe, context, spec,
                    position_overlay=position_overlay,
                )
            elif panel_name == "derivatives":
                self._plot_derivatives_panel(ax, candles, context, spec)
            elif panel_name == "delta":
                has_taker = any(float(c.get("taker_buy_base", 0)) > 0 for c in candles)
                if has_taker:
                    self._plot_delta_panel(ax, candles)
                else:
                    self._plot_micro_panel(ax, candles, context)
            elif panel_name == "micro":
                self._plot_micro_panel(ax, candles, context)
            elif panel_name == "volume":
                self._plot_volume_panel(ax, candles)

        plt.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # ─── public entry points ─────────────────────────────────────────────────

    def generate(
        self,
        symbol: str,
        klines_by_timeframe: Dict[str, Sequence[Dict]],
        context: Dict | None = None,
        position_overlay: bool = False,
    ) -> Dict[str, str]:
        """Generate multi-timeframe charts.

        Args:
            position_overlay: When True, draws Entry / BE / SL / T0 / T1 / T2 lines
                              derived from account_positions and position_sizing.
                              Defaults to False — charts are kept clean for AI structural
                              analysis without position-anchoring overlays.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        chart_files: Dict[str, str] = {}
        _NO_CHART_TF = {"1d", "1w"}
        for timeframe in self.bars_by_timeframe:
            if timeframe in _NO_CHART_TF:
                continue
            candles = list(klines_by_timeframe.get(timeframe, []))
            if not candles:
                continue
            bars   = min(len(candles), self.bars_by_timeframe[timeframe])
            window = candles[-bars:]
            path   = self.output_dir / f"{symbol}_{timeframe}.png"
            self._plot_single(
                symbol, timeframe, window, path, context or {},
                position_overlay=position_overlay,
            )
            chart_files[timeframe] = str(path.resolve())
        return chart_files

    @staticmethod
    def _rolling_zscore(values: Sequence[float], window: int = 20) -> List[float]:
        result: List[float] = []
        for i in range(len(values)):
            w = values[max(0, i - window + 1):i + 1]
            if len(w) < 2:
                result.append(0.0)
                continue
            mean = sum(w) / len(w)
            var = sum((x - mean) ** 2 for x in w) / len(w)
            std = var ** 0.5
            result.append((values[i] - mean) / std if std > 0 else 0.0)
        return result

    def generate_spot_perp_chart(
        self,
        symbol: str,
        perp_klines: Sequence[Dict],
        spot_klines: Sequence[Dict],
        output_path: Path,
        context: Dict | None = None,
    ) -> str | None:
        """Spot vs perp comparison chart with basis z-score and funding overlay."""
        if not perp_klines or not spot_klines:
            return None
        ctx = context or {}

        perp_sorted = sorted(perp_klines, key=lambda x: x["open_time"])[-120:]
        spot_lookup = {c["open_time"]: float(c["close"]) for c in spot_klines}
        aligned     = [c for c in perp_sorted if c["open_time"] in spot_lookup] or perp_sorted

        spot_closes = [spot_lookup.get(c["open_time"], 0.0) for c in aligned]
        basis_bps   = [
            (c["close"] - s) / s * 10000 if s > 0 else 0.0
            for c, s in zip(aligned, spot_closes)
        ]

        fig, axes = plt.subplots(
            4, 1, figsize=(15, 11), sharex=True,
            gridspec_kw={"height_ratios": [5, 1.5, 1.0, 1.0]},
        )
        fig.patch.set_facecolor("#f8f5f0")
        for a in axes:
            a.set_facecolor("#fffdf8")

        # Panel 1: Price
        ax_p = axes[0]
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

        zs = self._rolling_zscore(basis_bps)
        for i in range(1, len(zs)):
            if (zs[i - 1] <= 0 < zs[i]) or (zs[i - 1] >= 0 > zs[i]):
                ax_p.axvline(i, color="#7b4fa6", linewidth=0.8, alpha=0.45, linestyle=":")
                ax_p.text(i, ax_p.get_ylim()[1], "basis flip", fontsize=6,
                          ha="center", va="top", color="#7b4fa6", alpha=0.7)

        ax_p.set_title(f"{symbol} Spot vs Perp (1H)", fontsize=12, pad=10)
        ax_p.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
        ax_p.legend(loc="upper left", fontsize=8)
        ax_p.set_xlim(-1, len(aligned) + 2)

        # Panel 2: Basis bps + z-score
        ax_b = axes[1]
        ax_b.bar(range(len(basis_bps)),
                 basis_bps,
                 color=["#0f9d58" if b >= 0 else "#d24c36" for b in basis_bps],
                 width=0.72, alpha=0.70)
        ax_b.axhline(0, color="#888", linewidth=0.8)
        ax_b.set_ylabel("Basis bps", fontsize=8)
        ax_b.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)

        ax_z = ax_b.twinx()
        ax_z.plot(range(len(zs)), zs, color="#7b4fa6", linewidth=1.2, label="z-score")
        ax_z.axhline(2, color="#d24c36", linewidth=0.6, linestyle=":", alpha=0.5)
        ax_z.axhline(-2, color="#d24c36", linewidth=0.6, linestyle=":", alpha=0.5)
        for i, z in enumerate(zs):
            if abs(z) >= 2:
                ax_z.plot(i, z, "o", color="#d24c36", markersize=4, alpha=0.8)
        ax_z.set_ylabel("z-score", fontsize=8, color="#7b4fa6")
        ax_z.tick_params(axis="y", labelsize=7, colors="#7b4fa6")

        # Panel 3: Funding rate
        ax_f = axes[2]
        funding = ctx.get("funding", {})
        fr = funding.get("funding_rate")
        if fr is not None:
            fr_val = float(fr) * 100
            ax_f.axhline(fr_val, color="#1d5c8b", linewidth=1.5, label=f"FR {fr_val:+.4f}%")
            ax_f.axhline(0, color="#888", linewidth=0.6)
            ax_f.legend(loc="upper left", fontsize=7)
        ax_f.set_ylabel("Funding %", fontsize=8)
        ax_f.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)

        # Panel 4: Volume
        ax_v = axes[3]
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
