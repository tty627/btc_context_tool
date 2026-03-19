"""BTC market data panel — raw_first / anti-bias mode.

report_mode="raw_first"  (default):
    Only raw facts. No derived interpretations, no directional labels,
    no signal scores, no plan zones in the main report body.

report_mode="full_debug":
    Same main report, plus an optional appendix with derived / hypothesis
    fields (signal_score, deployment_context, plan_zones, state_tags, etc.)
    that MUST NOT be used to anchor the primary analysis.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

_NA = "unavailable"


def _f(val: Any, decimals: int = 2) -> str:
    """Format float to fixed decimals; return _NA on missing/invalid."""
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return _NA


def _v(val: Any, default: str = _NA) -> str:
    """Return str(val), or default if None / empty."""
    if val is None or val == "" or val == {}:
        return default
    return str(val)


class PromptGenerator:
    """Build a raw-first BTC market data panel for downstream LLM analysis."""

    # ── Data panel header ─────────────────────────────────────────────────────
    PANEL_HEADER = (
        "=== BTC MARKET DATA PANEL ===\n"
        "输入分为四段。必须按顺序阅读：先读 SECTION 1 + SECTION 2，完成独立判断后，\n"
        "才允许参考 SECTION 3 / SECTION 4（若存在）。\n"
        "---"
    )

    # ── Analysis system prompt ────────────────────────────────────────────────
    ANALYSIS_SYSTEM_PROMPT = """\
You are a BTC/USDT perpetual futures market decision engine.

Your objective is NOT to maximize win rate or trading frequency.
Your objective is to maximize trade quality and expectancy:

- Prefer asymmetric opportunities with clean location and tight invalidation.
- Low hit rate is acceptable if payoff is large and loss is small.
- Small losses are acceptable. Large uncontrolled losses are not.
- Do not defend a losing trade with hope. If thesis weakens or location edge disappears, favor exit / reduce / cancel quickly.
- If price is in a dirty middle area or RR is mediocre, output WAIT.

==================================================
WHAT TO DO
==================================================

1. If the user has an open position: manage risk first.
2. If the user is flat: output one primary decision — 开多 / 开空 / 等待.
3. Choose the best execution path. You may use market_now / limit_pullback / stop_trigger / staged.
4. If a hybrid or staged plan is best, use execution_mode=staged and describe it naturally in trade_plan.
5. Prefer an execution memo style, not a report style. Be concise and direct.
6. Do not emit decorative headers like 【主判断】 / 【持仓处理】 unless the user explicitly wants a report.

Output in Chinese.

==================================================
DATA READING RULES
==================================================

Direction: 4H and 1H structure still matter, but do not ignore lower-timeframe or microstructure evidence when it creates a clearly asymmetric entry with tight invalidation.
Evidence hierarchy: structural > tactical > ephemeral, but the hierarchy is for bias weighting, not for suppressing high-RR entries.
LEVELS labels (below_price / above_price) are positional only — not confirmed support/resistance.
Check flow tags before trusting any level.

==================================================
DATA QUALITY GATES
==================================================

Use the pre-computed verdicts in DATA_QUALITY directly. Do not re-derive them.

- dom_direction_gate = BLOCKED → DOM cannot determine direction by itself
- volume_gate = DEGRADED → lowers continuation confidence, does not auto-cancel a setup
- 15m/30m_flow_gate = FAIL → that window becomes secondary evidence
- weak_ref_* and optional_plan_hint_* fields → calibration only, never override raw evidence

==================================================
PRIOR DECISIONS
==================================================

If prior_decisions is present, use it only to detect systematic bias or repeated poor execution choices.
Never let historical calls override the current setup.

==================================================
OUTPUT FORMAT
==================================================

Keep only the minimum machine-readable core fields.
Use concise trader language, not essay/report language.
If a field is not useful, omit it rather than padding.
(See OUTPUT_FORMAT_TEMPLATE appended to the data prompt.)"""

    # ── Output format template (appended to data prompt, not in system) ───
    OUTPUT_FORMAT_TEMPLATE = """\
==================================================
OUTPUT FORMAT — 交易备忘录风格
==================================================

Choose ONE branch only. Do not print both branches.
Do not print decorative headers or separators.
Keep machine-readable keys, then use short trader language.

If NO position:
position_state: <flat / flat_assumed>
primary_decision: <开多 / 开空 / 等待>
execution_mode: <market_now / limit_pullback / stop_trigger / staged / wait>
confidence:
setup_quality: <A_edge / B_edge / C_edge / avoid>
thesis:
trade_plan:
  side:
  entries:
  stop_loss:
  thesis_invalidated_if:
  management:
  expected_RR:
  cancel_if:
  re_entry_if:
wait_plan:
  thesis:
  what_to_wait_for:
  bullish_trigger:
  bearish_trigger:
summary:
<3-5 句。直接说现在该怎么做、为什么、错了哪里撤。若某块不适用可省略，不要填空话。>

If POSITION exists:
current_position:
position_action: <hold / reduce / exit / add / reverse>
confidence:
thesis_status: <valid / weakening / invalid>
main_reason:
risk_line:
hard_invalidation:
exit_plan:
re_entry_plan:
summary:
<3-5 句。只说这笔单现在该怎么处理、什么情况下立刻撤。>

STRICT DECISION RULES

1. In no-position cases, output one and only one primary_decision.
2. Optimize for expectancy and setup quality, not for activity level or hit rate.
3. Prefer trades with clean location + tight invalidation + asymmetric payoff.
4. If price is in a dirty middle area or RR is mediocre, output WAIT.
5. Do not defend a loser because of higher-timeframe hope.
6. Do not add to a losing position unless a new setup with a new invalidation clearly exists.
7. stop_loss / hard_invalidation must be specific.
8. You may use staged execution if it genuinely improves edge.
9. Keep the writing direct and trader-like."""

    DECISION_SYSTEM_PROMPT = ANALYSIS_SYSTEM_PROMPT
    DECISION_OUTPUT_FORMAT_TEMPLATE = OUTPUT_FORMAT_TEMPLATE
    VISION_SYSTEM_NOTE = (
        "[Vision] The user message includes candlestick structure charts. "
        "Integrate visual structure (swings, ranges, trend) with the numeric panel."
    )
    RESEARCH_SYSTEM_PROMPT = """\
You are a BTC/USDT perpetual futures market microstructure researcher.

Your job is NOT to place the trade directly. Your job is to study the market from
raw facts, charts, and high-value raw appendix evidence, then hand off a clean,
anti-bias research summary to a separate decision engine.

Output in Chinese for natural-language values, but keep JSON keys in English.

==================================================
RESEARCH RULES
==================================================

1. Read SECTION 1 + SECTION 2 first. Treat RAW_APPENDIX as supporting raw evidence.
2. Do not use prior_decisions, deployment hints, signal_score, or directional labels to decide bias.
3. Separate structure, tactical timing, and microstructure evidence clearly.
4. Produce a trader handoff, not a balanced research report.
5. Focus on the single best path now; only mention a secondary path if it materially matters.
6. Be explicit about what would change your mind and what would force a fast exit.
7. Return one valid JSON object only.
"""
    RESEARCH_OUTPUT_FORMAT_TEMPLATE = """\
==================================================
OUTPUT FORMAT — RETURN JSON ONLY
==================================================

Return one valid JSON object. No markdown fence. No prose before or after.
Use Chinese for natural-language values, English snake_case for keys.

{
  "stage": "research",
  "position_state": "flat|open_position|unknown",
  "market_regime": "",
  "best_path_now": "",
  "edge_quality": "A|B|C|avoid",
  "confidence": "low|medium|high",
  "why_now": "",
  "entry_focus": "",
  "invalidation_focus": "",
  "fast_exit_signal": "",
  "secondary_path": "",
  "do_not_do": [],
  "key_levels": [],
  "supporting_evidence": [],
  "contradictory_evidence": [],
  "raw_appendix_takeaways": [],
  "if_wrong_then": "",
  "position_management_priority": ""
}"""

    def build(
        self,
        context: Dict,
        report_mode: str = "raw_first",
        include_instructions: bool = True,
    ) -> str:
        """Build the legacy single-stage decision panel."""
        return self._build_panel(
            context,
            report_mode=report_mode,
            include_instructions=include_instructions,
            instruction_prompt=self.ANALYSIS_SYSTEM_PROMPT,
            instruction_title="=== ANALYSIS INSTRUCTIONS (请先阅读此段指令，再分析下方数据) ===",
            include_prior_decisions=True,
            include_debug_sections=(report_mode == "full_debug"),
            include_raw_appendix=False,
            output_template=self.OUTPUT_FORMAT_TEMPLATE,
        )

    def build_research_prompt(
        self,
        context: Dict,
        include_instructions: bool = False,
    ) -> str:
        return self._build_panel(
            context,
            report_mode="raw_first",
            include_instructions=include_instructions,
            instruction_prompt=self.RESEARCH_SYSTEM_PROMPT,
            instruction_title="=== RESEARCH INSTRUCTIONS (先研究，再交给决策阶段) ===",
            include_prior_decisions=False,
            include_debug_sections=False,
            include_raw_appendix=True,
            output_template=self.RESEARCH_OUTPUT_FORMAT_TEMPLATE,
        )

    def build_decision_prompt(
        self,
        context: Dict,
        research_handoff: str,
        report_mode: str = "raw_first",
        include_instructions: bool = False,
    ) -> str:
        return self._build_panel(
            context,
            report_mode=report_mode,
            include_instructions=include_instructions,
            instruction_prompt=self.DECISION_SYSTEM_PROMPT,
            instruction_title="=== DECISION INSTRUCTIONS (基于研究交接生成最终决策) ===",
            include_prior_decisions=True,
            include_debug_sections=(report_mode == "full_debug"),
            include_raw_appendix=True,
            extra_sections=self._research_handoff_block(research_handoff),
            output_template=self.DECISION_OUTPUT_FORMAT_TEMPLATE,
        )

    def build_system_prompt(self, stage: str = "decision", include_vision: bool = False) -> str:
        if stage == "research":
            system = self.RESEARCH_SYSTEM_PROMPT + "\n\n" + self.RESEARCH_OUTPUT_FORMAT_TEMPLATE
        else:
            system = self.DECISION_SYSTEM_PROMPT + "\n\n" + self.DECISION_OUTPUT_FORMAT_TEMPLATE
        if include_vision:
            system += "\n\n" + self.VISION_SYSTEM_NOTE
        return system

    def _build_panel(
        self,
        context: Dict,
        report_mode: str,
        *,
        include_instructions: bool,
        instruction_prompt: str,
        instruction_title: str,
        include_prior_decisions: bool,
        include_debug_sections: bool,
        include_raw_appendix: bool,
        output_template: str,
        extra_sections: List[str] | None = None,
    ) -> str:
        sections: List[str] = []

        if include_instructions:
            sections += [
                instruction_title,
                "",
                instruction_prompt,
                "",
                "─" * 60,
                "",
            ]

        sections += self._panel_body(
            context,
            report_mode=report_mode,
            include_prior_decisions=include_prior_decisions,
            include_debug_sections=include_debug_sections,
            include_raw_appendix=include_raw_appendix,
        )
        if extra_sections:
            sections += ["", *extra_sections]
        sections += [
            "",
            "─" * 60,
            "",
            output_template,
        ]
        return "\n".join(sections).strip()

    def _panel_body(
        self,
        context: Dict,
        report_mode: str,
        *,
        include_prior_decisions: bool,
        include_debug_sections: bool,
        include_raw_appendix: bool,
    ) -> List[str]:
        sections: List[str] = [
            self.PANEL_HEADER,
            "",
            "╔══════════════════════════════════════════════════════════╗",
            "║  SECTION 1: RAW_FACTS  (客观市场事实，AI 独立判断依据)      ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
        ]
        sections += self._market_facts(context)
        sections.append("")
        sections += self._indicators(context)
        sections.append("")
        sections += self._candle_structure(context)
        sections.append("")
        sections += self._levels(context)
        sections.append("")
        sections += self._orderbook(context)
        sections.append("")
        sections += self._trade_flow(context)
        sections.append("")
        sections += self._derivatives(context)
        sections.append("")
        sections += self._transition_dynamics(context)
        sections.append("")
        sections += self._spot_vs_perp(context)
        sections.append("")
        sections += self._position_facts(context)
        sections.append("")
        sections += self._external_drivers(context)
        if include_prior_decisions:
            sections.append("")
            sections += self._prior_decisions_context(context)

        sections += [
            "",
            "╔══════════════════════════════════════════════════════════╗",
            "║  SECTION 2: DATA_QUALITY  (数据质量与可靠性约束)            ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
        ]
        sections += self._data_quality(context)

        if include_raw_appendix:
            sections += [
                "",
                "╔══════════════════════════════════════════════════════════╗",
                "║  SECTION 3: RAW_APPENDIX  (高价值原始附录，仅供二次核验)   ║",
                "╚══════════════════════════════════════════════════════════╝",
                "",
            ]
            sections += self._raw_appendix(context)

        if include_debug_sections and report_mode == "full_debug":
            sections += [
                "",
                "╔══════════════════════════════════════════════════════════╗",
                "║  SECTION 3: DERIVED_SIGNALS  ⚠ weak_ref_only — 仅在      ║",
                "║  PHASE A 完成后参考，与 raw facts 冲突时忽略               ║",
                "╚══════════════════════════════════════════════════════════╝",
                "",
            ]
            sections += self._derived_signals(context)
            sections += [
                "",
                "╔══════════════════════════════════════════════════════════╗",
                "║  SECTION 4: DEPLOYMENT_HINTS  ⚠ optional_plan_hint —     ║",
                "║  不得用于决定开仓方向，仅供执行细节参考                    ║",
                "╚══════════════════════════════════════════════════════════╝",
                "",
            ]
            sections += self._deployment_hints(context)

        return sections

    # ─── 1. DATA QUALITY ────────────────────────────────────────────────────

    @staticmethod
    def _data_quality(ctx: Dict) -> List[str]:
        lines = ["=== DATA QUALITY ==="]

        generated_at = ctx.get("generated_at", _NA)
        lines.append(f"generated_at: {generated_at}")
        try:
            dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            freshness_sec = round((datetime.now(timezone.utc) - dt).total_seconds())
            lines.append(f"freshness_sec: {freshness_sec}")
        except Exception:
            lines.append(f"freshness_sec: {_NA}")

        tf = ctx.get("trade_flow", {})
        cov_min = tf.get("coverage_minutes", _NA)
        trade_count = tf.get("trade_count", _NA)
        lines.append(f"trade_flow_coverage: {_f(cov_min, 1)}min  trades: {trade_count}")

        windows = tf.get("windows", {})
        w_parts = []
        for label in ("1m", "5m", "15m", "30m"):
            w = windows.get(label, {})
            ratio = w.get("coverage_ratio")
            if ratio is not None:
                flag = " [LOW]" if float(ratio) < 0.5 else ""
                w_parts.append(f"{label}={_f(ratio, 2)}{flag}")
        if w_parts:
            lines.append(f"window_coverage_ratio: {' | '.join(w_parts)}")

        # ── HARD GATE VERDICTS ───────────────────────────────────────────
        _FLOW_THRESHOLDS = {"15m": 0.30, "30m": 0.20}
        for label, threshold in _FLOW_THRESHOLDS.items():
            w = windows.get(label, {})
            ratio = w.get("coverage_ratio")
            if ratio is not None:
                r = float(ratio)
                if r >= threshold:
                    lines.append(
                        f"{label}_flow_gate: PASS (cov={_f(r, 2)} >= {threshold})"
                    )
                else:
                    lines.append(
                        f"{label}_flow_gate: FAIL (cov={_f(r, 2)} < {threshold})"
                        f" -> {label} flow = secondary only"
                    )
            else:
                lines.append(f"{label}_flow_gate: FAIL (no data)")

        od = ctx.get("orderbook_dynamics", {})
        snap = od.get("snapshot_count", _NA)
        dur = od.get("sample_duration_seconds", _NA)
        lines.append(f"orderbook_samples: {snap}  sample_duration_sec: {_f(dur, 1)}")

        spoof = od.get("spoofing_risk", "unknown")
        dur_val = float(dur) if dur != _NA else 0.0
        if dur_val < 20:
            lines.append(
                f"dom_direction_gate: BLOCKED (sample_duration={_f(dur_val, 1)}s < 20s)"
                f" -> DOM cannot determine direction"
            )
        elif spoof in ("high", "medium"):
            lines.append(
                f"dom_direction_gate: BLOCKED (spoofing_risk={spoof})"
                f" -> DOM cannot determine direction"
            )
        else:
            lines.append(f"dom_direction_gate: PASS (spoofing_risk={spoof})")

        vc = ctx.get("volume_change", {})
        vol_parts = []
        for vtf in ("15m", "1h", "4h"):
            v = vc.get(vtf, {})
            vs_avg = v.get("vs_avg20_pct")
            if vs_avg is not None:
                vol_parts.append(f"{vtf}={_f(vs_avg, 1)}%")
        any_low = any(
            float(vc.get(t, {}).get("vs_avg20_pct", 0)) < -30
            for t in ("15m", "1h", "4h")
            if vc.get(t, {}).get("vs_avg20_pct") is not None
        )
        if any_low:
            lines.append(
                f"volume_gate: DEGRADED ({', '.join(vol_parts)})"
                f" -> lower breakout/continuation confidence"
            )
        else:
            lines.append(f"volume_gate: PASS ({', '.join(vol_parts)})")

        kf = tf.get("kline_flow", {})
        if not kf.get("available", False):
            reason = kf.get("reason", "unknown")
            lines.append(f"kline_flow: unavailable ({reason})")
        else:
            lines.append("kline_flow: available")

        sp = ctx.get("spot_perp", {})
        lines.append(f"spot_perp: {'available' if sp.get('available') else 'unavailable'}")

        cx = ctx.get("cross_exchange_oi", {})
        lines.append(f"cross_exchange_oi: {cx.get('available_count', 0)}/3 sources")

        acc = ctx.get("account_positions", {})
        lines.append(f"account_positions: {'available' if acc.get('available') else 'unavailable'}")

        # ── LEVEL-FLOW CONTRADICTION CHECK ───────────────────────────────
        # Cross-reference positional level labels with actual flow at those levels.
        # If a below_price level has sell_dominant flow → label is contradicted by flow.
        # If an above_price level has buy_dominant flow → label is contradicted by flow.
        klf_for_check = ctx.get("trade_flow", {}).get("key_level_flows", [])
        ref_levels_for_check = ctx.get("deployment_context", {}).get("reference_levels", [])
        if klf_for_check and ref_levels_for_check:
            conflicts: List[str] = []
            for kl in klf_for_check:
                tag = str(kl.get("tag", ""))
                if tag not in ("sell_dominant", "buy_dominant"):
                    continue
                kl_price = round(float(kl.get("price", 0)), 1)
                for lv in ref_levels_for_check:
                    lv_price = round(float(lv.get("price", 0)), 1)
                    if abs(lv_price - kl_price) < 1.0:
                        role = str(lv.get("role", ""))
                        if (role == "support" and tag == "sell_dominant") or (
                            role == "resistance" and tag == "buy_dominant"
                        ):
                            conflicts.append(f"{kl.get('name', '?')}@{kl_price}({tag})")
            if conflicts:
                lines.append(
                    f"level_flow_conflict: {', '.join(conflicts)}"
                    f" -> flow contradicts positional label"
                    f" -> do NOT treat these as confirmed structure for entry"
                )
            else:
                lines.append("level_flow_conflict: none")

        return lines

    # ─── 2. MARKET FACTS ────────────────────────────────────────────────────

    @staticmethod
    def _market_facts(ctx: Dict) -> List[str]:
        lines = ["=== MARKET FACTS ==="]

        price = ctx.get("price", _NA)
        stats = ctx.get("stats_24h", {})
        session = ctx.get("session_context", {})
        r4h = ctx.get("recent_4h_range", {})
        funding = ctx.get("funding", {})

        lines.append(f"price: {price}")
        lines.append(
            f"24h: H={_v(stats.get('high_price'))} "
            f"L={_v(stats.get('low_price'))} "
            f"chg={_f(stats.get('price_change_percent'), 3)}%  "
            f"vol={_f(stats.get('volume'), 0)}BTC"
        )
        lines.append(
            f"session: {_v(session.get('current_session')).upper()}  "
            f"session_H={_v(session.get('session_high'))}  "
            f"session_L={_v(session.get('session_low'))}"
        )
        lines.append(
            f"4h_range: H={_v(r4h.get('high'))} "
            f"L={_v(r4h.get('low'))} "
            f"range_pct={_f(r4h.get('range_pct'), 3)}%"
        )
        lines.append(f"funding_countdown: {_v(session.get('funding_countdown_label'))}")
        lines.append(f"next_funding_time: {_v(funding.get('next_funding_time'))}")

        return lines

    # ─── 3. INDICATORS ──────────────────────────────────────────────────────

    @staticmethod
    def _indicators(ctx: Dict) -> List[str]:
        lines = ["=== INDICATORS ==="]

        current_price = float(ctx.get("price", 0) or 0)
        vwap_parts: List[str] = []
        timeframes_data = ctx.get("timeframes", {})

        for tf, m in timeframes_data.items():
            rsi = m.get("rsi", {})
            macd = m.get("macd", {})
            bb = m.get("bollinger", {})
            atr = m.get("atr", {})
            ema = m.get("ema", {})
            vwap_val = m.get("vwap")

            # MACD: hist + direction only (DIF/DEA removed — redundant with EMA trend)
            hist = float(macd.get("hist", 0) or 0)
            macd_dir = "pos" if hist > 0 else ("neg" if hist < 0 else "flat")
            # DIF vs zero line tells if momentum is net positive/negative
            dif = float(macd.get("dif", 0) or 0)
            dif_sign = "above_zero" if dif > 0 else "below_zero"

            # BB: %B position label + bandwidth only
            pct_b = float(bb.get("percent_b", 0.5) or 0.5)
            if pct_b > 1.0:
                bb_pos = "above_upper"
            elif pct_b < 0.0:
                bb_pos = "below_lower"
            elif pct_b >= 0.8:
                bb_pos = "near_upper"
            elif pct_b <= 0.2:
                bb_pos = "near_lower"
            else:
                bb_pos = "mid"

            # ATR: include pct_rank if available
            atr_line = f"ATR={_f(atr.get('atr'), 1)}  ATR%={_f(atr.get('atr_pct'), 3)}%"
            pct_rank = atr.get("pct_rank_30")
            if pct_rank is not None:
                rank_label = "low_vol" if pct_rank < 30 else ("high_vol" if pct_rank > 70 else "mid_vol")
                atr_line += f"  pct30d={pct_rank:.0f}%({rank_label})"

            # VWAP: collect for summary line below
            if vwap_val is not None and current_price > 0:
                rel = "above" if current_price > float(vwap_val) else "below"
                vwap_parts.append(f"{tf}={_f(vwap_val, 0)}({rel})")

            lines.append(f"[{tf}]")
            lines.append(f"  RSI14={_f(rsi.get('14'))}  state={_v(rsi.get('state'))}")
            lines.append(f"  MACD: hist={_f(hist)}({macd_dir})  dif={dif_sign}")
            lines.append(f"  BB: %B={_f(pct_b)}({bb_pos})  bw={_f(bb.get('bandwidth'))}")
            lines.append(atr_line)
            lines.append(
                f"  EMA7={_f(ema.get('7'), 1)}  EMA25={_f(ema.get('25'), 1)}  EMA99={_f(ema.get('99'), 1)}"
            )

        # VWAP summary — single line replacing per-TF VWAP rows
        if vwap_parts:
            lines.append("vwap_vs_price: " + "  ".join(vwap_parts))

        # Volume profile
        vp = ctx.get("volume_profile", {})
        if vp.get("poc_price"):
            lines.append(
                f"VP ({_v(vp.get('source_timeframe'))} {_v(vp.get('window_size'))}h):"
            )
            lines.append(f"  POC={_f(vp.get('poc_price'), 1)}")
            lines.append(f"  HVN={[round(p, 1) for p in (vp.get('hvn_prices') or [])]}")
            lines.append(f"  LVN={[round(p, 1) for p in (vp.get('lvn_prices') or [])]}")

        # Session VP sub-profiles
        sess_profiles = vp.get("session_profiles", {}).get("profiles", {})
        for sess, prof in sess_profiles.items():
            if prof.get("poc_price"):
                lines.append(
                    f"  session_{sess}: POC={_f(prof.get('poc_price'), 1)} "
                    f"H={_v(prof.get('high'))} L={_v(prof.get('low'))}"
                )

        # Anchored VWAP profiles
        for ap in vp.get("anchored_profiles", []):
            lines.append(
                f"  AVWAP({ap.get('anchor_type')}): "
                f"anchor_price={_v(ap.get('anchor_price'))} "
                f"vwap={_f(ap.get('anchored_vwap'), 1)} "
                f"dist_bps={_f(ap.get('distance_to_vwap_bps'), 2)}"
            )

        # Volume change vs 20-bar average
        vc = ctx.get("volume_change", {})
        vc_parts = []
        for period in ("5m", "15m", "1h", "4h"):
            v = vc.get(period, {})
            if v:
                vc_parts.append(f"{period}: vs_avg20={_f(v.get('vs_avg20_pct'), 1)}%")
        if vc_parts:
            lines.append("vol_vs_avg20: " + "  ".join(vc_parts))

        return lines

    # ─── 3b. CANDLE STRUCTURE ───────────────────────────────────────────────

    @staticmethod
    def _candle_structure(ctx: Dict) -> List[str]:
        cs = ctx.get("candle_structure", {})
        if not cs:
            return []
        lines = ["=== CANDLE STRUCTURE (recent 3 closed bars — 4h / 1h) ==="]
        for tf in ("4h", "1h"):
            bars = cs.get(tf, [])
            if not bars:
                continue
            lines.append(f"[{tf}]")
            n = len(bars)
            for i, bar in enumerate(bars):
                offset = i - n  # e.g. -3, -2, -1
                vol_str = f"{bar['vol_ratio']:.1f}x" if bar.get("vol_ratio") is not None else "?"
                lines.append(
                    f"  [{offset}]: {bar['direction']}"
                    f"  body={bar['body_pct']:.0f}%"
                    f"  lwk={bar['lower_wick_pct']:.0f}%"
                    f"  uwk={bar['upper_wick_pct']:.0f}%"
                    f"  close={bar['close_pos']}"
                    f"  vol={vol_str}"
                    f"  pattern={bar['pattern']}"
                )
        return lines

    # ─── 4. LEVELS (source-labeled, positional grouping only) ──────────────
    # "below_price" / "above_price" are POSITIONAL markers only — they do NOT
    # confirm buying or selling pressure at that level.
    # Always cross-check flow tags before assuming a level is defended.

    @staticmethod
    def _levels(ctx: Dict) -> List[str]:
        lines = [
            "=== LEVELS (positional grouping only: below_price / above_price) ===",
            "# WARNING: grouping is price-relative, NOT a structural trading signal.",
            "# A level labeled below_price is NOT validated support unless flow=absorbed or buy_dominant.",
        ]

        deploy = ctx.get("deployment_context", {})
        ref_levels = deploy.get("reference_levels", [])

        # Build flow tag lookup keyed by rounded price
        flow_lookup: Dict[float, str] = {}
        for kl in ctx.get("trade_flow", {}).get("key_level_flows", []):
            kl_price = round(float(kl.get("price", 0)), 1)
            flow_lookup[kl_price] = str(kl.get("tag", ""))

        _ROLE_DISPLAY = {"support": "below_price", "resistance": "above_price", "neutral": "neutral"}
        by_role: Dict[str, List[Dict]] = {"support": [], "resistance": [], "neutral": []}
        for lv in ref_levels:
            role = str(lv.get("role", "neutral"))
            by_role.setdefault(role, []).append(lv)

        for role in ("support", "resistance", "neutral"):
            lvs = by_role.get(role, [])
            if lvs:
                lines.append(f"{_ROLE_DISPLAY.get(role, role)}:")
                for lv in sorted(lvs, key=lambda x: float(x.get("price", 0))):
                    price_key = round(float(lv.get("price", 0)), 1)
                    flow_tag = flow_lookup.get(price_key, "")
                    tags: List[str] = []
                    if lv.get("near_price"):
                        tags.append("near_price⚠")
                    if flow_tag:
                        contradicts = (
                            (role == "support" and flow_tag == "sell_dominant")
                            or (role == "resistance" and flow_tag == "buy_dominant")
                        )
                        if contradicts:
                            tags.append(f"flow={flow_tag} CONTRADICTS label — not confirmed")
                        else:
                            tags.append(f"flow={flow_tag}")
                    tag_suffix = ("  [" + " | ".join(tags) + "]") if tags else ""
                    lines.append(
                        f"  [{lv.get('source', '?')}] {lv.get('name', '?')} @ {_f(lv.get('price'), 1)}{tag_suffix}"
                    )

        # Recent 4h sweep zones (non-directional boundaries)
        r4h = ctx.get("recent_4h_range", {})
        high_zone = r4h.get("high_sweep_zone", {})
        low_zone = r4h.get("low_sweep_zone", {})
        if high_zone:
            lines.append(
                f"zone [range] 4h_high_sweep: "
                f"{_f(high_zone.get('zone_low'), 1)}-{_f(high_zone.get('zone_high'), 1)}"
            )
        if low_zone:
            lines.append(
                f"zone [range] 4h_low_sweep: "
                f"{_f(low_zone.get('zone_low'), 1)}-{_f(low_zone.get('zone_high'), 1)}"
            )

        da = ctx.get("daily_anchors", {})
        if da.get("available"):
            lines.append(f"[daily] prev_day_high: {_f(da.get('prev_day_high'), 1)}")
            lines.append(f"[daily] prev_day_low: {_f(da.get('prev_day_low'), 1)}")
            lines.append(f"[daily] weekly_vwap: {_f(da.get('weekly_vwap'), 1)}")
            lines.append(
                f"[daily] week_high: {_f(da.get('week_high'), 1)}  "
                f"week_low: {_f(da.get('week_low'), 1)}"
            )
            lines.append(f"[daily] month_open: {_f(da.get('month_open'), 1)}")
            lines.append(
                f"[daily] month_high: {_f(da.get('month_high'), 1)}  "
                f"month_low: {_f(da.get('month_low'), 1)}"
            )

        return lines

    # ─── 5. ORDERBOOK ───────────────────────────────────────────────────────

    @staticmethod
    def _orderbook(ctx: Dict) -> List[str]:
        lines = ["=== ORDERBOOK ==="]

        ob = ctx.get("orderbook", {})
        lines.append(
            f"bid_vol={_v(ob.get('bid_volume'))}  "
            f"ask_vol={_v(ob.get('ask_volume'))}  "
            f"imbalance={_f(ob.get('imbalance'))}"
        )
        bw = ob.get("bid_wall", {})
        aw = ob.get("ask_wall", {})
        if isinstance(bw, dict):
            lines.append(f"largest_bid_wall: @{_v(bw.get('price'))} qty={_v(bw.get('qty'))}")
        if isinstance(aw, dict):
            lines.append(f"largest_ask_wall: @{_v(aw.get('price'))} qty={_v(aw.get('qty'))}")

        # ── DOM dynamics (VETO data — cannot determine direction) ──
        od = ctx.get("orderbook_dynamics", {})
        if od:
            dur = float(od.get("sample_duration_seconds", 0) or 0)
            spoof = od.get("spoofing_risk", "unknown")
            blocked = dur < 20 or spoof in ("high", "medium")
            lines.append(
                f"[DOM veto-only] sample={_f(dur, 0)}s  spoofing={spoof}  "
                f"{'BLOCKED' if blocked else 'usable for timing only'}"
            )
            if not blocked:
                lines.append(
                    f"  imbalance avg/max={_f(od.get('avg_imbalance'), 3)}/"
                    f"{_f(od.get('max_imbalance'), 3)}  "
                    f"cancel/add={_f(od.get('cancel_to_add_ratio'), 2)}  "
                    f"sweep_quote={_f(od.get('aggressive_sweep_quote', 0), 0)}"
                )

        return lines

    # ─── 6. TRADE FLOW ──────────────────────────────────────────────────────

    @staticmethod
    def _trade_flow(ctx: Dict) -> List[str]:
        lines = ["=== TRADE FLOW ==="]

        tf = ctx.get("trade_flow", {})
        if not tf:
            lines.append(_NA)
            return lines

        lines.append(
            f"total: buy={_f(tf.get('buy_quote', 0), 0)} "
            f"sell={_f(tf.get('sell_quote', 0), 0)} "
            f"delta={_f(tf.get('delta_quote', 0), 0)}  "
            f"coverage={_f(tf.get('coverage_minutes'), 1)}min"
        )

        # ── structural windows (15m / 30m) ──
        windows = tf.get("windows", {})
        for label in ("15m", "30m"):
            w = windows.get(label, {})
            if not w:
                continue
            ratio = w.get("coverage_ratio", 1.0)
            cov_flag = " [LOW_COV]" if float(ratio) < 0.5 else ""
            lines.append(
                f"{label}: buy={_f(w.get('buy_quote', 0), 0)} "
                f"sell={_f(w.get('sell_quote', 0), 0)} "
                f"delta={_f(w.get('delta_quote', 0), 0)}  "
                f"cov={_f(ratio, 2)}{cov_flag}"
            )

        # Kline-based delta (30m / 1h — structural)
        kf = tf.get("kline_flow", {})
        if kf.get("available"):
            for label in ("30m", "1h"):
                w = kf.get("windows", {}).get(label, {})
                if w:
                    lines.append(
                        f"kline_{label}: buy={_f(w.get('buy_base', 0), 2)}BTC "
                        f"sell={_f(w.get('sell_base', 0), 2)}BTC "
                        f"delta={_f(w.get('delta_qty', 0), 2)}BTC"
                    )

        # key-level flow (structural: validated support/resistance reaction)
        klf = tf.get("key_level_flows", [])
        if klf:
            lines.append("key_level_flow:")
            seen_names: set = set()
            for kl in klf:
                dedup_key = f"{kl.get('name')}@{kl.get('price')}"
                if dedup_key in seen_names:
                    continue
                seen_names.add(dedup_key)
                # Base flow line
                flow_line = (
                    f"  @{kl['price']}({kl['name']}): "
                    f"buy={_f(kl.get('buy', 0), 0)} "
                    f"sell={_f(kl.get('sell', 0), 0)} "
                    f"net={_f(kl.get('net', 0), 0)} -> {kl.get('tag', '?')}"
                )
                # Append test history if available
                tests = kl.get("tests_12h")
                if tests is not None:
                    first_ago = kl.get("first_test_min_ago")
                    bounce = kl.get("avg_bounce_pct")
                    test_str = f"  tests={tests}"
                    if first_ago is not None:
                        test_str += f"  first={first_ago}min_ago"
                    if bounce is not None:
                        test_str += f"  bounce_avg={bounce}%"
                    flow_line += test_str
                lines.append(flow_line)

        # ── EPHEMERAL micro signals (timing/veto only — cannot establish direction) ──
        lines.append("")
        lines.append(
            "--- ephemeral (veto/timing only, "
            "DO NOT use to establish direction) ---"
        )

        for label in ("1m", "5m"):
            w = windows.get(label, {})
            if not w:
                continue
            ratio = w.get("coverage_ratio", 1.0)
            cov_flag = " [LOW_COV]" if float(ratio) < 0.5 else ""
            lines.append(
                f"[ephemeral] {label}: buy={_f(w.get('buy_quote', 0), 0)} "
                f"sell={_f(w.get('sell_quote', 0), 0)} "
                f"delta={_f(w.get('delta_quote', 0), 0)}  "
                f"cov={_f(ratio, 2)}{cov_flag}"
            )

        al = tf.get("aggressor_layers", {})
        if al:
            parts = []
            for bucket in ("large", "block"):
                b = al.get(bucket, {})
                if b:
                    parts.append(
                        f"{bucket}: buy={_f(b.get('buy_quote', 0), 0)} "
                        f"sell={_f(b.get('sell_quote', 0), 0)} "
                        f"delta={_f(b.get('delta_quote', 0), 0)}"
                    )
            if parts:
                lines.append("[ephemeral] aggressor_large_block: " + " | ".join(parts))

        clusters = tf.get("large_trade_clusters", [])
        if clusters:
            top = clusters[:2]
            lines.append(
                "[ephemeral] clusters(top2): "
                + " ; ".join(
                    f"@{c.get('center_price')} net={_f(float(c.get('buy_quote',0))-float(c.get('sell_quote',0)),0)} "
                    f"total={_f(c.get('total_quote', 0), 0)}"
                    for c in top
                )
            )

        pld = tf.get("price_level_delta", {})
        if pld.get("available"):
            cov = pld.get("actual_coverage_minutes", 0)
            if float(cov) >= 5:
                si = pld.get("stacked_imbalance", [])
                if si:
                    s = si[0]
                    lines.append(
                        f"[ephemeral] footprint_stacked: "
                        f"{s['from_price']}-{s['to_price']}  "
                        f"side={s.get('direction')} ({s.get('count')} bins)"
                    )

        return lines

    # ─── 7. DERIVATIVES ─────────────────────────────────────────────────────

    @staticmethod
    def _derivatives(ctx: Dict) -> List[str]:
        lines = ["=== DERIVATIVES ==="]

        funding = ctx.get("funding", {})
        basis = ctx.get("basis", {})
        lines.append(f"funding_rate: {_v(funding.get('funding_rate'))}")
        lines.append(f"next_funding: {_v(funding.get('next_funding_time'))}")
        lines.append(
            f"mark_price={_f(funding.get('mark_price'), 1)}  "
            f"index_price={_f(funding.get('index_price'), 1)}"
        )
        lines.append(
            f"basis_abs={_f(basis.get('basis_abs'), 2)}  "
            f"basis_bps={_f(basis.get('basis_bps'), 2)}"
        )

        # OI — raw values + per-period breakdown
        oi = ctx.get("open_interest_trend", {})
        lines.append(
            f"OI_current={_v(ctx.get('open_interest'))}  "
            f"delta%={_f(oi.get('delta_pct'), 3)}  "
            f"vs_avg%={_f(oi.get('vs_avg_pct'), 3)}"
        )
        for period in ("5m", "15m", "1h"):
            p = oi.get("periods", {}).get(period, {})
            if p:
                lines.append(
                    f"  OI_{period}: delta={_f(p.get('delta_abs'), 1)}  "
                    f"delta%={_f(p.get('delta_pct'), 3)}  "
                    f"vs_avg%={_f(p.get('vs_avg_pct'), 3)}"
                )

        # Long/short ratios — top trader only (smart money); global L/S demoted to weak_ref
        ls = ctx.get("long_short_ratio", {})
        tt = ls.get("top_trader_position", {})
        lines.append(
            f"top_trader_L/S: ratio={_f(tt.get('latest_ratio'))}  "
            f"long%={_f(tt.get('long_account'), 4)}  "
            f"short%={_f(tt.get('short_account'), 4)}  "
            f"avg={_f(tt.get('avg_ratio'))}  delta%={_f(tt.get('delta_pct'), 2)}"
        )

        # Cross-exchange OI
        cx = ctx.get("cross_exchange_oi", {})
        src = cx.get("sources", {})
        oi_parts = []
        for ex in ("binance", "okx", "bybit"):
            s = src.get(ex, {})
            if s.get("available"):
                oi_parts.append(f"{ex}={_v(s.get('oi'))}{s.get('unit', '')}")
            else:
                oi_parts.append(f"{ex}=unavailable")
        lines.append("cross_exchange_OI: " + " | ".join(oi_parts))

        # Cross-exchange funding rates (raw)
        cef = ctx.get("cross_exchange_funding", {})
        cf_parts = []
        for ex in ("binance", "bybit", "okx"):
            e = cef.get(ex, {})
            if e.get("available"):
                cf_parts.append(f"{ex}={e.get('funding_rate')}")
            else:
                cf_parts.append(f"{ex}=unavailable")
        if cf_parts:
            lines.append("cross_exchange_funding_rate: " + " | ".join(cf_parts))

        # Funding spread (raw bps only)
        spread = ctx.get("funding_spread", {})
        if spread.get("available_count", 0) >= 2:
            lines.append(
                f"funding_spread: {_f(spread.get('max_spread_bps'), 4)}bps  "
                f"({spread.get('highest_exchange')} vs {spread.get('lowest_exchange')})"
            )

        return lines

    # ─── 7b. TRANSITION DYNAMICS ────────────────────────────────────────────

    @staticmethod
    def _transition_dynamics(ctx: Dict) -> List[str]:
        lines = ["=== TRANSITION DYNAMICS (change rates & regime shifts) ==="]
        tr = ctx.get("transition", {})
        if not tr:
            lines.append("transition_data: unavailable")
            return lines

        oi = tr.get("oi_rates", {})
        for period in ("5m", "15m", "1h"):
            r = oi.get(period, {})
            lines.append(
                f"OI_{period}: delta%={_f(r.get('delta_pct'), 4)}  "
                f"velocity={_f(r.get('velocity'), 1)}  "
                f"acceleration={_f(r.get('acceleration'), 1)}"
            )

        bd = tr.get("basis_dynamics", {})
        lines.append(
            f"basis: {_f(bd.get('basis_bps'), 2)}bps  regime={_v(bd.get('regime'))}"
        )

        fd = tr.get("funding_dynamics", {})
        lines.append(
            f"funding: rate={_f(fd.get('rate_pct'), 4)}%  "
            f"regime={_v(fd.get('regime'))}  intensity={_v(fd.get('intensity'))}"
        )

        cd = tr.get("cvd_dynamics", {})
        lines.append(
            f"CVD: slope_5m={_f(cd.get('slope_5m'), 4)}  "
            f"slope_15m={_f(cd.get('slope_15m'), 4)}  "
            f"momentum={_v(cd.get('momentum'))}"
        )

        # CVD divergence
        cvd_div = tr.get("cvd_divergence")
        if cvd_div:
            parts = []
            for window_key in ("1h_window", "15m_window"):
                d = cvd_div.get(window_key, {})
                div_type = d.get("type", "no_divergence")
                strength = d.get("strength", "none")
                label = f"{window_key}={div_type}" + (f"({strength})" if strength != "none" else "")
                parts.append(label)
            lines.append("cvd_divergence: " + "  ".join(parts))

        return lines

    # ─── 8. SPOT VS PERP ────────────────────────────────────────────────────

    @staticmethod
    def _spot_vs_perp(ctx: Dict) -> List[str]:
        sp = ctx.get("spot_perp", {})
        if not sp.get("available"):
            return ["=== SPOT VS PERP ===", _NA]

        lines = ["=== SPOT VS PERP ==="]
        lines.append(
            f"spot_price={_v(sp.get('spot_price'))}  perp_price={_v(sp.get('perp_price'))}"
        )
        lines.append(
            f"basis_abs={_f(sp.get('basis_abs'), 2)}  "
            f"basis_bps={_f(sp.get('basis_bps'), 2)}"
        )
        s5 = sp.get("spot_5m", {})
        s15 = sp.get("spot_15m", {})
        lines.append(
            f"spot_5m: buy={_f(s5.get('buy_quote', 0), 0)} "
            f"sell={_f(s5.get('sell_quote', 0), 0)} "
            f"delta={_f(s5.get('delta_quote', 0), 0)}"
        )
        lines.append(
            f"spot_15m: buy={_f(s15.get('buy_quote', 0), 0)} "
            f"sell={_f(s15.get('sell_quote', 0), 0)} "
            f"delta={_f(s15.get('delta_quote', 0), 0)}"
        )
        lines.append(f"spot_vol_24h_quote={_f(sp.get('spot_vol_24h_quote', 0), 0)}")
        lines.append(f"spot_cvd_qty={_f(sp.get('spot_cvd_qty'), 4)}")

        return lines

    # ─── 9. POSITION FACTS ──────────────────────────────────────────────────

    @staticmethod
    def _position_facts(ctx: Dict) -> List[str]:
        lines = ["=== POSITION FACTS ==="]

        acc = ctx.get("account_positions", {})
        if not acc.get("available"):
            lines.append(
                "has_open_position: unavailable "
                "(API credentials missing or 401 — position-aware logic disabled; "
                "treat as NO known position but acknowledge blind spot)"
            )
            return lines

        sym = acc.get("symbol_position")
        if not sym or abs(float(sym.get("position_amt", 0) or 0)) == 0:
            lines.append("has_open_position: false")
            sizing = ctx.get("position_sizing", {})
            if sizing and sizing.get("available"):
                lines.append(
                    f"ATR_ref: ATR({_v(sizing.get('atr_timeframe'))})={_f(sizing.get('atr'), 1)}  "
                    f"1.5xATR_dist={_f(sizing.get('sl_distance'), 1)}  "
                    f"ATR%={_f(sizing.get('sl_pct'), 3)}%"
                )
            return lines

        notional = abs(float(sym.get("notional", 0) or 0))
        lines.append("has_open_position: true")
        lines.append(f"side: {_v(sym.get('side'))}")
        lines.append(f"size_btc: {_v(sym.get('position_amt'))}")
        lines.append(f"entry_price: {_v(sym.get('entry_price'))}")
        lines.append(f"mark_price: {_v(sym.get('mark_price'))}")
        lines.append(f"leverage: {_v(sym.get('leverage'))}x")
        lines.append(f"liquidation_price: {_v(sym.get('liquidation_price'))}")
        lines.append(f"unrealized_pnl: {_v(sym.get('unrealized_pnl'))}")
        lines.append(f"notional_usdt: {notional:.2f}")
        lines.append(f"margin_type: {_v(sym.get('margin_type'))}")

        # ATR reference — raw distance only, no directional label
        sizing = ctx.get("position_sizing", {})
        if sizing and sizing.get("available"):
            lines.append(
                f"ATR_ref: ATR({_v(sizing.get('atr_timeframe'))})={_f(sizing.get('atr'), 1)}  "
                f"1.5xATR_dist={_f(sizing.get('sl_distance'), 1)}  "
                f"ATR%={_f(sizing.get('sl_pct'), 3)}%"
            )

        return lines

    # ─── EXTERNAL DRIVERS ────────────────────────────────────────────────────

    @staticmethod
    def _external_drivers(ctx: Dict) -> List[str]:
        lines = ["=== EXTERNAL DRIVERS ==="]
        ext = ctx.get("external_drivers", {})
        if not ext:
            lines.append("external_data: unavailable")
            return lines

        # ── high-value for intraday perp ──
        etf = ext.get("etf_flow", {})
        if etf.get("available"):
            flow = etf.get("total_net_flow_usd", 0)
            lines.append(
                f"btc_etf_flow: date={etf.get('date', 'N/A')}  "
                f"net_flow=${flow:,.0f}"
            )
        else:
            lines.append(f"btc_etf_flow: unavailable ({etf.get('reason', '')})")

        fg = ext.get("fear_greed", {})
        if fg.get("available"):
            lines.append(
                f"fear_greed_index: {fg.get('value', 'N/A')} "
                f"({fg.get('classification', '')})"
            )

        # options_iv / skew: not yet integrated
        lines.append("options_iv: not_integrated")

        return lines

    # ─── PRIOR DECISIONS CONTEXT ─────────────────────────────────────────────

    @staticmethod
    def _raw_appendix(ctx: Dict) -> List[str]:
        raw = ctx.get("raw_appendix")
        if not isinstance(raw, dict) or not raw:
            return ["raw_appendix: unavailable"]
        return [
            "=== RAW APPENDIX ===",
            "# 仅把这里当作底层证据补充，不得覆盖主面板里的结构优先级规则。",
            json.dumps(raw, ensure_ascii=False, indent=2),
        ]

    @staticmethod
    def _research_handoff_block(research_handoff: str) -> List[str]:
        payload = (research_handoff or "").strip() or '{"stage": "research", "note": "empty_handoff"}'
        return [
            "╔══════════════════════════════════════════════════════════╗",
            "║  RESEARCH HANDOFF  (研究阶段输出，供最终决策阶段消费)      ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            payload,
        ]

    @staticmethod
    def _prior_decisions_context(ctx: Dict) -> List[str]:
        """Inject recent AI call history for bias calibration.

        Only renders if context["prior_decisions"] is a non-empty string.
        The block is placed at the END of SECTION 1 so that the AI reads
        the raw market data independently first (Phase A), then uses this
        block purely to detect systematic directional bias.
        """
        block = ctx.get("prior_decisions", "")
        if not block or not block.strip():
            return []
        return [block]

    # ─── [SECTION 3] DERIVED_SIGNALS (weak reference only) ──────────────────

    @staticmethod
    def _derived_signals(ctx: Dict) -> List[str]:
        """State labels inferred from raw data.  All prefixed weak_ref_ to
        signal that these MUST NOT anchor the primary direction judgment."""
        lines = ["=== weak_ref: DERIVED_SIGNALS ==="]

        # OI derived state labels (e.g. new_longs / new_shorts)
        oi = ctx.get("open_interest_trend", {})
        lines.append(f"weak_ref_OI_composite_signal: {_v(oi.get('composite_signal'))}")
        lines.append(f"weak_ref_OI_latest_interpretation: {_v(oi.get('latest_interpretation'))}")
        voi = oi.get("volume_oi_cvd_state", {})
        if voi:
            lines.append(
                f"weak_ref_OI_volume_oi_cvd_state: vol={_v(voi.get('volume_state'))}  "
                f"oi={_v(voi.get('oi_state'))}  cvd={_v(voi.get('cvd_state'))}"
            )

        # L/S crowding labels (global demoted here; top_trader in main DERIVATIVES section)
        ls = ctx.get("long_short_ratio", {})
        ga = ls.get("global_account", {})
        tt = ls.get("top_trader_position", {})
        lines.append(
            f"weak_ref_global_L/S: ratio={_f(ga.get('latest_ratio'))}  "
            f"long%={_f(ga.get('long_account'), 4)}  avg={_f(ga.get('avg_ratio'))}  "
            f"delta%={_f(ga.get('delta_pct'), 2)}  crowding={_v(ga.get('crowding'))}"
        )
        lines.append(
            f"weak_ref_LS_crowding: global={_v(ga.get('crowding'))}  "
            f"top_trader={_v(tt.get('crowding'))}  "
            f"overall={_v(ls.get('overall_crowding'))}"
        )

        # Spot/perp derived interpretation
        sp = ctx.get("spot_perp", {})
        if sp.get("available"):
            lines.append(f"weak_ref_spot_perp_basis_signal: {_v(sp.get('basis_signal'))}")
            lines.append(f"weak_ref_spot_perp_cvd_state: {_v(sp.get('cvd_state'))}")
            lines.append(f"weak_ref_spot_perp_interpretation: {_v(sp.get('interpretation'))}")

        # Funding spread signal label
        spread = ctx.get("funding_spread", {})
        if spread.get("signal"):
            lines.append(f"weak_ref_funding_spread_signal: {_v(spread.get('signal'))}")

        # Market structure labels
        ms = ctx.get("market_structure", {})
        if ms:
            lines.append(f"weak_ref_market_structure: {ms}")

        # Signal score (composite bias)
        ss = ctx.get("signal_score", {})
        if ss:
            lines.append(
                f"weak_ref_signal_score: composite={_v(ss.get('composite_score'))}  "
                f"bias={_v(ss.get('bias'))}  strength={_v(ss.get('strength'))}"
            )
            for k, v in (ss.get("components") or {}).items():
                score_val = v.get("score") if isinstance(v, dict) else v
                lines.append(f"  weak_ref_signal.{k}: score={_v(score_val)}")

        return lines

    # ─── [SECTION 4] DEPLOYMENT_HINTS (do not use for direction) ────────────

    @staticmethod
    def _deployment_hints(ctx: Dict) -> List[str]:
        """Pre-computed plan zones, entry sides, deployment context.
        Prefixed optional_plan_hint_ — must NOT be used to decide direction."""
        lines = ["=== optional_plan_hint: DEPLOYMENT_HINTS ==="]

        deploy = ctx.get("deployment_context", {})
        if deploy:
            lines.append(
                f"optional_plan_hint_primary_bias: {_v(deploy.get('primary_bias'))}"
            )
            lines.append(
                f"optional_plan_hint_transition_state: {_v(deploy.get('transition_state'))}"
            )
            lines.append(
                f"optional_plan_hint_deployment_score: {_v(deploy.get('deployment_score'))} "
                f"({_v(deploy.get('deployment_score_value'))})"
            )
            lines.append(
                f"optional_plan_hint_state_tags: {deploy.get('state_tags', [])}"
            )
            exec_ctx = deploy.get("execution_context", {})
            if exec_ctx:
                lines.append(f"optional_plan_hint_execution_context: {exec_ctx}")

            pz = deploy.get("plan_zones", {})
            if pz:
                for zone_key, zone_val in pz.items():
                    if isinstance(zone_val, dict):
                        lines.append(
                            f"  optional_plan_hint_zone.{zone_key}"
                            f" (entry.side={_v(zone_val.get('side'))}): "
                            f"{_f(zone_val.get('zone_low'), 1)}-{_f(zone_val.get('zone_high'), 1)}"
                        )

        # Position sizing directional SL/TP reference (hypothesis, not ground truth)
        sizing = ctx.get("position_sizing", {})
        if sizing and sizing.get("available"):
            ref = sizing.get("reference_levels", {})
            for side in ("long", "short"):
                r = ref.get(side, {})
                if r:
                    lines.append(
                        f"optional_plan_hint_sizing.{side}: "
                        f"SL={_f(r.get('stop_loss'), 1)}  "
                        f"TP1={_f(r.get('tp1'), 1)}  TP2={_f(r.get('tp2'), 1)}  "
                        f"RR1={_f(r.get('risk_reward_tp1'))}  RR2={_f(r.get('risk_reward_tp2'))}"
                    )

        return lines
