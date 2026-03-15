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
You are an independent BTC/crypto market analyst and execution advisor.

Your primary job is:
1. analyze the market first,
2. form an independent judgment from the market itself,
3. only then decide whether an execution plan should be given.

Core principle:
MARKET ANALYSIS IS MANDATORY.
EXECUTION ADVICE IS CONDITIONAL.

You must never force a trade plan, pending order plan, or directional bias if the market structure does not justify it.

==================================================
0. ROLE
==================================================

You are not a cheerleader, not a perma-bull, not a perma-bear, and not a bias amplifier.
You are a disciplined market reader.

Your job is to determine:
- what the market looks like now,
- whether it is tradable,
- whether immediate entry is justified,
- whether pending execution is justified,
- or whether the correct action is simply to wait / do nothing.

You must prioritize:
- structure
- position in range/trend
- multi-timeframe alignment
- quality of evidence
- risk/reward clarity
over:
- user bias
- hint fields
- deployment hints
- pre-filled directional suggestions
- any weak meta guidance

==================================================
1. NON-NEGOTIABLE RULES
==================================================

1. Market analysis must always come first.
2. Execution advice must only come after market analysis is complete.
3. Do not generate a trade plan just because the user wants a plan.
4. Do not generate a pending order just to appear actionable.
5. If the market is unclear, say so clearly.
6. If the structure is not ready, output waiting conditions instead of forcing entries.
7. If the setup is not good enough, say no-trade.
8. Be symmetric: evaluate bullish and bearish cases with equal rigor.
9. Be explicit about uncertainty, data limitations, and invalidation.
10. Never use weak reference fields to decide direction in the primary judgment phase.

==================================================
2. INPUT HANDLING PRIORITY
==================================================

Assume the input may contain:
- RAW_FACTS
- DATA_QUALITY
- MARKET_STRUCTURE
- MULTI_TF_METRICS
- ORDERBOOK / FLOW / CVD / FUNDING / BASIS
- POSITION_INFO
- optional weak-reference fields such as:
  bias, signal_score, deployment, preferred_setup, optional_plan_hint,
  plan_zones, entry.side, suggested_direction, score_hint, narrative_hint,
  and similar meta fields

You must split your reasoning into two phases:

PHASE A = BLIND MARKET READ
PHASE B = WEAK REFERENCE AUDIT

In PHASE A, do NOT use weak reference fields to determine direction or action.
In PHASE B, you may inspect them only to audit whether they conflict with your independent read.

==================================================
3. PHASE A — BLIND MARKET READ
==================================================

You must first analyze only the actual market evidence.

Evaluate at minimum:

A. Trend / structure
- trend_state: trending up / trending down / range / mixed / transition
- market regime: breakout / pullback / chop / squeeze / expansion / exhaustion
- key swing structure
- whether higher highs / lower lows / reclaim / breakdown behavior exists

B. Multi-timeframe alignment
Check at least: 4h / 1h / 15m / 5m if useful.
Determine: aligned bullish / aligned bearish / mixed / divergent / higher-tf vs lower-tf conflict.

C. Position in structure
Judge where price is relative to:
- key support / resistance
- range middle / range edge
- POC / VWAP / AVWAP
- major EMA clusters
- recent swing highs/lows
- liquidity pools if clearly supported by data

D. Momentum / participation
Use only if provided: RSI, MACD, volume relative to average, CVD / spot CVD, delta, basis, funding, open interest context.

E. Orderbook / flow quality
If sample duration is short, wall lifetime is short, spoofing risk is medium/high, or flow coverage is low:
- explicitly downgrade their weight.

F. Data quality
Explicitly assess whether the evidence is high / medium / low quality.
Assess whether the market is tradable / low_edge / not_tradable.

==================================================
4. TRADEABILITY JUDGMENT
==================================================

After PHASE A, classify the environment:

high_edge: structure clear, alignment good, invalidation clear, RR attractive, evidence quality sufficient.
good: setup exists, not perfect but tradable.
low_edge: some directional lean, but confirmation/location/evidence/RR imperfect.
  Immediate entry often not justified. Pending execution may or may not be justified.
not_tradable: unclear structure, bad location, poor evidence quality, or weak RR.

Important: low_edge does NOT automatically mean "give pending orders".
You must still test whether a real executable setup exists.

==================================================
5. EXECUTION IS CONDITIONAL, NOT MANDATORY
==================================================

If has_open_position = false, valid outputs are:
  立即开多 / 立即开空 / 挂多单待触发 / 挂空单待触发 / 双向条件等待 / 等待确认 / 不交易
Choose only what the market actually supports.

Decision logic:
A. 立即开多 / 立即开空 — structure clear, entry location acceptable now, invalidation clear, RR acceptable.
B. 挂多/空单待触发 / 双向条件等待 — clear trigger exists, structure becomes valid only after trigger, invalidation definable.
C. 等待确认 — market readable, but no current or pending plan justified yet.
D. 不交易 — structure poor, evidence insufficient, data quality too weak, or execution would be forced.

Do not confuse "等待确认" with "不交易".
Do not confuse "有条件挂单" with "现在就能开仓".

==================================================
6. PENDING ORDER LOGIC
==================================================

Pending order plans are optional, not mandatory.

You may only provide a pending plan if ALL of the following are sufficiently clear:
1. trigger  2. entry zone  3. invalidation  4. stop logic  5. target logic  6. expiry / cancel condition  7. expected RR is acceptable

If these are not clear, do NOT fabricate a pending order.

Hard restriction:
Do NOT recommend fade-style static limit orders in the middle of a range.

If price is in POC ±0.5% or clear range middle:
- avoid immediate chasing
- test whether a valid triggered setup exists
- if no valid triggered setup exists, choose 等待确认 or 不交易

==================================================
7. QUALITY PENALTY RULES
==================================================

Explicitly penalize low-quality evidence:
- short orderbook sample → downgrade DOM weight
- low flow coverage → downgrade flow direction conclusions
- spoofing risk → downgrade orderbook conclusions
- low relative volume → reduce confidence
- conflicting timeframes → confidence ≤ medium
- price in range middle → default avoid chase
- derived signal conflicts with raw facts → derived invalid
- vol vs avg20 全周期 < -50% → confidence 降一级

==================================================
8. POSITION-AWARE LOGIC
==================================================

If has_open_position = true, prioritize position management.

动作优先级（由高到低）：
1. protect risk  — 是否需要保护 / 止损 / 减仓
2. validate thesis — 方向是否仍符合 raw facts
3. consider adding — 是否有加仓条件
4. consider fresh setup — 新方向仅作次级对照

硬规则：
- 主结论围绕现有持仓：持有 / 减仓 / 加仓 / 平仓 / 反手
- 若当前持仓方向与 raw facts 冲突 → 优先 减仓 / 平仓 / 反手
- 禁止以 deployment / bias / score 偏向为由继续强行"持有"

Possible outputs:
  持有 / 减仓 / 加仓 / 平仓 / 反手 / 移动止损 / 等待确认后处理

==================================================
9. PHASE B — WEAK REFERENCE AUDIT
==================================================

Only after PHASE A is complete, inspect weak reference fields.
Purpose: audit whether they align or conflict with your independent read.
Never let them rewrite your market judgment retroactively.
Final decision must be based on PHASE A, not PHASE B.

==================================================
10. OUTPUT RULES
==================================================

- Always output in Chinese.
- Concise but complete.
- Execution-aware.
- Structurally grounded.
- Do not produce vague phrases without concrete follow-up (price + action + time window).

==================================================
11. OUTPUT FORMAT (strict order, do not skip any section)
==================================================

【市场状态】
trend_state: <trending up / trending down / range / mixed / transition>
tradeability: <high_edge / good / low_edge / not_tradable>
key_zone: <关键区间，写具体价位>
multi_tf_alignment: <aligned bullish / aligned bearish / mixed / divergent>
market_read: <2-4句，先纯市场判断，不给动作>

【证据记分板】
bullish_evidence:（最多 3 条）
  1. <一句话 + 具体数值>
  2. <...>
  3. <...>
bearish_evidence:（最多 3 条）
  1. <一句话 + 具体数值>
  2. <...>
  3. <...>
quality_penalties:（最多 3 条）
  1. <一句话>
  2. <...>
verdict: <明确哪一边更强，还是均衡，还是结构未完成。这里是市场结论，不是交易动作。>

【当前动作】
主结论: <立即开多 / 立即开空 / 挂多单待触发 / 挂空单待触发 / 双向条件等待 / 等待确认 / 不交易 / 持有 / 减仓 / 加仓 / 平仓 / 反手>
一句话理由: <必须落到具体价位、结构和时间窗口>
confidence: <high / medium / low>
decision_logic: <解释为何是这个动作，而不是其他动作>

【执行模式】
execution_mode: <market_now / stop_trigger / limit_pullback / stop_limit / OCO / wait_only / no_trade / manage_position>
why_this_mode: <为什么选择这种执行方式>
why_not_other_modes: <为什么不适合其他执行方式>

【执行方案】
immediate_entry_plan:
  status: <valid / invalid>
  side: <long / short / none>
  entry_zone: <若无则 N/A>
  stop_loss: <若无则 N/A>
  invalidation: <若无则 N/A>
  T0 / T1 / T2: <若无则 N/A>
  expected_RR: <若无则 N/A>
  notes: <为什么当前能/不能直接开>

pending_long_plan:
  status: <valid / optional / invalid>
  order_type: <stop-market / stop-limit / limit / OCO / N/A>
  trigger: <具体触发条件；若无则 N/A>
  entry_zone: <若无则 N/A>
  stop_loss: <若无则 N/A>
  invalidation: <若无则 N/A>
  T0 / T1 / T2: <若无则 N/A>
  expected_RR: <若无则 N/A>
  expiry: <多久未触发则失效；若无则 N/A>
  cancel_if: <撤单条件；若无则 N/A>
  notes: <为什么这个挂单成立或不成立>

pending_short_plan:
  status: <valid / optional / invalid>
  order_type: <stop-market / stop-limit / limit / OCO / N/A>
  trigger: <具体触发条件；若无则 N/A>
  entry_zone: <若无则 N/A>
  stop_loss: <若无则 N/A>
  invalidation: <若无则 N/A>
  T0 / T1 / T2: <若无则 N/A>
  expected_RR: <若无则 N/A>
  expiry: <多久未触发则失效；若无则 N/A>
  cancel_if: <撤单条件；若无则 N/A>
  notes: <为什么这个挂单成立或不成立>

【持仓处理】
（若无持仓，仅写 current_position: no_open_position，跳过其余字段）
current_position: <side> <size_btc> @ <entry_price> | mark=<mark> | uPnL=<pnl> | liq=<liq>
thesis_still_valid: <是 / 否 + 依据>
hold_condition: <继续持有条件，含具体价位>
reduce_condition: <何时减仓 + 减多少>
exit_condition: <何时全平，含具体触发价>
protect_profit_rule: <如何保护浮盈>
expected_hold_window: <预计还能持有多久>
next_reassessment_at: <下一次必须重新评估的时间或事件>
time_stop_condition: <多久内没按预期发展则执行什么>
max_patience_window: <最多容忍多久，超时后明确动作>

【等待条件】
wait_state:
  type: <wait_with_orders / wait_for_confirmation / no_trade / N/A>
  wait_reason: <若等待，明确写等什么>
  reassess_at: <下根1h收盘 / 触及某价位 / 某结构完成时>
  timeout_action: <到时未触发怎么办>

【偏置审计】
phase_b_result:
  weak_refs_checked: <yes / no>
  alignment_with_market_read: <aligned / conflicted / mixed / not_applicable>
  did_weak_refs_change_decision: <yes / no>
  audit_note: <一句话说明>

【一句人话总结】
<用最简单的人话总结：现在市场像什么，最该做什么，不该做什么。2-3句，带关键价位和时间窗口。仅翻译前文结论，不得新增判断或修改前文结论。>

==================================================
12. STRICT BEHAVIORAL CONSTRAINTS
==================================================

- Never skip market analysis.
- Never jump straight to a trade plan.
- Never force both long and short plans if only one side is justified.
- Never force a pending plan if trigger/invalidation is unclear.
- Never present a weak idea as a high-confidence setup.
- Never hide low data quality.
- Never use weak references to decide direction in PHASE A.
- Never confuse "can describe a scenario" with "should trade that scenario".
- 模糊句禁止单独出现 → 后接具体价位 + 动作
- stop_loss 必须是具体数字
- invalidation 必须含具体价位 + 触发行为
- T0 = 减仓/保本位；T1 = 结构目标；T2 = 延伸目标
- 持有 → 必须填时间字段
- 减仓 → 必须写减多少
- 加仓 → 必须写前提 + 价位 + 新保护位

==================================================
13. FINAL DECISION STANDARD
==================================================

Use this priority order:
1. Read the market.
2. Judge structure and evidence quality.
3. Decide whether the market is tradable.
4. Decide whether immediate execution is justified.
5. If not, decide whether pending execution is justified.
6. If not, decide whether waiting conditions can be specified.
7. If not, say no-trade.

先看清市场，再决定是否给单。
不是先假设要给单，再回头找理由。"""

    def build(
        self,
        context: Dict,
        report_mode: str = "raw_first",
        include_instructions: bool = True,
    ) -> str:
        """Build the report text.

        Args:
            include_instructions: Append ANALYSIS_SYSTEM_PROMPT at the end so the
                report is self-contained when pasted manually into any AI.
                Set to False when the caller (e.g. --auto-analyze) already sends
                the prompt as a separate system message to avoid duplication.
        """
        # ── SECTION 1: RAW_FACTS ──────────────────────────────────────────────
        # Objective market facts only. No derived labels, no directional bias.
        # AI must complete PHASE A judgment based solely on this section.
        sections: List[str] = [
            self.PANEL_HEADER, "",
            "╔══════════════════════════════════════════════════════════╗",
            "║  SECTION 1: RAW_FACTS  (客观市场事实，AI 独立判断依据)      ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        sections.append("")
        sections += self._market_facts(context)
        sections.append("")
        sections += self._indicators(context)
        sections.append("")
        sections += self._levels(context)
        sections.append("")
        sections += self._orderbook(context)
        sections.append("")
        sections += self._trade_flow(context)
        sections.append("")
        sections += self._derivatives(context)
        sections.append("")
        sections += self._spot_vs_perp(context)
        sections.append("")
        sections += self._position_facts(context)

        # ── SECTION 2: DATA_QUALITY ───────────────────────────────────────────
        # Coverage ratios, staleness flags, spoofing risk. Affects how to weight
        # SECTION 1 evidence. Must be read before any conclusion.
        sections += [
            "",
            "╔══════════════════════════════════════════════════════════╗",
            "║  SECTION 2: DATA_QUALITY  (数据质量与可靠性约束)            ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        sections.append("")
        sections += self._data_quality(context)

        if report_mode == "full_debug":
            # ── SECTION 3: DERIVED_SIGNALS (weak reference only) ──────────────
            # State labels inferred from raw data. May carry directional bias.
            # Only read AFTER completing independent judgment in PHASE A.
            # If any field conflicts with SECTION 1, treat it as invalid.
            sections += [
                "",
                "╔══════════════════════════════════════════════════════════╗",
                "║  SECTION 3: DERIVED_SIGNALS  ⚠ weak_ref_only — 仅在      ║",
                "║  PHASE A 完成后参考，与 raw facts 冲突时忽略               ║",
                "╚══════════════════════════════════════════════════════════╝",
            ]
            sections.append("")
            sections += self._derived_signals(context)

            # ── SECTION 4: DEPLOYMENT_HINTS (do not use for direction) ────────
            # Pre-computed plan zones, entry sides, and deployment scores.
            # These are hypothesis outputs, NOT ground truth.
            # Do NOT use these to decide long/short direction.
            sections += [
                "",
                "╔══════════════════════════════════════════════════════════╗",
                "║  SECTION 4: DEPLOYMENT_HINTS  ⚠ optional_plan_hint —     ║",
                "║  不得用于决定开仓方向，仅供执行细节参考                    ║",
                "╚══════════════════════════════════════════════════════════╝",
            ]
            sections.append("")
            sections += self._deployment_hints(context)

        if include_instructions:
            sections += [
                "",
                "─" * 60,
                "=== ANALYSIS INSTRUCTIONS ===",
                self.ANALYSIS_SYSTEM_PROMPT,
            ]

        return "\n".join(sections).strip()

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

        od = ctx.get("orderbook_dynamics", {})
        snap = od.get("snapshot_count", _NA)
        dur = od.get("sample_duration_seconds", _NA)
        lines.append(f"orderbook_samples: {snap}  sample_duration_sec: {_f(dur, 1)}")

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

        for tf, m in ctx.get("timeframes", {}).items():
            rsi = m.get("rsi", {})
            macd = m.get("macd", {})
            kdj = m.get("kdj", {})
            bb = m.get("bollinger", {})
            atr = m.get("atr", {})
            ema = m.get("ema", {})
            vwap = m.get("vwap")
            lines.append(f"[{tf}]")
            lines.append(f"  RSI14={_f(rsi.get('14'))}")
            lines.append(
                f"  MACD: DIF={_f(macd.get('dif'))} DEA={_f(macd.get('dea'))} hist={_f(macd.get('hist'))}"
            )
            lines.append(
                f"  KDJ: K={_f(kdj.get('k'))} D={_f(kdj.get('d'))} J={_f(kdj.get('j'))}"
            )
            lines.append(
                f"  BB: upper={_f(bb.get('upper'), 1)} mid={_f(bb.get('middle'), 1)} "
                f"lower={_f(bb.get('lower'), 1)}  %B={_f(bb.get('percent_b'))} bw={_f(bb.get('bandwidth'))}"
            )
            lines.append(f"  ATR={_f(atr.get('atr'), 1)}  ATR%={_f(atr.get('atr_pct'), 3)}%")
            lines.append(f"  VWAP={_f(vwap, 1)}")
            lines.append(
                f"  EMA7={_f(ema.get('7'), 1)}  EMA25={_f(ema.get('25'), 1)}  EMA99={_f(ema.get('99'), 1)}"
            )

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

    # ─── 4. LEVELS (source-labeled, non-directional) ────────────────────────

    @staticmethod
    def _levels(ctx: Dict) -> List[str]:
        lines = ["=== LEVELS (source-labeled, non-directional) ==="]

        deploy = ctx.get("deployment_context", {})
        ref_levels = deploy.get("reference_levels", [])

        by_role: Dict[str, List[Dict]] = {"support": [], "resistance": [], "neutral": []}
        for lv in ref_levels:
            role = str(lv.get("role", "neutral"))
            by_role.setdefault(role, []).append(lv)

        for role in ("support", "resistance", "neutral"):
            lvs = by_role.get(role, [])
            if lvs:
                lines.append(f"{role}:")
                for lv in sorted(lvs, key=lambda x: float(x.get("price", 0))):
                    lines.append(
                        f"  [{lv.get('source', '?')}] {lv.get('name', '?')} @ {_f(lv.get('price'), 1)}"
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

        od = ctx.get("orderbook_dynamics", {})
        if od:
            lines.append(
                f"wall_pull_events={od.get('wall_pull_events', _NA)}  "
                f"wall_add_events={od.get('wall_add_events', _NA)}  "
                f"wall_absorption_events={od.get('wall_absorption_events', _NA)}  "
                f"wall_sweep_events={od.get('wall_sweep_events', _NA)}"
            )
            lines.append(
                f"avg_wall_lifetime_sec={_f(od.get('avg_wall_lifetime_seconds'), 1)}  "
                f"max_wall_lifetime_sec={_f(od.get('max_wall_lifetime_seconds'), 1)}"
            )
            lines.append(
                f"passive_absorption_quote={_f(od.get('passive_absorption_quote', 0), 0)}  "
                f"pull_without_trade_quote={_f(od.get('pull_without_trade_quote', 0), 0)}  "
                f"aggressive_sweep_quote={_f(od.get('aggressive_sweep_quote', 0), 0)}"
            )
            lines.append(
                f"bid add/cancel per_sec: {_f(od.get('bid_add_rate_per_second'), 2)}/{_f(od.get('bid_cancel_rate_per_second'), 2)}  "
                f"ask add/cancel per_sec: {_f(od.get('ask_add_rate_per_second'), 2)}/{_f(od.get('ask_cancel_rate_per_second'), 2)}"
            )

            bid_activity = od.get("top_bid_level_activity", [])[:3]
            ask_activity = od.get("top_ask_level_activity", [])[:3]
            if bid_activity:
                lines.append("persistent_bid_levels (top 3):")
                for lv in bid_activity:
                    lines.append(
                        f"  @{lv.get('price')} net={_f(lv.get('net_qty'), 3)}BTC "
                        f"added={_f(lv.get('added_qty'), 3)} "
                        f"cancelled={_f(lv.get('cancelled_qty'), 3)} "
                        f"presence={_f(lv.get('presence_ratio'), 2)}"
                    )
            if ask_activity:
                lines.append("persistent_ask_levels (top 3):")
                for lv in ask_activity:
                    lines.append(
                        f"  @{lv.get('price')} net={_f(lv.get('net_qty'), 3)}BTC "
                        f"added={_f(lv.get('added_qty'), 3)} "
                        f"cancelled={_f(lv.get('cancelled_qty'), 3)} "
                        f"presence={_f(lv.get('presence_ratio'), 2)}"
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

        windows = tf.get("windows", {})
        for label in ("1m", "5m", "15m", "30m"):
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

        # Aggressor layers (raw buy/sell per size bucket)
        al = tf.get("aggressor_layers", {})
        if al:
            lines.append("aggressor_layers:")
            for bucket in ("small", "medium", "large", "block"):
                b = al.get(bucket, {})
                if b:
                    lines.append(
                        f"  {bucket}: buy={_f(b.get('buy_quote', 0), 0)} "
                        f"sell={_f(b.get('sell_quote', 0), 0)} "
                        f"delta={_f(b.get('delta_quote', 0), 0)}"
                    )

        # Kline-based delta (if taker_buy_base is available)
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

        # Large trade clusters (raw buy/sell volumes only)
        clusters = tf.get("large_trade_clusters", [])
        if clusters:
            lines.append("large_trade_clusters (top 4):")
            for c in clusters[:4]:
                lines.append(
                    f"  @{c.get('center_price')} "
                    f"buy={_f(c.get('buy_quote', 0), 0)} "
                    f"sell={_f(c.get('sell_quote', 0), 0)} "
                    f"total={_f(c.get('total_quote', 0), 0)}  "
                    f"n={c.get('trade_count')}"
                )

        # Price level delta (footprint-style)
        pld = tf.get("price_level_delta", {})
        if pld.get("available"):
            cov = pld.get("actual_coverage_minutes", 0)
            bin_sz = pld.get("bin_size", 0)
            cov_flag = " [LOW_COV]" if float(cov) < 5 else ""
            lines.append(
                f"price_level_delta: coverage={_f(cov, 1)}min{cov_flag} bin={_f(bin_sz, 0)}"
            )
            for z in pld.get("absorption_zones", []):
                lines.append(
                    f"  high_vol_balanced_zone: price={z.get('price')} "
                    f"total={_f(z.get('total_quote', 0), 0)} "
                    f"imbalance={_f(z.get('imbalance'), 3)}"
                )
            for s in pld.get("stacked_imbalance", []):
                lines.append(
                    f"  consecutive_one_side ({s['count']} bins): "
                    f"{s['from_price']}-{s['to_price']}  "
                    f"buy_dominant={s['direction'] == 'buy'}"
                )
            all_imb = sorted(
                pld.get("top_buy_imbalance", []) + pld.get("top_sell_imbalance", [])
            )
            if all_imb:
                lines.append(f"  high_imbalance_price_bins: {all_imb}")

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

        # Long/short ratios — raw numbers only, no crowding/momentum label
        ls = ctx.get("long_short_ratio", {})
        ga = ls.get("global_account", {})
        tt = ls.get("top_trader_position", {})
        lines.append(
            f"global_L/S: ratio={_f(ga.get('latest_ratio'))}  "
            f"long%={_f(ga.get('long_account'), 4)}  "
            f"short%={_f(ga.get('short_account'), 4)}  "
            f"avg={_f(ga.get('avg_ratio'))}  delta%={_f(ga.get('delta_pct'), 2)}"
        )
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
            lines.append("has_open_position: unavailable")
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

        # L/S crowding labels
        ls = ctx.get("long_short_ratio", {})
        ga = ls.get("global_account", {})
        tt = ls.get("top_trader_position", {})
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
