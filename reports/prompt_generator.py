from typing import Dict


class PromptGenerator:
    """Build a concise BTC trading prompt from market context."""

    SYSTEM_PROMPT = (
        "你是 BTC 永续合约交易助手。基于下方数据，给出可执行的 4h 波段交易计划。\n"
        "规则：\n"
        "- 以分批挂单为主，不追市价单\n"
        "- 目标持仓几小时到半天，目标盈利 2-5%\n"
        "- 仓位用账户百分比表示，不要建议杠杆倍数\n"
        "- 结构不清或盈亏比低于 1.5:1 就建议观望\n"
        "- 不要反问，不要教学，不要免责声明\n"
        "- 若有持仓数据，必须给出持仓操作建议"
    )

    OUTPUT_FORMAT = (
        "严格按以下格式输出：\n"
        "\n"
        "【方向】做多 / 做空 / 观望\n"
        "【市场状态】趋势 / 震荡 / 修正\n"
        "【信号强度】强 / 中 / 弱\n"
        "\n"
        "【分批挂单】\n"
        "  第1档: 价格=xxx, 仓位=账户xx%\n"
        "  第2档: 价格=xxx, 仓位=账户xx%\n"
        "  (可选第3档)\n"
        "【止损】价格\n"
        "【止盈】TP1=价格(约x%), TP2=价格(约x%)\n"
        "【预计等待】大致窗口\n"
        "\n"
        "【持仓建议】\n"
        "  当前状态: 多/空/空仓\n"
        "  操作: 持有/加仓/减仓x%/移动止损到xxx/平仓\n"
        "  平仓信号: 什么条件下平仓\n"
        "\n"
        "【理由】3-5行，包含：趋势配合、关键位、流动性方向、主要风险\n"
        "\n"
        "若观望，挂单写「无」，理由写为什么不做。\n"
        "持仓建议必须写，空仓也要写「当前空仓，等待入场」。"
    )

    def build(self, context: Dict) -> str:
        lines: list[str] = [
            self.SYSTEM_PROMPT,
            "",
            self.OUTPUT_FORMAT,
            "",
        ]

        lines.extend(self._build_snapshot(context))
        lines.append("")
        lines.extend(self._build_technicals(context))
        lines.append("")
        lines.extend(self._build_microstructure(context))
        lines.append("")
        lines.extend(self._build_derivatives_compact(context))
        lines.append("")
        lines.extend(self._build_sizing(context))
        lines.append("")
        lines.extend(self._build_position_status(context))

        return "\n".join(lines).strip()

    @staticmethod
    def _build_snapshot(ctx: Dict) -> list[str]:
        price = ctx.get("price", 0)
        signal = ctx.get("signal_score", {})
        session = ctx.get("session_context", {})
        stats = ctx.get("stats_24h", {})
        r4h = ctx.get("recent_4h_range", {})
        deploy = ctx.get("deployment_context", {})
        structure = ctx.get("market_structure", {})

        score = signal.get("composite_score", "?")
        bias = signal.get("bias", "?")
        strength = signal.get("strength", "?")

        trend_parts = [f"{tf}={structure.get(tf, '?')}" for tf in ("4h", "1h", "15m")]

        lines = [
            "=== 快照 ===",
            f"价格: {price}  评分: {score}/100 ({bias}, {strength})",
            f"趋势: {' | '.join(trend_parts)}",
            f"24h: high={stats.get('high_price')} low={stats.get('low_price')} chg={stats.get('price_change_percent')}%",
            f"4h区间: {r4h.get('high')}-{r4h.get('low')} (range {r4h.get('range_pct', 0):.2f}%)",
            f"Session: {session.get('current_session', '?').upper()} | Funding: {session.get('funding_countdown_label', '?')}",
        ]
        if deploy:
            lines.append(
                f"部署: bias={deploy.get('primary_bias')} "
                f"state={deploy.get('transition_state')} "
                f"score={deploy.get('deployment_score')}({deploy.get('deployment_score_value')})"
            )
        return lines

    @staticmethod
    def _build_technicals(ctx: Dict) -> list[str]:
        lines = ["=== 技术指标 ==="]
        for tf, m in ctx.get("timeframes", {}).items():
            feat = m.get("features", {})
            ema = m.get("ema", {})
            rsi = m.get("rsi", {})
            atr = m.get("atr", {})
            macd = m.get("macd", {})
            bb = m.get("bollinger", {})

            lines.append(
                f"[{tf}] trend={feat.get('trend', '?')} mom={feat.get('momentum', '?')} | "
                f"RSI={rsi.get('14', '?')}({rsi.get('state', '')}) div={rsi.get('divergence', 'none')} | "
                f"MACD_hist={macd.get('hist', '?')} | "
                f"ATR={atr.get('atr', '?')}({atr.get('atr_pct', '?')}%) | "
                f"BB%B={bb.get('percent_b', '?')} bw={bb.get('bandwidth', '?')} | "
                f"EMA7={ema.get('7')} E25={ema.get('25')} E99={ema.get('99')}"
            )

        vp = ctx.get("volume_profile", {})
        if vp.get("poc_price"):
            lines.append(f"VP: POC={vp.get('poc_price')} HVN={vp.get('hvn_prices')} LVN={vp.get('lvn_prices')}")

        return lines

    @staticmethod
    def _build_microstructure(ctx: Dict) -> list[str]:
        lines = ["=== 盘口与成交流 ==="]

        ob = ctx.get("orderbook", {})
        if ob:
            lines.append(
                f"盘口: bid_vol={ob.get('bid_volume')} ask_vol={ob.get('ask_volume')} "
                f"imbalance={ob.get('imbalance')}"
            )
            bw = ob.get("bid_wall", {})
            aw = ob.get("ask_wall", {})
            if isinstance(bw, dict) and isinstance(aw, dict):
                lines.append(
                    f"挂单墙: bid@{bw.get('price')}({bw.get('qty')}) "
                    f"ask@{aw.get('price')}({aw.get('qty')})"
                )

        od = ctx.get("orderbook_dynamics", {})
        if od:
            lines.append(
                f"盘口动态: spoofing={od.get('spoofing_risk', '?')} "
                f"wall_behavior={od.get('wall_behavior', '?')} "
                f"absorption={od.get('passive_absorption_quote', 0)} "
                f"pull={od.get('pull_without_trade_quote', 0)}"
            )

        tf = ctx.get("trade_flow", {})
        if tf:
            w5 = tf.get("windows", {}).get("5m", {})
            lines.append(
                f"成交流(5m): buy={w5.get('buy_quote', 0):.0f} sell={w5.get('sell_quote', 0):.0f} "
                f"delta={w5.get('delta_quote', 0):.0f} cvd={w5.get('cvd_qty', 0):.3f} "
                f"大单={w5.get('large_trade_direction', tf.get('large_trade_direction', '?'))}"
            )
            clusters = tf.get("large_trade_clusters", [])
            if clusters:
                top = clusters[0]
                lines.append(
                    f"最大聚集: price={top.get('center_price')} "
                    f"quote={top.get('total_quote', 0):.0f} "
                    f"side={top.get('dominant_side', '?')}"
                )

        liq = ctx.get("liquidation_heatmap", {})
        zones = liq.get("zones", [])
        near_zones = [z for z in zones if "recent_4h" in str(z.get("name", ""))]
        if near_zones:
            lines.append("近端流动性:")
            for z in near_zones:
                lines.append(
                    f"  {str(z.get('name', '')).replace('_', ' ')}: "
                    f"{z.get('zone_low')}-{z.get('zone_high')} "
                    f"({z.get('estimated_pressure', '?')})"
                )

        return lines

    @staticmethod
    def _build_derivatives_compact(ctx: Dict) -> list[str]:
        lines = ["=== 衍生品 ==="]

        funding = ctx.get("funding", {})
        lines.append(
            f"费率: {funding.get('funding_rate')} | "
            f"mark={funding.get('mark_price')} index={funding.get('index_price')}"
        )

        basis = ctx.get("basis", {})
        lines.append(f"基差: {basis.get('basis_bps', 0):.1f}bps ({basis.get('structure', '?')})")

        oi = ctx.get("open_interest_trend", {})
        lines.append(
            f"OI: {ctx.get('open_interest')} "
            f"trend={oi.get('trend', '?')} delta={oi.get('delta_pct', '?')}% "
            f"state={oi.get('latest_state', '?')} ({oi.get('latest_interpretation', '?')})"
        )
        voc = oi.get("volume_oi_cvd_state", {})
        if isinstance(voc, dict) and voc:
            lines.append(
                f"OI综合: vol={voc.get('volume_state', '?')} "
                f"oi={voc.get('oi_state', '?')} cvd={voc.get('cvd_state', '?')}"
            )

        ls = ctx.get("long_short_ratio", {})
        ga = ls.get("global_account", {})
        tt = ls.get("top_trader_position", {})
        lines.append(
            f"多空: global={ga.get('latest_ratio', '?')} "
            f"top={tt.get('latest_ratio', '?')} "
            f"crowding={ls.get('overall_crowding', '?')}"
        )

        spread = ctx.get("funding_spread", {})
        if spread.get("available_count", 0) >= 2:
            lines.append(
                f"跨所费率差: {spread.get('max_spread_bps', 0):.2f}bps "
                f"({spread.get('signal', '?')})"
            )

        return lines

    @staticmethod
    def _build_sizing(ctx: Dict) -> list[str]:
        sizing = ctx.get("position_sizing", {})
        if not sizing or not sizing.get("available"):
            return []
        ref = sizing.get("reference_levels", {})
        lr = ref.get("long", {})
        sr = ref.get("short", {})
        return [
            "=== ATR仓位参考 ===",
            f"ATR({sizing.get('atr_timeframe')})={sizing.get('atr')} SL距离={sizing.get('sl_distance')} SL%={sizing.get('sl_pct')}",
            f"做多: SL={lr.get('stop_loss')} TP1={lr.get('tp1')} TP2={lr.get('tp2')}",
            f"做空: SL={sr.get('stop_loss')} TP1={sr.get('tp1')} TP2={sr.get('tp2')}",
        ]

    @staticmethod
    def _build_position_status(ctx: Dict) -> list[str]:
        acc = ctx.get("account_positions", {})
        if not acc.get("available"):
            return ["=== 持仓 ===", "仓位数据不可用，按空仓处理"]

        sym = acc.get("symbol_position")
        active_count = acc.get("active_positions_count", 0)

        if not sym or abs(float(sym.get("position_amt", 0))) == 0:
            return ["=== 持仓 ===", "当前空仓"]

        side = sym.get("side", "?")
        amt = sym.get("position_amt", 0)
        entry = sym.get("entry_price", 0)
        mark = sym.get("mark_price", 0)
        liq = sym.get("liquidation_price", 0)
        pnl = sym.get("unrealized_pnl", 0)
        lev = sym.get("leverage", 0)
        notional = abs(float(sym.get("notional", 0)))

        return [
            "=== 持仓 ===",
            f"方向={side} 数量={amt} 杠杆={lev}x",
            f"开仓价={entry} 标记价={mark} 强平价={liq}",
            f"持仓价值={notional:.2f}U 未实现盈亏={pnl}",
            f"（AI 必须基于此持仓给出：持有/加仓/减仓/调止损/平仓建议）",
        ]
