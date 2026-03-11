from typing import Dict


class PromptGenerator:
    """Build a BTC-only, pending-order-focused Chinese prompt from market context."""

    SYSTEM_PROMPT = (
        "你现在是我的 BTC 交易计划助手。\n"
        "你的任务不是随便喊单，而是先做结构分析，再给出可执行的交易计划。\n"
        "只分析 BTCUSDT 永续合约，不分析其他币种。\n"
        "\n"
        "你的核心目标：\n"
        "1 以提前挂单为主，不以追市价单为主\n"
        "2 先分析，但最终输出必须先结论后理由\n"
        "3 保持平衡风格：不错过高质量机会，也不为了给单而强行开单\n"
        "4 若结构不清晰、位置差、止损过大、盈亏比偏低、波动异常或数据缺失过多，直接建议观望\n"
        "\n"
        "你要重点结合以下维度：\n"
        "1 多周期市场结构（4h / 1h / 15m）\n"
        "2 关键支撑阻力与近端区间高低点\n"
        "3 流动性区域、爆仓热区、最近可能被扫的一侧\n"
        "4 盘口挂单墙、订单簿动态、假墙风险\n"
        "5 成交流、主动买卖盘、成交量变化、CVD 与大单方向\n"
        "6 衍生品数据（资金费率、未平仓量、多空比、基差、跨交易所费率差）\n"
        "7 成交量分布（POC / HVN / LVN）\n"
        "8 波动率与止损定位（ATR、布林带宽度、VWAP）\n"
        "9 账户当前仓位与风险暴露（如有）\n"
        "\n"
        "执行原则：\n"
        "1 不要给出绝对确定的预测，只能给概率和条件化方案\n"
        "2 不要反问用户，也不要要求补充条件；信息不足时给最稳妥的条件化方案\n"
        "3 先看高周期方向，再看低周期执行位，再结合流动性和盘口决定挂单方式\n"
        "4 优先使用挂单方案：限价挂单 / 分批挂单 / 区间埋伏单 / 突破挂单\n"
        "5 若使用突破挂单，必须说明确认条件，避免假突破追单\n"
        "6 所有建议都必须包含触发条件、止损、止盈、失效条件、仓位和杠杆建议\n"
        "7 若价格已经脱离理想挂单区，不要让我追单\n"
        "8 若提供了账户仓位数据，必须体现持仓保护、减仓、对冲或止损调整建议\n"
        "9 若某项数据缺失，必须明确写“该项暂无数据，已降级处理”，禁止臆测"
    )

    ANALYSIS_WORKFLOW = (
        "请先在内部按以下顺序完成分析，但不要在输出开头堆砌过程：\n"
        "第一步：4h / 1h 定方向，判断趋势、震荡还是反弹修正\n"
        "第二步：15m 找执行位，判断是否适合提前挂单而不是追单\n"
        "第三步：判断最近更可能被扫的流动性位置，以及主要支撑阻力是否接近\n"
        "第四步：结合订单簿、挂单墙、撤单速度、假墙风险确认短线压力方向\n"
        "第五步：结合成交流、成交量变化、OI、资金费率、多空比、基差判断拥挤方向\n"
        "第六步：给出主方案、备选方案和明确的不做条件"
    )

    OUTPUT_REQUIREMENTS = (
        "最终输出必须严格按以下结构，顺序不能变，必须先结论后理由：\n"
        "结论：\n"
        "- 方向：做多 / 做空 / 观望\n"
        "- 建议：一句话概括当前最优动作\n"
        "- 最优挂单方案：限价挂单 / 分批挂单 / 区间埋伏单 / 突破挂单 / 暂不部署\n"
        "- 当前是否值得提前部署：是 / 否\n"
        "- 预计等待时间：从现在到挂单触发、或到下一次重新评估的大致时间窗口\n"
        "\n"
        "交易计划：\n"
        "- 当前市场状态：趋势 / 震荡 / 修正 / 诱多诱空中的哪一种\n"
        "- 挂单类型：限价 / 分批 / 区间 / 突破 / 无\n"
        "- 挂单区间/价格：给出明确价格或价格带\n"
        "- 分批方式：最多 1-3 档，写清每档的大致仓位占比\n"
        "- 触发条件：什么条件成立才执行\n"
        "- 若未触发怎么办：继续等待 / 撤单 / 改价 / 重新评估，并给出大致时间窗口\n"
        "- 止损：必须与结构失效位对应，并参考 ATR 建议止损距离验证合理性\n"
        "- 止盈：至少给出 TP1 / TP2 或分批止盈逻辑\n"
        "- 预期盈亏比：若明显偏低，直接建议观望\n"
        "- 仓位建议：轻仓 / 中性 / 偏重，并说明原因\n"
        "- 杠杆建议：给出保守到平衡的杠杆区间\n"
        "- 账户仓位联动：若已有持仓，说明持有/减仓/对冲/保护动作；若空仓或不可用也要明确写出\n"
        "- 失效条件：哪些价格行为或数据变化出现后，这个计划作废\n"
        "\n"
        "理由：\n"
        "- 趋势结构：4h、1h、15m 如何配合\n"
        "- 关键位置：支撑、阻力、区间边界、POC/HVN/LVN\n"
        "- 流动性与爆仓区：市场更可能先扫哪一侧\n"
        "- 盘口与订单簿动态：是否存在明显挂单墙、撤单、假墙风险\n"
        "- 成交流与成交量：主动买卖盘、CVD、大单方向、量能是否支持\n"
        "- 衍生品与拥挤度：资金费率、OI、多空比、基差、跨交易所费率差\n"
        "- 为什么适合挂单而不是追单：必须说明\n"
        "- 风险点：列出最重要的 2-4 条\n"
        "\n"
        "备选方案：\n"
        "- 条件成立时怎么做：当价格先扫另一侧或突破关键位后如何切换\n"
        "- 主方案失效时怎么做：明确撤单、等待还是反手观察\n"
        "\n"
        "不做条件：\n"
        "- 什么情况下这单不做\n"
        "- 重新评估触发条件：未来看到什么信号再回来部署\n"
        "\n"
        "数据降级说明：\n"
        "- 只允许引用下方“数据质量与降级提示”里已经给出的事项，不得自行编造新的缺失项或置信度结论\n"
        "- 区分写清三类：缺失/不可用、样本偏少、模型估算/辅助参考\n"
        "- 若没有明确降级项，就写“无明显降级项”\n"
        "\n"
        "补充要求：\n"
        "1 若结论为观望，仍要按同一结构输出，并将挂单项明确写为“暂不部署”或“无”\n"
        "2 不要输出空泛教学内容，不要写成宏观长报告\n"
        "3 若建议挂单，默认以提前部署为主，而不是等触发后追市价\n"
        "4 预计等待时间只给大致窗口即可，例如 15-30 分钟、1-3 小时、4-12 小时，不要假装精确到分钟"
    )

    def build(self, context: Dict) -> str:
        lines: list[str] = [
            self.SYSTEM_PROMPT,
            "",
            self.ANALYSIS_WORKFLOW,
            "",
            self.OUTPUT_REQUIREMENTS,
            "",
            "请优先判断“市场下一步最可能扫的流动性位置”，再决定是否值得提前挂单。",
            "若信号冲突，请降低激进度，并在结论中直接写观望。",
            "",
        ]

        lines.extend(self._build_account_state_lines(context.get("account_positions", {})))
        lines.append("")
        lines.extend(self._build_data_quality_lines(context))
        lines.append("")

        lines.append("行情数据：")
        lines.append(f"生成时间: {context.get('generated_at')}")
        lines.append(f"交易品种: {context.get('symbol')}")
        lines.append(f"当前价格: {context.get('price')}")
        lines.append("")

        market_structure = context.get("market_structure", {})
        if market_structure:
            lines.append("多周期结构：")
            for tf, trend in market_structure.items():
                lines.append(f"- {tf}: {trend}")
            lines.append("")

        stats_24h = context.get("stats_24h", {})
        if stats_24h:
            lines.append("24h 市场统计：")
            lines.append(
                f"high={stats_24h.get('high_price')} low={stats_24h.get('low_price')} "
                f"last={stats_24h.get('last_price')} change_pct={stats_24h.get('price_change_percent')} "
                f"volume={stats_24h.get('volume')} quote_volume={stats_24h.get('quote_volume')}"
            )
            lines.append("")

        recent_4h_range = context.get("recent_4h_range", {})
        if recent_4h_range:
            lines.append("近 4 小时区间：")
            lines.append(
                f"high={recent_4h_range.get('high')} low={recent_4h_range.get('low')} "
                f"range_abs={recent_4h_range.get('range_abs')} range_pct={recent_4h_range.get('range_pct')}"
            )
            lines.append("")

        session_context = context.get("session_context", {})
        if session_context:
            lines.append("Session / Funding 语境：")
            lines.append(
                f"current_session={session_context.get('current_session')} "
                f"session_high={session_context.get('session_high')} "
                f"session_low={session_context.get('session_low')} "
                f"day_high={session_context.get('day_high')} "
                f"day_low={session_context.get('day_low')} "
                f"funding_countdown={session_context.get('funding_countdown_label')}"
            )
            lines.append("")

        deployment_context = context.get("deployment_context", {})
        if deployment_context:
            lines.append("部署评分与计划层：")
            lines.append(
                f"primary_bias={deployment_context.get('primary_bias')} "
                f"transition_state={deployment_context.get('transition_state')} "
                f"deployment_score={deployment_context.get('deployment_score')} "
                f"deployment_score_value={deployment_context.get('deployment_score_value')} "
                f"distance_to_entry_bps={deployment_context.get('distance_to_entry_bps')}"
            )
            lines.append(f"component_scores={deployment_context.get('component_scores')}")
            lines.append(f"plan_zones={deployment_context.get('plan_zones')}")
            lines.append(f"state_tags={deployment_context.get('state_tags')}")
            lines.append("")

        for tf, metrics in context.get("timeframes", {}).items():
            ema = metrics.get("ema", {})
            macd = metrics.get("macd", {})
            kdj = metrics.get("kdj", {})
            rsi = metrics.get("rsi", {})
            atr = metrics.get("atr", {})
            bollinger = metrics.get("bollinger", {})
            vwap = metrics.get("vwap")
            features = metrics.get("features", {})
            bar_state = metrics.get("bar_state", {})
            lines.append(f"[{tf}] 技术结构：")
            lines.append(f"EMA7={ema.get('7')} EMA25={ema.get('25')} EMA99={ema.get('99')}")
            lines.append(f"MACD DIF={macd.get('dif')} DEA={macd.get('dea')} HIST={macd.get('hist')}")
            lines.append(f"KDJ K={kdj.get('k')} D={kdj.get('d')} J={kdj.get('j')}")
            lines.append(f"RSI14={rsi.get('14')} state={rsi.get('state')} divergence={rsi.get('divergence')}")
            lines.append(
                f"ATR={atr.get('atr')} ATR%={atr.get('atr_pct')} "
                f"suggested_SL_distance={atr.get('suggested_sl_distance')} "
                f"suggested_SL%={atr.get('suggested_sl_pct')}"
            )
            lines.append(
                f"Bollinger upper={bollinger.get('upper')} mid={bollinger.get('middle')} "
                f"lower={bollinger.get('lower')} bandwidth={bollinger.get('bandwidth')} "
                f"%B={bollinger.get('percent_b')}"
            )
            if vwap:
                lines.append(f"VWAP={vwap}")
            lines.append(
                f"features trend={features.get('trend')} momentum={features.get('momentum')} "
                f"kdj_state={features.get('kdj_state')} rsi_state={features.get('rsi_state')} "
                f"rsi_divergence={features.get('rsi_divergence')}"
            )
            lines.append(
                f"bar_closed={bar_state.get('bar_closed')} close_time={bar_state.get('close_time')} "
                f"seconds_to_close={bar_state.get('seconds_to_close')}"
            )
            lines.append("")

        volume_change = context.get("volume_change", {})
        if volume_change:
            lines.append("成交量变化：")
            for tf, values in volume_change.items():
                lines.append(
                    f"- {tf}: last={values.get('last_volume')} prev={values.get('prev_volume')} "
                    f"delta={values.get('delta_volume')} delta_pct={values.get('delta_pct')} "
                    f"avg20={values.get('avg20_volume')} vs_avg20_pct={values.get('vs_avg20_pct')}"
                )
            lines.append("")

        orderbook = context.get("orderbook", {})
        if orderbook:
            lines.append("盘口信息：")
            lines.append(
                f"bid_volume={orderbook.get('bid_volume')} ask_volume={orderbook.get('ask_volume')} "
                f"imbalance={orderbook.get('imbalance')}"
            )
            lines.append(f"bid_wall={orderbook.get('bid_wall')} ask_wall={orderbook.get('ask_wall')}")
            lines.append("")

        orderbook_dynamics = context.get("orderbook_dynamics", {})
        if orderbook_dynamics:
            lines.append("订单簿动态：")
            lines.append(
                f"snapshot_count={orderbook_dynamics.get('snapshot_count')} "
                f"sample_duration_seconds={orderbook_dynamics.get('sample_duration_seconds')} "
                f"avg_bid_change={orderbook_dynamics.get('avg_bid_volume_change')} "
                f"avg_ask_change={orderbook_dynamics.get('avg_ask_volume_change')} "
                f"best_bid_change_per_minute={orderbook_dynamics.get('best_bid_change_per_minute')} "
                f"best_ask_change_per_minute={orderbook_dynamics.get('best_ask_change_per_minute')}"
            )
            lines.append(
                f"bid_add_rate={orderbook_dynamics.get('bid_add_rate_per_second')} "
                f"bid_cancel_rate={orderbook_dynamics.get('bid_cancel_rate_per_second')} "
                f"ask_add_rate={orderbook_dynamics.get('ask_add_rate_per_second')} "
                f"ask_cancel_rate={orderbook_dynamics.get('ask_cancel_rate_per_second')}"
            )
            lines.append(
                f"avg_wall_lifetime_seconds={orderbook_dynamics.get('avg_wall_lifetime_seconds')} "
                f"max_wall_lifetime_seconds={orderbook_dynamics.get('max_wall_lifetime_seconds')} "
                f"wall_behavior={orderbook_dynamics.get('wall_behavior')} "
                f"spoofing_risk={orderbook_dynamics.get('spoofing_risk')}"
            )
            lines.append(
                f"passive_absorption_quote={orderbook_dynamics.get('passive_absorption_quote')} "
                f"aggressive_sweep_quote={orderbook_dynamics.get('aggressive_sweep_quote')} "
                f"pull_without_trade_quote={orderbook_dynamics.get('pull_without_trade_quote')}"
            )
            pw = orderbook_dynamics.get("persistent_walls", [])
            if pw and isinstance(pw, list):
                lines.append(f"persistent_walls({len(pw)}):")
                for w in pw[:5]:
                    if isinstance(w, dict):
                        lines.append(
                            f"  side={w.get('side')} price={w.get('price')} "
                            f"avg_qty={w.get('avg_qty')} lifetime={w.get('lifetime_seconds')}s"
                        )
            else:
                lines.append("persistent_walls=none")
            tba = orderbook_dynamics.get("top_bid_level_activity", [])
            taa = orderbook_dynamics.get("top_ask_level_activity", [])
            if tba and isinstance(tba, list):
                lines.append(f"top_bid_levels({len(tba)}):")
                for lv in tba[:3]:
                    if isinstance(lv, dict):
                        lines.append(f"  price={lv.get('price')} net_change={lv.get('net_change')}")
            if taa and isinstance(taa, list):
                lines.append(f"top_ask_levels({len(taa)}):")
                for lv in taa[:3]:
                    if isinstance(lv, dict):
                        lines.append(f"  price={lv.get('price')} net_change={lv.get('net_change')}")
            lines.append("")

        trade_flow = context.get("trade_flow", {})
        if trade_flow:
            lines.append("成交流：")
            lines.append(
                f"trade_count={trade_flow.get('trade_count')} coverage_seconds={trade_flow.get('coverage_seconds')} "
                f"coverage_minutes={trade_flow.get('coverage_minutes')} trades_per_minute={trade_flow.get('trades_per_minute')} "
                f"buy_quote={trade_flow.get('buy_quote')} "
                f"sell_quote={trade_flow.get('sell_quote')} delta_qty={trade_flow.get('delta_qty')} "
                f"delta_quote={trade_flow.get('delta_quote')} cvd_qty={trade_flow.get('cvd_qty')}"
            )
            lines.append(
                f"large_trade_threshold_quote={trade_flow.get('large_trade_threshold_quote')} "
                f"large_buy_quote={trade_flow.get('large_buy_quote')} "
                f"large_sell_quote={trade_flow.get('large_sell_quote')} "
                f"large_trade_direction={trade_flow.get('large_trade_direction')}"
            )
            lines.extend(PromptGenerator._summarize_trade_flow_windows(trade_flow.get("windows")))
            lines.extend(PromptGenerator._summarize_trade_flow_layers(trade_flow.get("aggressor_layers")))
            clusters = trade_flow.get("large_trade_clusters", [])
            if clusters:
                lines.append(f"large_trade_clusters({len(clusters)}):")
                for c in clusters[:5]:
                    lines.append(
                        f"  price={c.get('price')} side={c.get('dominant_side')} "
                        f"total_quote={c.get('total_quote')} count={c.get('count')}"
                    )
            absorption = trade_flow.get("absorption_zones", [])
            if absorption:
                lines.append(f"absorption_zones({len(absorption)}):")
                for z in absorption[:5]:
                    lines.append(
                        f"  price={z.get('price')} absorbed_quote={z.get('absorbed_quote')} "
                        f"side={z.get('side')}"
                    )
            lines.append("")

        liquidation_heatmap = context.get("liquidation_heatmap", {})
        if liquidation_heatmap:
            lines.append("流动性/爆仓热区：")
            lines.append(
                f"source={liquidation_heatmap.get('source')} confidence={liquidation_heatmap.get('confidence')} "
                f"model_assumptions={liquidation_heatmap.get('model_assumptions')}"
            )
            for zone in liquidation_heatmap.get("zones", []):
                lines.append(
                    f"- {self._zone_name(zone)}: zone_low={zone.get('zone_low')} "
                    f"zone_high={zone.get('zone_high')} pressure={self._zone_pressure(zone)}"
                )
            lines.append("")

        signal = context.get("signal_score", {})
        if signal:
            components = signal.get("components", {})
            lines.append("综合信号评分（0=极度看空, 50=中性, 100=极度看多）：")
            lines.append(
                f"composite={signal.get('composite_score')} "
                f"bias={signal.get('bias')} strength={signal.get('strength')}"
            )
            for name, comp in components.items():
                lines.append(f"  {name}: score={comp.get('score')}")
            lines.append("")

        lines.extend(self._build_derivatives_lines(context))
        lines.extend(self._build_volume_profile_lines(context.get("volume_profile", {})))
        lines.extend(self._build_position_sizing_lines(context.get("position_sizing", {})))
        lines.extend(self._build_options_lines(context.get("options_iv", {})))
        lines.extend(self._build_account_detail_lines(context.get("account_positions", {})))

        chart_files = context.get("chart_files", {})
        if chart_files:
            lines.append("图表文件（如当前使用环境可查看则参考，否则忽略）：")
            for timeframe, file_path in chart_files.items():
                lines.append(f"- {timeframe}: {file_path}")
            lines.append("")

        summary_files = context.get("summary_files", {})
        if summary_files:
            lines.append("摘要文件：")
            for name, file_path in summary_files.items():
                lines.append(f"- {name}: {file_path}")
            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def _build_account_state_lines(account_positions: Dict) -> list[str]:
        available = bool(account_positions.get("available"))
        active_count = int(account_positions.get("active_positions_count") or 0)
        reason = PromptGenerator._humanize_reason(account_positions.get("reason"))
        if available and active_count == 0:
            position_state = "flat_with_data"
        elif available:
            position_state = "has_open_positions"
        else:
            position_state = "unavailable"
        return [
            "账户仓位数据状态：",
            (
                f"available={account_positions.get('available')} "
                f"reason={reason} "
                f"active_positions_count={account_positions.get('active_positions_count')}"
            ),
            f"position_state={position_state}",
        ]

    @staticmethod
    def _build_data_quality_lines(context: Dict) -> list[str]:
        notes: list[str] = []

        options_iv = context.get("options_iv", {})
        if options_iv and not options_iv.get("available", True):
            reason = PromptGenerator._humanize_reason(options_iv.get("reason"))
            notes.append(
                f"- 缺失/不可用: 期权隐波数据不可用（reason={reason}），该项暂无数据，已降级处理。"
            )

        account_positions = context.get("account_positions", {})
        if account_positions and not account_positions.get("available", True):
            reason = PromptGenerator._humanize_reason(account_positions.get("reason"))
            notes.append(
                f"- 缺失/不可用: 账户仓位数据不可用（reason={reason}），仓位联动建议需降级处理。"
            )

        cross_exchange_funding = context.get("cross_exchange_funding", {})
        unavailable_exchanges: list[str] = []
        for exchange, values in cross_exchange_funding.items():
            if not values.get("available", True):
                reason = PromptGenerator._humanize_reason(values.get("reason"))
                unavailable_exchanges.append(f"{exchange.upper()}（{reason}）")
        if unavailable_exchanges:
            joined = "、".join(unavailable_exchanges)
            notes.append(
                f"- 缺失/不可用: 跨交易所资金费率部分不可用，当前缺失 {joined}，跨所比较按剩余可用交易所降级处理。"
            )

        orderbook_dynamics = context.get("orderbook_dynamics", {})
        snapshot_count = int(orderbook_dynamics.get("snapshot_count") or 0)
        if snapshot_count and snapshot_count < 30:
            notes.append(
                f"- 样本偏少: 盘口动态仅有 {snapshot_count} 个 snapshot，低于目标的 30 个以上连续样本，适合作为辅助判断。"
            )

        trade_flow = context.get("trade_flow", {})
        coverage_seconds = float(trade_flow.get("coverage_seconds") or 0.0)
        if trade_flow and coverage_seconds < 900:
            notes.append(
                f"- 样本偏少: AggTrades 实际覆盖约 {round(coverage_seconds, 2)} 秒，15m 级别滚动 CVD 可能不完整。"
            )

        liquidation_heatmap = context.get("liquidation_heatmap", {})
        heatmap_source = liquidation_heatmap.get("source")
        if heatmap_source == "model_estimate":
            notes.append(
                "- 模型估算/辅助参考: 爆仓热区来源为 model_estimate，不是交易所原生清算热图，仅作辅助参考。"
            )

        if not notes:
            notes.append("- 无明显降级项。")

        return ["数据质量与降级提示：", *notes]

    @staticmethod
    def _summarize_oi_periods(periods: Dict) -> list[str]:
        """Summarize OI period data: only trend/latest/delta, skip full series."""
        if not periods or not isinstance(periods, dict):
            return []
        lines: list[str] = []
        for period_key, data in periods.items():
            if not isinstance(data, dict):
                continue
            lines.append(
                f"  {period_key}: trend={data.get('trend')} delta_pct={data.get('delta_pct')} "
                f"vs_avg_pct={data.get('vs_avg_pct')} latest_state={data.get('latest_state')} "
                f"points={len(data.get('series', []))}"
            )
        return lines

    @staticmethod
    def _summarize_ratio_block(block: Dict) -> str:
        """Summarize a long/short ratio block: latest value + trend, skip full series."""
        if not block or not isinstance(block, dict):
            return "N/A"
        latest = block.get("latest_ratio") or block.get("long_short_ratio")
        trend = block.get("trend") or block.get("crowding")
        series = block.get("series", [])
        latest_long = block.get("latest_long") or block.get("long_account")
        latest_short = block.get("latest_short") or block.get("short_account")
        parts = [f"ratio={latest}"]
        if latest_long is not None:
            parts.append(f"long={latest_long}")
        if latest_short is not None:
            parts.append(f"short={latest_short}")
        if trend:
            parts.append(f"trend={trend}")
        if series:
            parts.append(f"points={len(series)}")
        return " ".join(parts)

    @staticmethod
    def _build_derivatives_lines(context: Dict) -> list[str]:
        lines: list[str] = []

        oi = context.get("open_interest")
        oi_trend = context.get("open_interest_trend", {})
        if oi is not None or oi_trend:
            lines.append("未平仓量信息：")
            lines.append(
                f"open_interest={oi} trend={oi_trend.get('trend')} "
                f"delta_pct={oi_trend.get('delta_pct')} vs_avg_pct={oi_trend.get('vs_avg_pct')} "
                f"summary_period={oi_trend.get('summary_period')} "
                f"latest_state={oi_trend.get('latest_state')} "
                f"latest_interpretation={oi_trend.get('latest_interpretation')} "
                f"composite_signal={oi_trend.get('composite_signal')}"
            )
            period_lines = PromptGenerator._summarize_oi_periods(oi_trend.get("periods"))
            if period_lines:
                lines.append("OI 分周期摘要：")
                lines.extend(period_lines)
            voc = oi_trend.get("volume_oi_cvd_state")
            if isinstance(voc, dict):
                lines.append(
                    f"volume_oi_cvd: volume_trend={voc.get('volume_trend')} "
                    f"oi_trend={voc.get('oi_trend')} cvd_trend={voc.get('cvd_trend')} "
                    f"interpretation={voc.get('interpretation')}"
                )
            elif voc is not None:
                lines.append(f"volume_oi_cvd_state={voc}")
            lines.append("")

        long_short_ratio = context.get("long_short_ratio", {})
        if long_short_ratio:
            lines.append("多空持仓比：")
            lines.append(f"overall_crowding={long_short_ratio.get('overall_crowding')}")
            lines.append(f"global_account: {PromptGenerator._summarize_ratio_block(long_short_ratio.get('global_account'))}")
            lines.append(f"top_trader: {PromptGenerator._summarize_ratio_block(long_short_ratio.get('top_trader_position'))}")
            lines.append("")

        funding = context.get("funding", {})
        if funding:
            lines.append("资金费率与标记价格：")
            lines.append(
                f"funding_rate={funding.get('funding_rate')} mark_price={funding.get('mark_price')} "
                f"index_price={funding.get('index_price')}"
            )
            lines.append("")

        basis = context.get("basis", {})
        if basis:
            lines.append("期现基差：")
            lines.append(
                f"basis_abs={basis.get('basis_abs')} basis_bps={basis.get('basis_bps')} "
                f"structure={basis.get('structure')}"
            )
            lines.append("")

        funding_spread = context.get("funding_spread", {})
        if funding_spread:
            lines.append("跨交易所资金费率差：")
            lines.append(
                f"available_count={funding_spread.get('available_count')} "
                f"max_spread_bps={funding_spread.get('max_spread_bps')} "
                f"highest={funding_spread.get('highest_exchange')} "
                f"lowest={funding_spread.get('lowest_exchange')} "
                f"signal={funding_spread.get('signal')}"
            )
            lines.append("")

        cross_exchange_funding = context.get("cross_exchange_funding", {})
        if cross_exchange_funding:
            lines.append("跨交易所资金费率明细：")
            for exchange, values in cross_exchange_funding.items():
                if values.get("available", True):
                    lines.append(
                        f"- {exchange.upper()}: available=True funding_rate={values.get('funding_rate')}"
                    )
                else:
                    lines.append(
                        f"- {exchange.upper()}: available=False reason={PromptGenerator._humanize_reason(values.get('reason'))}"
                    )
            lines.append("")

        return lines

    @staticmethod
    def _build_volume_profile_lines(volume_profile: Dict) -> list[str]:
        if not volume_profile:
            return []
        lines = [
            "成交量分布：",
            (
                f"source_tf={volume_profile.get('source_timeframe')} "
                f"poc={volume_profile.get('poc_price')} "
                f"hvn={volume_profile.get('hvn_prices')} "
                f"lvn={volume_profile.get('lvn_prices')} "
                f"window={volume_profile.get('window_size')} "
                f"bins={volume_profile.get('bins')}"
            ),
        ]
        session_profiles = volume_profile.get("session_profiles", {})
        if session_profiles and isinstance(session_profiles, dict):
            stf = volume_profile.get("session_source_timeframe", "")
            lines.append(f"session_profiles(tf={stf}):")
            for session_name, sp in session_profiles.items():
                if isinstance(sp, dict):
                    lines.append(
                        f"  {session_name}: poc={sp.get('poc_price')} "
                        f"hvn={sp.get('hvn_prices')} lvn={sp.get('lvn_prices')}"
                    )
        anchored_profiles = volume_profile.get("anchored_profiles", [])
        if anchored_profiles and isinstance(anchored_profiles, list):
            atf = volume_profile.get("anchored_source_timeframe", "")
            lines.append(f"anchored_profiles(tf={atf}):")
            for ap in anchored_profiles[:5]:
                if isinstance(ap, dict):
                    lines.append(
                        f"  anchor={ap.get('anchor_label')} vwap={ap.get('vwap')} "
                        f"poc={ap.get('poc_price')}"
                    )
        lines.append("")
        return lines

    @staticmethod
    def _build_options_lines(options_iv: Dict) -> list[str]:
        if not options_iv:
            return []
        if not options_iv.get("available", True):
            reason = PromptGenerator._humanize_reason(options_iv.get("reason"))
            return [
                "期权隐波信息：",
                f"available=False reason={reason}",
                "",
            ]
        return [
            "期权隐波信息：",
            f"{options_iv}",
            "",
        ]

    @staticmethod
    def _build_account_detail_lines(account_positions: Dict) -> list[str]:
        if not account_positions:
            return []
        return [
            "账户仓位明细：",
            f"symbol_position={account_positions.get('symbol_position')}",
            f"active_positions={account_positions.get('active_positions')}",
            "",
        ]

    @staticmethod
    def _build_position_sizing_lines(sizing: Dict) -> list[str]:
        if not sizing or not sizing.get("available"):
            return []
        ref = sizing.get("reference_levels", {})
        long_ref = ref.get("long", {})
        short_ref = ref.get("short", {})
        return [
            "仓位计算参考（默认 1% 风险，10x 杠杆，基于 ATR）：",
            (
                f"ATR周期={sizing.get('atr_timeframe')} ATR={sizing.get('atr')} "
                f"SL距离={sizing.get('sl_distance')} SL%={sizing.get('sl_pct')}"
            ),
            (
                f"建议仓位={sizing.get('position_size_usdt')}U "
                f"({sizing.get('position_size_btc')} BTC) "
                f"保证金={sizing.get('margin_required')}U "
                f"占比={sizing.get('margin_usage_pct')}%"
            ),
            (
                f"做多参考: SL={long_ref.get('stop_loss')} "
                f"TP1={long_ref.get('tp1')}(RR 2:1) TP2={long_ref.get('tp2')}(RR 3:1)"
            ),
            (
                f"做空参考: SL={short_ref.get('stop_loss')} "
                f"TP1={short_ref.get('tp1')}(RR 2:1) TP2={short_ref.get('tp2')}(RR 3:1)"
            ),
            "",
        ]

    @staticmethod
    def _summarize_trade_flow_windows(windows: object) -> list[str]:
        if not windows or not isinstance(windows, dict):
            return []
        lines = ["rolling_windows:"]
        for label, w in windows.items():
            if not isinstance(w, dict):
                continue
            lines.append(
                f"  {label}: buy={w.get('buy_quote')} sell={w.get('sell_quote')} "
                f"delta={w.get('delta_quote')} cvd={w.get('cvd_qty')} "
                f"direction={w.get('large_trade_direction')}"
            )
        return lines

    @staticmethod
    def _summarize_trade_flow_layers(layers: object) -> list[str]:
        if not layers or not isinstance(layers, dict):
            return []
        lines = ["aggressor_layers:"]
        for label, layer in layers.items():
            if not isinstance(layer, dict):
                continue
            lines.append(
                f"  {label}: buy={layer.get('buy_quote')} sell={layer.get('sell_quote')} "
                f"delta={layer.get('delta_quote')} direction={layer.get('large_trade_direction')}"
            )
        return lines

    @staticmethod
    def _humanize_reason(reason: object) -> str:
        if reason is None:
            return "unknown"
        text = str(reason).strip()
        return text or "unknown"

    @staticmethod
    def _zone_name(zone: Dict) -> str:
        name = str(zone.get("name") or zone.get("estimated_pressure") or "zone")
        return name.replace("_", " ")

    @staticmethod
    def _zone_pressure(zone: Dict) -> str:
        return str(zone.get("estimated_pressure") or zone.get("assumption") or "unknown")
