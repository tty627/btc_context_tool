from datetime import datetime
from typing import Dict


class SummaryTableGenerator:
    @staticmethod
    def _fmt_pct(value: float, decimals: int = 2) -> str:
        return f"{value:.{decimals}f}%"

    @staticmethod
    def _fmt_bps(value: float, decimals: int = 2) -> str:
        return f"{value:.{decimals}f} bps"

    @staticmethod
    def _fmt_seconds(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def build(self, context: Dict) -> str:
        funding = context.get("funding", {})
        basis = context.get("basis", {})
        open_interest = context.get("open_interest_trend", {})
        long_short = context.get("long_short_ratio", {})
        orderbook_dynamics = context.get("orderbook_dynamics", {})
        trade_flow = context.get("trade_flow", {})
        deployment = context.get("deployment_context", {})
        session = context.get("session_context", {})
        account = context.get("account_positions", {})

        trade_window = trade_flow.get("windows", {}).get("5m", {})
        next_funding = funding.get("next_funding_time")
        countdown = session.get("funding_countdown_seconds")
        generated_at = context.get("generated_at")

        lines = [
            "# BTC Context Summary",
            "",
            f"Generated at: {generated_at}",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Session | {session.get('current_session', 'unknown')} |",
            f"| Session Range | {session.get('session_low', 0.0)} - {session.get('session_high', 0.0)} |",
            f"| Funding | {self._fmt_pct(float(funding.get('funding_rate', 0.0)) * 100, 4)} |",
            f"| Funding Countdown | {self._fmt_seconds(float(countdown or 0.0))} |",
            f"| Next Funding Time | {next_funding or 'unknown'} |",
            f"| Basis | {self._fmt_bps(float(basis.get('basis_bps', 0.0)))} |",
            f"| OI State | {open_interest.get('latest_state', 'unknown')} / {open_interest.get('latest_interpretation', 'unknown')} |",
            f"| OI+CVD Composite | {open_interest.get('composite_signal', 'unknown')} |",
            f"| Long/Short Crowding | {long_short.get('overall_crowding', 'unknown')} |",
            f"| Spoof Risk | {orderbook_dynamics.get('spoofing_risk', 'unknown')} |",
            f"| Wall Behavior | {orderbook_dynamics.get('wall_behavior', 'unknown')} |",
            f"| Trade Delta 5m | {trade_window.get('delta_quote', 0.0)} |",
            f"| Large Trade 5m | {trade_window.get('large_trade_direction', 'unknown')} |",
            f"| Deployment Score | {deployment.get('deployment_score', 'unknown')} ({deployment.get('deployment_score_value', 0)}) |",
            f"| Deployment Bias | {deployment.get('primary_bias', 'unknown')} |",
            f"| Transition State | {deployment.get('transition_state', 'unknown')} |",
            f"| Account Position State | {account.get('active_positions_count', 0)} active / available={account.get('available')} |",
        ]

        plan = deployment.get("plan_zones", {})
        if plan:
            lines.extend(
                [
                    "",
                    "| Plan Layer | Zone |",
                    "| --- | --- |",
                    f"| Entry | {plan.get('entry', {}).get('zone_low', 'n/a')} - {plan.get('entry', {}).get('zone_high', 'n/a')} |",
                    f"| Invalidation | {plan.get('invalidation', {}).get('zone_low', 'n/a')} - {plan.get('invalidation', {}).get('zone_high', 'n/a')} |",
                    f"| TP | {plan.get('take_profit', {}).get('zone_low', 'n/a')} - {plan.get('take_profit', {}).get('zone_high', 'n/a')} |",
                ]
            )

        state_tags = deployment.get("state_tags", [])
        if state_tags:
            lines.extend(["", f"State Tags: {', '.join(state_tags)}"])

        return "\n".join(lines).strip()
