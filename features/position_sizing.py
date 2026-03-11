"""Position sizing calculator based on ATR and risk parameters."""

from typing import Dict, Optional

from ._base import FeatureBase


class PositionSizingMixin(FeatureBase):
    @staticmethod
    def calculate_position_size(
        account_balance: float,
        current_price: float,
        atr: Dict,
        risk_pct: float = 1.0,
        leverage: int = 10,
        sl_atr_multiplier: float = 1.5,
    ) -> Dict:
        """Calculate recommended position size based on ATR-derived stop-loss.

        Args:
            account_balance: available margin in USDT
            current_price: current BTC price
            atr: ATR dict from IndicatorEngine (must have 'atr' key)
            risk_pct: max % of account to risk per trade (default 1%)
            leverage: desired leverage
            sl_atr_multiplier: ATR multiples for stop-loss distance
        """
        atr_value = float(atr.get("atr", 0.0))
        if atr_value <= 0 or current_price <= 0 or account_balance <= 0:
            return {
                "available": False,
                "reason": "insufficient_data",
                "account_balance": round(account_balance, 2),
            }

        risk_amount = account_balance * (risk_pct / 100)
        sl_distance = atr_value * sl_atr_multiplier
        sl_pct = sl_distance / current_price * 100

        position_size_usdt = risk_amount / (sl_pct / 100) if sl_pct > 0 else 0.0
        position_size_btc = position_size_usdt / current_price if current_price > 0 else 0.0
        margin_required = position_size_usdt / leverage if leverage > 0 else position_size_usdt
        margin_usage_pct = margin_required / account_balance * 100 if account_balance > 0 else 0.0

        if margin_required > account_balance:
            position_size_usdt = account_balance * leverage
            position_size_btc = position_size_usdt / current_price
            margin_required = account_balance
            margin_usage_pct = 100.0

        sl_long = current_price - sl_distance
        sl_short = current_price + sl_distance
        tp1_long = current_price + sl_distance * 2
        tp2_long = current_price + sl_distance * 3
        tp1_short = current_price - sl_distance * 2
        tp2_short = current_price - sl_distance * 3

        return {
            "available": True,
            "account_balance": round(account_balance, 2),
            "risk_pct": round(risk_pct, 2),
            "risk_amount": round(risk_amount, 2),
            "leverage": leverage,
            "atr": round(atr_value, 2),
            "sl_atr_multiplier": sl_atr_multiplier,
            "sl_distance": round(sl_distance, 2),
            "sl_pct": round(sl_pct, 4),
            "position_size_usdt": round(position_size_usdt, 2),
            "position_size_btc": round(position_size_btc, 6),
            "margin_required": round(margin_required, 2),
            "margin_usage_pct": round(margin_usage_pct, 2),
            "reference_levels": {
                "long": {
                    "stop_loss": round(sl_long, 2),
                    "tp1": round(tp1_long, 2),
                    "tp2": round(tp2_long, 2),
                    "risk_reward_tp1": 2.0,
                    "risk_reward_tp2": 3.0,
                },
                "short": {
                    "stop_loss": round(sl_short, 2),
                    "tp1": round(tp1_short, 2),
                    "tp2": round(tp2_short, 2),
                    "risk_reward_tp1": 2.0,
                    "risk_reward_tp2": 3.0,
                },
            },
        }

    @classmethod
    def extract_position_sizing(
        cls,
        current_price: float,
        indicators_by_timeframe: Dict,
        account_positions: Optional[Dict] = None,
        risk_pct: float = 2.0,
        leverage: int = 25,
    ) -> Dict:
        """Build position sizing data from the best available ATR timeframe.

        Prefers 1h ATR for 4h swing trading style.
        """
        for tf in ("1h", "4h", "15m"):
            atr = indicators_by_timeframe.get(tf, {}).get("atr", {})
            if atr.get("atr", 0.0) > 0:
                sizing = cls.calculate_position_size(
                    account_balance=10000.0,
                    current_price=current_price,
                    atr=atr,
                    risk_pct=risk_pct,
                    leverage=leverage,
                )
                sizing["atr_timeframe"] = tf

                if account_positions and account_positions.get("available"):
                    sym_pos = account_positions.get("symbol_position", {})
                    if sym_pos and abs(float(sym_pos.get("position_amt", 0))) > 0:
                        sizing["has_open_position"] = True
                        sizing["current_side"] = sym_pos.get("side", "flat")
                        sizing["current_notional"] = abs(float(sym_pos.get("notional", 0)))
                    else:
                        sizing["has_open_position"] = False
                else:
                    sizing["has_open_position"] = False

                return sizing

        return {"available": False, "reason": "no_atr_data"}
