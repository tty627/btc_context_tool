"""Facade class that composes all feature extraction mixins.

The original monolithic FeatureExtractor has been split into domain-specific
mixin classes for maintainability.  This module re-exports the composed class
so existing callers (main.py, etc.) continue to work unchanged.
"""

from .deployment import DeploymentMixin
from .derivatives import DerivativesMixin
from .liquidation import LiquidationMixin
from .orderbook import OrderbookMixin
from .position_sizing import PositionSizingMixin
from .session import SessionMixin
from .signal_score import SignalScoreMixin
from .spot_perp import SpotPerpMixin
from .technical import TechnicalMixin
from .trade_flow import TradeFlowMixin
from .transition import TransitionMixin
from .volume import VolumeMixin


class FeatureExtractor(
    TechnicalMixin,
    OrderbookMixin,
    VolumeMixin,
    DerivativesMixin,
    TradeFlowMixin,
    TransitionMixin,
    LiquidationMixin,
    SessionMixin,
    DeploymentMixin,
    PositionSizingMixin,
    SignalScoreMixin,
    SpotPerpMixin,
):
    """All feature-extraction capabilities composed via mixins.

    Public API is identical to the previous monolithic class.
    """

    pass
