"""External market-driver data collector.

Collects data beyond the pure on-chart / derivatives layer:
  - Fear & Greed Index  (alternative.me — free, no auth)
  - On-chain basics       (blockchain.info — free, no auth)
  - BTC ETF net flow      (free endpoints where available)
  - Gas / mempool         (mempool.space — free, no auth)

All endpoints are best-effort: failures return ``available: False`` so
downstream consumers can degrade gracefully.
"""

import logging
import time
from typing import Any, Dict

logger = logging.getLogger("btc_context.external")

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

import json
import os
from urllib.request import Request, urlopen, build_opener, ProxyHandler
from urllib.error import HTTPError, URLError


class ExternalDataCollector:
    """Lightweight collector for external / macro / on-chain drivers."""

    _TIMEOUT = 10

    def __init__(self, proxy: str | None = None):
        self._proxy = proxy or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
        if _HAS_HTTPX:
            kw: dict = {"timeout": self._TIMEOUT, "follow_redirects": True}
            if self._proxy:
                kw["proxy"] = self._proxy
                kw["trust_env"] = False
            self._client = httpx.Client(**kw)
        else:
            self._client = None

    # ── low-level GET ────────────────────────────────────────────────────

    def _get_json(self, url: str) -> Any:
        if self._client is not None:
            resp = self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        req = Request(url, headers={"User-Agent": "btc_context_tool/1.0"})
        if self._proxy:
            opener = build_opener(ProxyHandler({"http": self._proxy, "https": self._proxy}))
            raw = opener.open(req, timeout=self._TIMEOUT).read()
        else:
            raw = urlopen(req, timeout=self._TIMEOUT).read()
        return json.loads(raw)

    # ── Fear & Greed Index ───────────────────────────────────────────────

    def get_fear_greed_index(self) -> Dict:
        """Current Fear & Greed index from alternative.me."""
        url = "https://api.alternative.me/fng/?limit=1&format=json"
        try:
            payload = self._get_json(url)
            data = payload.get("data", [])
            if data:
                entry = data[0]
                return {
                    "available": True,
                    "value": int(entry.get("value", 0)),
                    "classification": entry.get("value_classification", ""),
                    "timestamp": int(entry.get("timestamp", 0)),
                }
        except Exception as exc:
            logger.debug("fear_greed fetch failed: %s", exc)
            return {"available": False, "reason": str(exc)}
        return {"available": False, "reason": "empty_response"}

    # ── On-chain basics (blockchain.info) ────────────────────────────────

    def get_onchain_basics(self) -> Dict:
        """Hash rate, difficulty, unconfirmed tx count, avg block time."""
        result: Dict = {"available": False}
        try:
            stats = self._get_json("https://api.blockchain.info/stats")
            result = {
                "available": True,
                "hash_rate_th": round(float(stats.get("hash_rate", 0)) / 1e6, 2),
                "difficulty": float(stats.get("difficulty", 0)),
                "n_tx_unconfirmed": int(stats.get("n_tx", 0)),
                "minutes_between_blocks": round(float(stats.get("minutes_between_blocks", 0)), 2),
                "market_price_usd": float(stats.get("market_price_usd", 0)),
                "total_btc_sent_24h": round(float(stats.get("total_btc_sent", 0)) / 1e8, 2),
                "n_blocks_mined_24h": int(stats.get("n_blocks_mined", 0)),
            }
        except Exception as exc:
            logger.debug("onchain_basics fetch failed: %s", exc)
            result = {"available": False, "reason": str(exc)}
        return result

    # ── Mempool state (mempool.space) ────────────────────────────────────

    def get_mempool_state(self) -> Dict:
        """Current mempool statistics from mempool.space."""
        url = "https://mempool.space/api/mempool"
        try:
            data = self._get_json(url)
            return {
                "available": True,
                "count": int(data.get("count", 0)),
                "vsize_bytes": int(data.get("vsize", 0)),
                "total_fee_btc": round(float(data.get("total_fee", 0)) / 1e8, 6),
            }
        except Exception as exc:
            logger.debug("mempool fetch failed: %s", exc)
            return {"available": False, "reason": str(exc)}

    # ── Recommended fees (mempool.space) ─────────────────────────────────

    def get_recommended_fees(self) -> Dict:
        """Recommended fee rates from mempool.space (sat/vB)."""
        url = "https://mempool.space/api/v1/fees/recommended"
        try:
            data = self._get_json(url)
            return {
                "available": True,
                "fastest_fee": int(data.get("fastestFee", 0)),
                "half_hour_fee": int(data.get("halfHourFee", 0)),
                "hour_fee": int(data.get("hourFee", 0)),
                "economy_fee": int(data.get("economyFee", 0)),
                "minimum_fee": int(data.get("minimumFee", 0)),
            }
        except Exception as exc:
            logger.debug("fees fetch failed: %s", exc)
            return {"available": False, "reason": str(exc)}

    # ── Difficulty adjustment (mempool.space) ────────────────────────────

    def get_difficulty_adjustment(self) -> Dict:
        """Current difficulty epoch progress from mempool.space."""
        url = "https://mempool.space/api/v1/difficulty-adjustment"
        try:
            data = self._get_json(url)
            return {
                "available": True,
                "progress_pct": round(float(data.get("progressPercent", 0)), 2),
                "difficulty_change_pct": round(float(data.get("difficultyChange", 0)), 2),
                "remaining_blocks": int(data.get("remainingBlocks", 0)),
                "remaining_time_ms": int(data.get("remainingTime", 0)),
                "estimated_retarget_date": int(data.get("estimatedRetargetDate", 0)),
            }
        except Exception as exc:
            logger.debug("difficulty_adjustment fetch failed: %s", exc)
            return {"available": False, "reason": str(exc)}

    # ── BTC ETF net flow (best-effort, free tier) ────────────────────────

    def get_btc_etf_flow(self) -> Dict:
        """Try to fetch BTC spot ETF daily net flow.

        Uses coinglass public endpoint; may require API key for full data.
        Returns ``available: False`` if the endpoint is gated or fails.
        """
        url = "https://api.coinglass.com/api/etf/bitcoin/flow-total"
        try:
            data = self._get_json(url)
            if isinstance(data, dict) and data.get("code") == "0":
                rows = data.get("data", [])
                if rows:
                    latest = rows[-1] if isinstance(rows, list) else rows
                    return {
                        "available": True,
                        "date": latest.get("date", ""),
                        "total_net_flow_usd": float(latest.get("totalNetFlow", 0)),
                        "source": "coinglass_free",
                    }
            return {"available": False, "reason": "gated_or_empty"}
        except Exception as exc:
            logger.debug("etf_flow fetch failed: %s", exc)
            return {"available": False, "reason": str(exc)}

    # ── aggregate convenience method ─────────────────────────────────────

    def collect_all(self) -> Dict:
        """Run all external data fetches and return a combined dict."""
        return {
            "fear_greed": self.get_fear_greed_index(),
            "onchain": self.get_onchain_basics(),
            "mempool": self.get_mempool_state(),
            "fees": self.get_recommended_fees(),
            "difficulty_adjustment": self.get_difficulty_adjustment(),
            "etf_flow": self.get_btc_etf_flow(),
        }
