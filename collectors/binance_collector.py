import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    from ..config import BASE_URL, REQUEST_TIMEOUT
except ImportError:
    from config import BASE_URL, REQUEST_TIMEOUT

logger = logging.getLogger("btc_context.collector")


class _TTLCache:
    """Simple in-memory cache with per-key TTL (seconds)."""

    def __init__(self, default_ttl: float = 30.0) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        self._store[key] = (time.time() + (ttl if ttl is not None else self.default_ttl), value)

    def clear(self) -> None:
        self._store.clear()


class BinanceFuturesCollector:
    """Collect market data from Binance USDT-M Futures REST APIs.

    Uses httpx with connection pooling when available; falls back to urllib.
    Includes optional TTL cache to avoid redundant requests in watch mode.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: int = REQUEST_TIMEOUT,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        cache_ttl: float = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = self._sanitize_credential(api_key)
        self.api_secret = self._sanitize_credential(api_secret)
        self._cache = _TTLCache(default_ttl=cache_ttl) if cache_ttl > 0 else None
        self._httpx_client: Optional[Any] = None
        if _HAS_HTTPX:
            self._httpx_client = httpx.Client(
                timeout=httpx.Timeout(timeout, connect=5.0),
                follow_redirects=True,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )

    @staticmethod
    def _sanitize_credential(value: Optional[str]) -> str:
        if value is None:
            return ""
        cleaned = value.strip()
        # tolerate credentials copied with surrounding quotes in env files
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
            cleaned = cleaned[1:-1].strip()
        return cleaned

    @staticmethod
    def _format_binance_error(body: str) -> str:
        try:
            payload = json.loads(body)
        except Exception:
            return body

        if isinstance(payload, dict):
            code = payload.get("code")
            message = payload.get("msg")
            if code is not None or message is not None:
                return f"code={code} msg={message}"
        return body

    def _get_json(self, path: str, params: Dict[str, str]) -> Any:
        """Fetch JSON from a Binance REST endpoint with retry logic and optional caching."""
        cache_key = f"{path}|{urlencode(sorted(params.items()))}" if self._cache else ""
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("cache hit: %s", path)
                return cached
        url = f"{self.base_url}{path}"
        result = self._request_with_retry(url, params=params, label=path)
        if self._cache:
            self._cache.set(cache_key, result)
        return result

    def _get_json_url(self, url: str) -> Any:
        """Fetch JSON from an absolute URL with retry logic and optional caching."""
        if self._cache:
            cached = self._cache.get(url)
            if cached is not None:
                logger.debug("cache hit: %s", url)
                return cached
        result = self._request_with_retry(url, params=None, label=url)
        if self._cache:
            self._cache.set(url, result)
        return result

    def _signed_get_json(self, path: str, params: Dict[str, str]) -> Any:
        """Fetch signed JSON from a private Binance endpoint."""
        if not self.api_key or not self.api_secret:
            raise RuntimeError("missing_api_credentials")

        signed_params = dict(params)
        signed_params["timestamp"] = str(int(time.time() * 1000))
        signed_params.setdefault("recvWindow", "5000")
        query = urlencode(signed_params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed_params["signature"] = signature
        url = f"{self.base_url}{path}"
        headers = {"X-MBX-APIKEY": self.api_key}
        return self._request_with_retry(url, params=signed_params, headers=headers, label=f"private:{path}")

    def _request_with_retry(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        label: str = "",
        max_attempts: int = 3,
    ) -> Any:
        delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                if self._httpx_client is not None:
                    return self._httpx_get(url, params, headers)
                return self._urllib_get(url, params, headers)
            except RuntimeError:
                raise
            except Exception as exc:
                if attempt == max_attempts:
                    raise RuntimeError(f"connection error for {label}: {exc}") from exc
                logger.debug("retry %d/%d for %s: %s", attempt, max_attempts, label, exc)
                time.sleep(delay)
                delay *= 2

    def _httpx_get(
        self,
        url: str,
        params: Optional[Dict[str, str]],
        headers: Optional[Dict[str, str]],
    ) -> Any:
        resp = self._httpx_client.get(url, params=params, headers=headers)
        if resp.status_code >= 400:
            details = self._format_binance_error(resp.text)
            raise RuntimeError(f"HTTP {resp.status_code} for {url}: {details}")
        return resp.json()

    def _urllib_get(
        self,
        url: str,
        params: Optional[Dict[str, str]],
        headers: Optional[Dict[str, str]],
    ) -> Any:
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(url, headers=headers or {})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = str(exc.reason)
            details = self._format_binance_error(body)
            raise RuntimeError(f"HTTP {exc.code} for {url}: {details}") from exc

    @staticmethod
    def _with_account_hint(reason: str) -> str:
        lowered = reason.lower()
        if "missing_api_credentials" in lowered:
            return f"{reason} | hint=set BINANCE_API_KEY/BINANCE_API_SECRET"
        if "code=-2014" in lowered:
            return f"{reason} | hint=api key format invalid; check value, env quoting, and mainnet/testnet mismatch"
        if "code=-2015" in lowered:
            return f"{reason} | hint=invalid api-key/ip/permissions; enable Futures read permission or check IP whitelist"
        if "code=-1022" in lowered:
            return f"{reason} | hint=signature invalid; check API secret"
        if "code=-1021" in lowered:
            return f"{reason} | hint=timestamp out of sync; sync local clock"
        return reason

    def _get_position_risk_payload(self, params: Dict[str, str]) -> Tuple[List[Dict], str]:
        # Prefer v3 and gracefully fallback to v2 for older accounts/environments.
        endpoints = ("/fapi/v3/positionRisk", "/fapi/v2/positionRisk")
        errors: List[str] = []

        for endpoint in endpoints:
            try:
                payload = self._signed_get_json(endpoint, params)
                rows = payload if isinstance(payload, list) else []
                return rows, endpoint
            except RuntimeError as exc:
                message = str(exc)
                errors.append(f"{endpoint}: {message}")
                lowered = message.lower()
                # auth/configuration errors won't be fixed by trying another versioned endpoint
                if (
                    "missing_api_credentials" in lowered
                    or "http 401" in lowered
                    or "code=-2014" in lowered
                    or "code=-2015" in lowered
                    or "code=-1022" in lowered
                ):
                    break
                continue

        raise RuntimeError(" ; ".join(errors))

    def has_private_api(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def get_klines(self, symbol: str, interval: str, limit: int) -> List[Dict]:
        payload = self._get_json(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": str(limit)},
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected klines payload for {symbol} {interval}: {payload}")
        candles = []
        for row in payload:
            candles.append(
                {
                    "open_time": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": int(row[6]),
                }
            )
        return candles

    def get_multi_klines(self, symbol: str, intervals: Sequence[str], limit: int) -> Dict[str, List[Dict]]:
        return {interval: self.get_klines(symbol, interval, limit) for interval in intervals}

    def get_orderbook(self, symbol: str, limit: int) -> Dict:
        payload = self._get_json(
            "/fapi/v1/depth",
            {"symbol": symbol, "limit": str(limit)},
        )
        event_time_ms = int(time.time() * 1000)
        bids = [{"price": float(price), "qty": float(qty)} for price, qty in payload["bids"]]
        asks = [{"price": float(price), "qty": float(qty)} for price, qty in payload["asks"]]
        return {
            "bids": bids,
            "asks": asks,
            "update_id": int(payload["lastUpdateId"]),
            "event_time_ms": event_time_ms,
            "event_time": datetime.fromtimestamp(event_time_ms / 1000, tz=timezone.utc).isoformat(),
        }

    def get_open_interest(self, symbol: str) -> float:
        payload = self._get_json("/fapi/v1/openInterest", {"symbol": symbol})
        return float(payload["openInterest"])

    def get_open_interest_hist(self, symbol: str, period: str = "5m", limit: int = 30) -> List[Dict]:
        try:
            payload = self._get_json(
                "/futures/data/openInterestHist",
                {"symbol": symbol, "period": period, "limit": str(limit)},
            )
        except RuntimeError:
            return []

        if not isinstance(payload, list):
            return []

        rows: List[Dict] = []
        for row in payload:
            rows.append(
                {
                    "sum_open_interest": float(row.get("sumOpenInterest") or 0.0),
                    "sum_open_interest_value": float(row.get("sumOpenInterestValue") or 0.0),
                    "timestamp": int(row.get("timestamp") or 0),
                }
            )
        return rows

    def get_global_long_short_ratio(self, symbol: str, period: str = "5m", limit: int = 30) -> List[Dict]:
        try:
            payload = self._get_json(
                "/futures/data/globalLongShortAccountRatio",
                {"symbol": symbol, "period": period, "limit": str(limit)},
            )
        except RuntimeError:
            return []

        if not isinstance(payload, list):
            return []

        rows: List[Dict] = []
        for row in payload:
            rows.append(
                {
                    "long_short_ratio": float(row.get("longShortRatio") or 0.0),
                    "long_account": float(row.get("longAccount") or 0.0),
                    "short_account": float(row.get("shortAccount") or 0.0),
                    "timestamp": int(row.get("timestamp") or 0),
                }
            )
        return rows

    def get_top_trader_long_short_ratio(self, symbol: str, period: str = "5m", limit: int = 30) -> List[Dict]:
        try:
            payload = self._get_json(
                "/futures/data/topLongShortPositionRatio",
                {"symbol": symbol, "period": period, "limit": str(limit)},
            )
        except RuntimeError:
            return []

        if not isinstance(payload, list):
            return []

        rows: List[Dict] = []
        for row in payload:
            rows.append(
                {
                    "long_short_ratio": float(row.get("longShortRatio") or 0.0),
                    "long_account": float(row.get("longAccount") or 0.0),
                    "short_account": float(row.get("shortAccount") or 0.0),
                    "timestamp": int(row.get("timestamp") or 0),
                }
            )
        return rows

    def get_funding(self, symbol: str) -> Dict:
        payload = self._get_json("/fapi/v1/premiumIndex", {"symbol": symbol})
        next_funding_ms = int(payload["nextFundingTime"])
        next_funding_time = datetime.fromtimestamp(next_funding_ms / 1000, tz=timezone.utc).isoformat()
        return {
            "funding_rate": float(payload["lastFundingRate"]),
            "mark_price": float(payload["markPrice"]),
            "index_price": float(payload["indexPrice"]),
            "next_funding_time": next_funding_time,
        }

    def get_ticker_24h(self, symbol: str) -> Dict:
        payload = self._get_json("/fapi/v1/ticker/24hr", {"symbol": symbol})
        return {
            "high_price": float(payload["highPrice"]),
            "low_price": float(payload["lowPrice"]),
            "last_price": float(payload["lastPrice"]),
            "volume": float(payload["volume"]),
            "quote_volume": float(payload["quoteVolume"]),
            "price_change_percent": float(payload["priceChangePercent"]),
            "open_time": int(payload["openTime"]),
            "close_time": int(payload["closeTime"]),
        }

    def get_orderbook_snapshots(
        self,
        symbol: str,
        limit: int,
        samples: int = 3,
        interval_seconds: float = 0.6,
    ) -> List[Dict]:
        count = max(1, samples)
        snapshots: List[Dict] = []
        for idx in range(count):
            try:
                snapshots.append(self.get_orderbook(symbol, limit))
            except RuntimeError:
                break
            if idx < count - 1 and interval_seconds > 0:
                time.sleep(interval_seconds)
        return snapshots

    def get_agg_trades(self, symbol: str, limit: int) -> List[Dict]:
        batch_size = min(max(1, limit), 1000)
        payload = self._get_json("/fapi/v1/aggTrades", {"symbol": symbol, "limit": str(batch_size)})
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected aggTrades payload for {symbol}: {payload}")

        rows = list(payload)
        earliest_id = int(rows[0]["a"]) if rows else 0
        while len(rows) < limit and earliest_id > 0:
            fetch_from = max(0, earliest_id - batch_size)
            older_payload = self._get_json(
                "/fapi/v1/aggTrades",
                {
                    "symbol": symbol,
                    "limit": str(batch_size),
                    "fromId": str(fetch_from),
                },
            )
            if not isinstance(older_payload, list) or not older_payload:
                break

            older_rows = [row for row in older_payload if int(row["a"]) < earliest_id]
            if not older_rows:
                break

            rows = older_rows + rows
            earliest_id = int(older_rows[0]["a"])
            if earliest_id == 0:
                break

        trades: List[Dict] = []
        for row in rows[-limit:]:
            price = float(row["p"])
            qty = float(row["q"])
            is_buyer_maker = bool(row["m"])
            trades.append(
                {
                    "id": int(row["a"]),
                    "price": price,
                    "qty": qty,
                    "quote_qty": price * qty,
                    "is_buyer_maker": is_buyer_maker,
                    # buyer maker=true means aggressive sell, false means aggressive buy
                    "aggressor_side": "sell" if is_buyer_maker else "buy",
                    "timestamp": int(row["T"]),
                }
            )

        trades.sort(key=lambda row: row["timestamp"])
        return trades

    def get_force_orders(self, symbol: str, limit: int = 100) -> List[Dict]:
        try:
            payload = self._get_json("/fapi/v1/allForceOrders", {"symbol": symbol, "limit": str(limit)})
        except RuntimeError:
            # Some environments/accounts may not have this endpoint available.
            return []
        if not isinstance(payload, list):
            return []

        force_orders: List[Dict] = []
        for row in payload:
            price = float(row.get("ap") or row.get("p") or 0.0)
            qty = float(row.get("q") or row.get("origQty") or row.get("z") or 0.0)
            side = str(row.get("S") or row.get("side") or "").lower()
            force_orders.append(
                {
                    "price": price,
                    "qty": qty,
                    "quote_qty": price * qty,
                    "side": side if side in ("buy", "sell") else "unknown",
                    "time": int(row.get("T") or row.get("time") or 0),
                }
            )
        return force_orders

    def get_cross_exchange_funding(
        self,
        symbol: str,
        binance_funding_rate: Optional[float] = None,
    ) -> Dict:
        """Collect funding rates from Binance/Bybit/OKX for spread comparison."""
        data: Dict[str, Dict] = {
            "binance": {
                "available": binance_funding_rate is not None,
                "funding_rate": float(binance_funding_rate or 0.0),
            },
            "bybit": {"available": False, "funding_rate": 0.0, "reason": "unavailable"},
            "okx": {"available": False, "funding_rate": 0.0, "reason": "unavailable"},
        }

        bybit_url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        try:
            payload = self._get_json_url(bybit_url)
            rows = payload.get("result", {}).get("list", [])
            if rows:
                funding_rate = float(rows[0].get("fundingRate") or 0.0)
                data["bybit"] = {"available": True, "funding_rate": funding_rate}
        except Exception as exc:
            data["bybit"] = {"available": False, "funding_rate": 0.0, "reason": str(exc)}

        if symbol.endswith("USDT"):
            base = symbol[:-4]
            okx_inst_id = f"{base}-USDT-SWAP"
        else:
            okx_inst_id = symbol
        okx_url = f"https://www.okx.com/api/v5/public/funding-rate?instId={okx_inst_id}"
        try:
            payload = self._get_json_url(okx_url)
            rows = payload.get("data", [])
            if rows:
                funding_rate = float(rows[0].get("fundingRate") or 0.0)
                data["okx"] = {"available": True, "funding_rate": funding_rate}
        except Exception as exc:
            data["okx"] = {"available": False, "funding_rate": 0.0, "reason": str(exc)}

        return data

    def _normalize_position(self, row: Dict) -> Dict:
        position_amt = float(row.get("positionAmt") or 0.0)
        entry_price = float(row.get("entryPrice") or 0.0)
        mark_price = float(row.get("markPrice") or 0.0)
        liquidation_price = float(row.get("liquidationPrice") or 0.0)
        notional = float(row.get("notional") or 0.0)
        unrealized_pnl = float(row.get("unRealizedProfit") or row.get("unrealizedProfit") or 0.0)
        leverage = int(float(row.get("leverage") or 0))
        isolated_margin = float(row.get("isolatedMargin") or 0.0)
        update_time = int(row.get("updateTime") or 0)

        if position_amt > 0:
            side = "long"
        elif position_amt < 0:
            side = "short"
        else:
            side = "flat"

        return {
            "symbol": str(row.get("symbol") or ""),
            "side": side,
            "position_amt": round(position_amt, 6),
            "entry_price": round(entry_price, 6),
            "mark_price": round(mark_price, 6),
            "liquidation_price": round(liquidation_price, 6),
            "leverage": leverage,
            "margin_type": str(row.get("marginType") or ""),
            "isolated_margin": round(isolated_margin, 6),
            "notional": round(notional, 6),
            "unrealized_pnl": round(unrealized_pnl, 6),
            "position_initial_margin": round(float(row.get("positionInitialMargin") or 0.0), 6),
            "open_order_initial_margin": round(float(row.get("openOrderInitialMargin") or 0.0), 6),
            "updated_time_ms": update_time,
        }

    def get_account_positions(self, symbol: Optional[str] = None) -> Dict:
        """Get current futures positions from private endpoint."""
        if not self.has_private_api():
            return {
                "available": False,
                "reason": self._with_account_hint("missing_api_credentials"),
                "active_positions_count": 0,
                "active_positions": [],
                "symbol_position": None,
            }

        params: Dict[str, str] = {}
        if symbol:
            params["symbol"] = symbol

        try:
            rows, endpoint = self._get_position_risk_payload(params)
        except RuntimeError as exc:
            return {
                "available": False,
                "reason": self._with_account_hint(str(exc)),
                "active_positions_count": 0,
                "active_positions": [],
                "symbol_position": None,
            }

        normalized = [self._normalize_position(row) for row in rows]
        active_positions = [row for row in normalized if abs(row["position_amt"]) > 0]

        symbol_position = None
        if symbol:
            symbol_upper = symbol.upper()
            for row in normalized:
                if row["symbol"].upper() == symbol_upper:
                    symbol_position = row
                    break

        return {
            "available": True,
            "source": f"binance_private_readonly:{endpoint}",
            "active_positions_count": len(active_positions),
            "active_positions": active_positions,
            "symbol_position": symbol_position,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
