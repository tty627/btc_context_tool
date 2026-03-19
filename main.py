import argparse
import concurrent.futures
import copy
from datetime import datetime, timezone
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger("btc_context")


def _load_env_local() -> None:
    """从项目根目录 .env.local 加载 export KEY=\"value\" 行到环境变量（文件已在 .gitignore）。"""
    p = Path(__file__).resolve().parent / ".env.local"
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ[key] = val


try:
    from .collectors import BinanceFuturesCollector, ExternalDataCollector
    from .config import (
        AGG_TRADES_LIMIT, AGG_TRADES_WINDOW_MINUTES,
        AI_ANALYSIS_FILE, BINANCE_API_KEY_ENV, BINANCE_API_SECRET_ENV,
        CHART_BARS, CHART_DIR, CONTEXT_FILE, DEPTH_LIMIT, KLINE_LIMIT, LARGE_TRADE_QUANTILE,
        DEEPSEEK_API_KEY_ENV, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
        LONG_SHORT_LIMIT, LONG_SHORT_PERIOD, OI_LIMIT, OI_PERIOD, OPENAI_API_KEY_ENV,
        OPENAI_MODEL, ORDERBOOK_DYNAMIC_INTERVAL, ORDERBOOK_DYNAMIC_SAMPLES, REPORT_FILE,
        REPORT_MODE, SUMMARY_FILE, SYMBOL, SYSTEM_PROMPT_FILE,
        AI_DECISION_FILE, AI_RESEARCH_FILE, DECISION_PROMPT_FILE, DECISION_SYSTEM_FILE,
        RESEARCH_HANDOFF_FILE, RESEARCH_PROMPT_FILE, RESEARCH_SYSTEM_FILE,
        PUSHPLUS_TOKEN_ENV, TELEGRAM_BOT_TOKEN_ENV, TELEGRAM_CHAT_ID_ENV,
        TIMEFRAMES, VOLUME_PROFILE_BINS, VOLUME_PROFILE_WINDOW,
    )
    from .context import MarketContextBuilder
    from .features import FeatureExtractor
    from .indicators import IndicatorEngine
    from .reports import PromptGenerator, SummaryTableGenerator
except ImportError:
    from collectors import BinanceFuturesCollector, ExternalDataCollector
    from config import (
        AGG_TRADES_LIMIT, AGG_TRADES_WINDOW_MINUTES,
        AI_ANALYSIS_FILE, BINANCE_API_KEY_ENV, BINANCE_API_SECRET_ENV,
        CHART_BARS, CHART_DIR, CONTEXT_FILE, DEPTH_LIMIT, KLINE_LIMIT, LARGE_TRADE_QUANTILE,
        DEEPSEEK_API_KEY_ENV, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
        LONG_SHORT_LIMIT, LONG_SHORT_PERIOD, OI_LIMIT, OI_PERIOD, OPENAI_API_KEY_ENV,
        OPENAI_MODEL, ORDERBOOK_DYNAMIC_INTERVAL, ORDERBOOK_DYNAMIC_SAMPLES, REPORT_FILE,
        REPORT_MODE, SUMMARY_FILE, SYMBOL, SYSTEM_PROMPT_FILE,
        AI_DECISION_FILE, AI_RESEARCH_FILE, DECISION_PROMPT_FILE, DECISION_SYSTEM_FILE,
        RESEARCH_HANDOFF_FILE, RESEARCH_PROMPT_FILE, RESEARCH_SYSTEM_FILE,
        PUSHPLUS_TOKEN_ENV, TELEGRAM_BOT_TOKEN_ENV, TELEGRAM_CHAT_ID_ENV,
        TIMEFRAMES, VOLUME_PROFILE_BINS, VOLUME_PROFILE_WINDOW,
    )
    from context import MarketContextBuilder
    from features import FeatureExtractor
    from indicators import IndicatorEngine
    from reports import PromptGenerator, SummaryTableGenerator


def _load_chart_generator():
    try:
        try:
            from .charts import KlineChartGenerator
        except ImportError:
            from charts import KlineChartGenerator
        return KlineChartGenerator, None
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            return (
                None,
                "matplotlib is not installed; charts are disabled. "
                "Install with: python3 -m pip install matplotlib",
            )
        raise


def _resolve_output_files(
    context_file: Optional[str],
    report_file: Optional[str],
    chart_dir: Optional[str],
) -> Tuple[Path, Path, Path]:
    if context_file:
        context_path = Path(context_file).expanduser().resolve()
    else:
        context_path = CONTEXT_FILE

    if report_file:
        report_path = Path(report_file).expanduser().resolve()
    else:
        report_path = REPORT_FILE

    if chart_dir:
        charts_path = Path(chart_dir).expanduser().resolve()
    else:
        charts_path = CHART_DIR

    return context_path, report_path, charts_path


def _parse_timeframes(raw: Optional[str]) -> Sequence[str]:
    if not raw:
        return TIMEFRAMES
    timeframes = [item.strip() for item in raw.split(",") if item.strip()]
    if not timeframes:
        raise ValueError("timeframes cannot be empty")
    return timeframes


def _load_local_env_file(path: Path) -> None:
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for line in lines:
        current = line.strip()
        if not current or current.startswith("#"):
            continue
        if current.startswith("export "):
            current = current[len("export ") :].strip()
        if "=" not in current:
            continue
        key, value = current.split("=", 1)
        env_key = key.strip()
        env_value = value.strip()
        if env_key and env_key not in os.environ:
            os.environ[env_key] = env_value


def _sanitize_env_credential(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    cleaned = raw.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _looks_like_placeholder_credential(raw: Optional[str]) -> bool:
    if not raw:
        return False
    lowered = raw.lower()
    placeholder_values = {
        "test",
        "your_api_key",
        "your_api_secret",
        "your_read_only_key",
        "your_read_only_secret",
        "changeme",
    }
    return lowered in placeholder_values or lowered.startswith("your_")


def _iso_utc_from_ms(ts_ms: int) -> str:
    if not ts_ms:
        return ""
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def _compact_kline_rows(rows: Sequence[Dict], limit: int) -> list[Dict[str, Any]]:
    compact: list[Dict[str, Any]] = []
    for row in list(rows)[-limit:]:
        volume = float(row.get("volume", 0) or 0)
        quote_volume = float(row.get("quote_volume", 0) or 0)
        taker_buy_base = float(row.get("taker_buy_base", 0) or 0)
        taker_buy_quote = float(row.get("taker_buy_quote", 0) or 0)
        compact.append(
            {
                "open_time": _iso_utc_from_ms(int(row.get("open_time", 0))),
                "close_time": _iso_utc_from_ms(int(row.get("close_time", 0))),
                "open": round(float(row.get("open", 0) or 0), 2),
                "high": round(float(row.get("high", 0) or 0), 2),
                "low": round(float(row.get("low", 0) or 0), 2),
                "close": round(float(row.get("close", 0) or 0), 2),
                "volume": round(volume, 3),
                "quote_volume": round(quote_volume, 2),
                "taker_buy_base": round(taker_buy_base, 3),
                "taker_sell_base": round(max(0.0, volume - taker_buy_base), 3),
                "taker_buy_quote": round(taker_buy_quote, 2),
                "taker_sell_quote": round(max(0.0, quote_volume - taker_buy_quote), 2),
            }
        )
    return compact


def _compress_price_level_bins(all_bins: Sequence[Dict], current_price: float, limit: int = 18) -> list[Dict[str, Any]]:
    if not all_bins:
        return []
    selected: Dict[float, Dict[str, Any]] = {}
    ranked = sorted(
        list(all_bins),
        key=lambda row: float(row.get("total_quote", 0) or 0),
        reverse=True,
    )
    for row in ranked[: max(6, limit // 2)]:
        price = float(row.get("price", 0) or 0)
        selected[price] = row
    if current_price > 0:
        near_rows = [
            row
            for row in all_bins
            if abs(float(row.get("price", 0) or 0) - current_price) / max(current_price, 1.0) <= 0.012
        ]
        for row in near_rows[:limit]:
            price = float(row.get("price", 0) or 0)
            selected[price] = row
    rows = sorted(selected.values(), key=lambda row: float(row.get("price", 0) or 0))
    output: list[Dict[str, Any]] = []
    for row in rows[:limit]:
        output.append(
            {
                "price": round(float(row.get("price", 0) or 0), 2),
                "buy_quote": round(float(row.get("buy_quote", 0) or 0), 2),
                "sell_quote": round(float(row.get("sell_quote", 0) or 0), 2),
                "delta_quote": round(float(row.get("delta_quote", 0) or 0), 2),
                "imbalance": round(float(row.get("imbalance", 0) or 0), 3),
                "signal": row.get("signal", "balanced"),
                "total_quote": round(float(row.get("total_quote", 0) or 0), 2),
            }
        )
    return output


def _build_raw_appendix(context: Dict, klines_by_timeframe: Dict[str, list]) -> Dict[str, Any]:
    price = float(context.get("price", 0) or 0)
    append: Dict[str, Any] = {}

    kline_appendix: Dict[str, Any] = {}
    for tf, limit in (("4h", 12), ("1h", 24), ("15m", 32), ("5m", 48)):
        rows = _compact_kline_rows(klines_by_timeframe.get(tf, []), limit)
        if not rows:
            continue
        kline_appendix[tf] = {
            "bar_state": context.get("timeframes", {}).get(tf, {}).get("bar_state", {}),
            "bars": rows,
        }
    if kline_appendix:
        append["klines"] = kline_appendix

    oi_appendix: Dict[str, Any] = {}
    for period in ("5m", "15m", "1h"):
        series = context.get("open_interest_trend", {}).get("periods", {}).get(period, {}).get("series", [])
        if not series:
            continue
        oi_appendix[period] = [
            {
                "timestamp": _iso_utc_from_ms(int(row.get("timestamp", 0))),
                "price": round(float(row.get("price", 0) or 0), 2),
                "open_interest": round(float(row.get("open_interest", 0) or 0), 4),
                "price_delta_pct": round(float(row.get("price_delta_pct", 0) or 0), 4),
                "oi_delta_pct": round(float(row.get("oi_delta_pct", 0) or 0), 4),
                "state": row.get("state", "unknown"),
                "interpretation": row.get("interpretation", "unknown"),
            }
            for row in list(series)[-12:]
        ]
    if oi_appendix:
        append["open_interest_series"] = oi_appendix

    ls_appendix: Dict[str, Any] = {}
    for key in ("global_account", "top_trader_position"):
        history = context.get("long_short_ratio", {}).get(key, {}).get("history", [])
        if history:
            ls_appendix[key] = list(history)[-12:]
    if ls_appendix:
        append["long_short_history"] = ls_appendix

    trade_flow = context.get("trade_flow", {})
    pld = trade_flow.get("price_level_delta", {})
    trade_appendix: Dict[str, Any] = {}
    cvd_path = trade_flow.get("cvd_path", [])
    if cvd_path:
        trade_appendix["cvd_path_tail"] = [
            {
                "timestamp": _iso_utc_from_ms(int(row.get("timestamp", 0))),
                "price": round(float(row.get("price", 0) or 0), 2),
                "cvd_qty": round(float(row.get("cvd_qty", 0) or 0), 4),
                "delta_quote": round(float(row.get("delta_quote", 0) or 0), 2),
            }
            for row in list(cvd_path)[-20:]
        ]
    if isinstance(pld, dict) and pld.get("available"):
        trade_appendix["price_level_delta"] = {
            "window_minutes": pld.get("window_minutes"),
            "actual_coverage_minutes": pld.get("actual_coverage_minutes"),
            "bin_size": pld.get("bin_size"),
            "stacked_imbalance": pld.get("stacked_imbalance", []),
            "absorption_zones": pld.get("absorption_zones", []),
            "focus_bins": _compress_price_level_bins(pld.get("all_bins", []), current_price=price),
        }
    if trade_flow.get("absorption_zones"):
        trade_appendix["trade_absorption_zones"] = trade_flow.get("absorption_zones", [])[:6]
    if trade_flow.get("large_trade_clusters"):
        trade_appendix["large_trade_clusters"] = trade_flow.get("large_trade_clusters", [])[:6]
    if trade_appendix:
        append["trade_flow_raw"] = trade_appendix

    orderbook = context.get("orderbook_dynamics", {})
    if orderbook:
        append["orderbook_dynamics_raw"] = {
            "summary": {
                "sample_duration_seconds": orderbook.get("sample_duration_seconds"),
                "snapshot_count": orderbook.get("snapshot_count"),
                "spoofing_risk": orderbook.get("spoofing_risk"),
                "wall_behavior": orderbook.get("wall_behavior"),
                "cancel_to_add_ratio": orderbook.get("cancel_to_add_ratio"),
                "pull_vs_fill_ratio": orderbook.get("pull_vs_fill_ratio"),
                "wall_absorption_events": orderbook.get("wall_absorption_events"),
                "wall_sweep_events": orderbook.get("wall_sweep_events"),
                "wall_pull_without_trade_events": orderbook.get("wall_pull_without_trade_events"),
            },
            "top_bid_level_activity": orderbook.get("top_bid_level_activity", [])[:6],
            "top_ask_level_activity": orderbook.get("top_ask_level_activity", [])[:6],
            "persistent_walls": orderbook.get("persistent_walls", [])[:8],
            "series_tail": orderbook.get("series", [])[-12:],
        }

    liquidation_heatmap = context.get("liquidation_heatmap", {})
    if liquidation_heatmap:
        append["liquidation_heatmap_raw"] = {
            "source": liquidation_heatmap.get("source"),
            "confidence": liquidation_heatmap.get("confidence"),
            "zones": liquidation_heatmap.get("zones", [])[:8],
        }

    spot_perp = context.get("spot_perp", {})
    if spot_perp.get("available"):
        append["spot_perp_raw"] = {
            "spot_delta_quote": spot_perp.get("spot_delta_quote"),
            "spot_buy_quote_total": spot_perp.get("spot_buy_quote_total"),
            "spot_sell_quote_total": spot_perp.get("spot_sell_quote_total"),
            "spot_5m": spot_perp.get("spot_5m", {}),
            "spot_15m": spot_perp.get("spot_15m", {}),
            "basis_bps": spot_perp.get("basis_bps"),
            "spot_cvd_qty": spot_perp.get("spot_cvd_qty"),
        }

    return append

def build_market_context(
    symbol: str,
    timeframes: Sequence[str],
    kline_limit: int,
    depth_limit: int,
    agg_trades_limit: int,
    agg_trades_window_minutes: int,
    large_trade_quantile: float,
    oi_period: str,
    oi_limit: int,
    long_short_period: str,
    long_short_limit: int,
    orderbook_dynamic_samples: int,
    orderbook_dynamic_interval: float,
    volume_profile_window: int,
    volume_profile_bins: int,
    include_account: bool,
    api_key: Optional[str],
    api_secret: Optional[str],
    charts_path: Path,
    include_charts: bool,
    chart_generator_cls,
    cache_ttl: float = 0,
) -> Dict:
    collector = BinanceFuturesCollector(api_key=api_key, api_secret=api_secret, cache_ttl=cache_ttl)
    ext_collector = ExternalDataCollector()
    engine = IndicatorEngine()
    feature_extractor = FeatureExtractor()
    context_builder = MarketContextBuilder()

    # ── Phase 1: fire all independent network calls in parallel ──────────
    all_kline_intervals = list(dict.fromkeys(
        list(timeframes) + [tf for tf in ("1m", "5m", "15m", "1h") if tf not in timeframes]
    ))
    all_kline_limit = max(kline_limit, 120)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        fut_klines = {
            interval: pool.submit(collector.get_klines, symbol, interval, all_kline_limit)
            for interval in all_kline_intervals
        }
        fut_orderbook_snapshots = pool.submit(
            collector.get_orderbook_snapshots,
            symbol=symbol, limit=depth_limit,
            samples=orderbook_dynamic_samples,
            interval_seconds=orderbook_dynamic_interval,
        )
        fut_agg_trades = pool.submit(
            collector.get_agg_trades, symbol,
            limit=agg_trades_limit,
            window_minutes=agg_trades_window_minutes,
        )
        fut_oi = pool.submit(collector.get_open_interest, symbol)
        fut_funding = pool.submit(collector.get_funding, symbol)
        fut_ticker = pool.submit(collector.get_ticker_24h, symbol)
        fut_global_ls = pool.submit(
            collector.get_global_long_short_ratio,
            symbol=symbol, period=long_short_period, limit=long_short_limit,
        )
        fut_top_ls = pool.submit(
            collector.get_top_trader_long_short_ratio,
            symbol=symbol, period=long_short_period, limit=long_short_limit,
        )
        fut_oi_hist = {
            period: pool.submit(
                collector.get_open_interest_hist, symbol, period=period, limit=oi_limit,
            )
            for period in ("5m", "15m", "1h")
        }
        fut_force = pool.submit(collector.get_force_orders, symbol, limit=100)
        fut_spot_trades = pool.submit(collector.get_spot_agg_trades, symbol, 1000)
        fut_spot_ticker = pool.submit(collector.get_spot_ticker, symbol)
        fut_spot_klines_1h = pool.submit(collector.get_spot_klines, symbol, "1h", 120)
        fut_okx_oi = pool.submit(collector.get_okx_open_interest, symbol)
        fut_bybit_oi = pool.submit(collector.get_bybit_open_interest, symbol)
        fut_external = pool.submit(ext_collector.collect_all)

        if include_account:
            fut_account = pool.submit(collector.get_account_positions, symbol=symbol)
        else:
            fut_account = None

        # ── Collect klines results ───────────────────────────────────────
        klines_by_timeframe: Dict[str, list] = {}
        for interval, fut in fut_klines.items():
            klines_by_timeframe[interval] = fut.result()

        # ── Collect other phase-1 results ────────────────────────────────
        agg_trades = fut_agg_trades.result()
        open_interest = fut_oi.result()
        funding = fut_funding.result()
        ticker_24h = fut_ticker.result()
        global_long_short = fut_global_ls.result()
        top_trader_ratio = fut_top_ls.result()
        oi_histories = {period: fut.result() for period, fut in fut_oi_hist.items()}
        force_orders = fut_force.result()
        orderbook_snapshots = fut_orderbook_snapshots.result()
        spot_trades = fut_spot_trades.result()
        spot_ticker = fut_spot_ticker.result()
        spot_klines_1h = fut_spot_klines_1h.result()
        okx_oi = fut_okx_oi.result()
        bybit_oi = fut_bybit_oi.result()
        external_data = fut_external.result()

        if fut_account is not None:
            account_positions = fut_account.result()
        else:
            account_positions = {
                "available": False,
                "reason": "disabled_by_flag",
                "active_positions_count": 0,
                "active_positions": [],
                "symbol_position": None,
            }

        # ── Phase 2: cross-exchange funding (needs funding result) ───────
        fut_cross_funding = pool.submit(
            collector.get_cross_exchange_funding,
            symbol=symbol,
            binance_funding_rate=funding.get("funding_rate"),
        )
        cross_exchange_funding = fut_cross_funding.result()

    # ── Phase 3: compute indicators and features (CPU, no I/O) ───────────
    indicators_by_timeframe: Dict[str, Dict] = {}
    now_ms = int(time.time() * 1000)
    for timeframe in timeframes:
        candles = klines_by_timeframe.get(timeframe, [])
        indicators_by_timeframe[timeframe] = engine.calculate_for_candles(candles)
        last_candle = candles[-1]
        close_time = int(last_candle.get("close_time", 0))
        indicators_by_timeframe[timeframe]["bar_state"] = {
            "bar_closed": now_ms >= close_time,
            "open_time": int(last_candle.get("open_time", 0)),
            "close_time": close_time,
            "seconds_to_close": round(max(0.0, (close_time - now_ms) / 1000), 6),
        }

    timeframe_features = feature_extractor.extract_timeframe_features(indicators_by_timeframe)
    for timeframe, features in timeframe_features.items():
        indicators_by_timeframe[timeframe]["features"] = features

    orderbook = orderbook_snapshots[0] if orderbook_snapshots else collector.get_orderbook(symbol, depth_limit)
    if not orderbook_snapshots:
        orderbook_snapshots = [orderbook]
    orderbook_features = feature_extractor.extract_orderbook_features(orderbook)
    trade_flow = feature_extractor.extract_trade_flow_features(agg_trades, large_trade_quantile)
    trade_flow["kline_flow"] = feature_extractor.extract_kline_flow(klines_by_timeframe.get("1m", []))
    trade_flow["price_level_delta"] = feature_extractor.extract_price_level_delta(agg_trades)
    orderbook_dynamics = feature_extractor.extract_orderbook_dynamics(orderbook_snapshots, trades=agg_trades)

    volume_change = feature_extractor.extract_volume_change(klines_by_timeframe)
    price_candles_by_period = {
        period: klines_by_timeframe.get(period, [])
        for period in ("5m", "15m", "1h")
    }
    open_interest_trend = feature_extractor.extract_open_interest_trend(
        current_open_interest=open_interest,
        history_by_period=oi_histories,
        price_candles_by_period=price_candles_by_period,
        trade_flow=trade_flow,
        volume_change=volume_change,
        summary_period=oi_period,
    )

    long_short_ratio = feature_extractor.extract_long_short_ratio(
        global_account_ratio=global_long_short,
        top_trader_ratio=top_trader_ratio,
    )

    basis = feature_extractor.extract_basis(funding)
    funding_spread = feature_extractor.extract_cross_exchange_funding_spread(cross_exchange_funding)
    options_iv = feature_extractor.extract_options_iv_placeholder()

    recent_4h_range = feature_extractor.extract_recent_4h_range(klines_by_timeframe.get("15m", []))
    daily_anchors = feature_extractor.extract_daily_anchors(klines_by_timeframe.get("1d", []))
    profile_tf = "1h" if "1h" in klines_by_timeframe else timeframes[0]
    volume_profile = feature_extractor.extract_volume_profile(
        candles=klines_by_timeframe.get(profile_tf, []),
        bins=volume_profile_bins,
        window=volume_profile_window,
    )
    volume_profile["source_timeframe"] = profile_tf
    session_profile_tf = "15m" if "15m" in klines_by_timeframe else profile_tf
    volume_profile["session_source_timeframe"] = session_profile_tf
    volume_profile["session_profiles"] = feature_extractor.extract_session_profiles(
        candles=klines_by_timeframe.get(session_profile_tf, []),
        bins=volume_profile_bins,
    )
    volume_profile["anchored_source_timeframe"] = session_profile_tf
    volume_profile["anchored_profiles"] = feature_extractor.extract_anchored_profiles(
        candles=klines_by_timeframe.get(session_profile_tf, []),
        bins=volume_profile_bins,
    )
    session_context = feature_extractor.extract_session_context(
        candles=klines_by_timeframe.get("15m", []),
        funding=funding,
        stats_24h=ticker_24h,
    )

    primary_tf = timeframes[0]
    price = indicators_by_timeframe[primary_tf]["price"]
    liquidation_heatmap = feature_extractor.extract_liquidation_heatmap(
        current_price=price,
        recent_high=recent_4h_range["high"],
        recent_low=recent_4h_range["low"],
        force_orders=force_orders,
    )
    deployment_context = feature_extractor.extract_deployment_assessment(
        current_price=price,
        indicators_by_timeframe=indicators_by_timeframe,
        recent_4h_range=recent_4h_range,
        volume_profile=volume_profile,
        liquidation_heatmap=liquidation_heatmap,
        open_interest_trend=open_interest_trend,
        orderbook_dynamics=orderbook_dynamics,
        trade_flow=trade_flow,
        session_context=session_context,
    )

    ref_levels = deployment_context.get("reference_levels", [])
    key_level_inputs = [
        {"name": lv.get("name", "?"), "price": lv.get("price", 0)}
        for lv in ref_levels
        if float(lv.get("price", 0)) > 0
    ]
    trade_flow["key_level_flows"] = feature_extractor.extract_key_level_flows(
        agg_trades, key_level_inputs,
    )

    # Merge level test counts into each key_level_flow entry
    level_tests = feature_extractor.extract_key_level_tests(
        candles_15m=klines_by_timeframe.get("15m", []),
        key_levels=key_level_inputs,
    )
    for entry in trade_flow["key_level_flows"]:
        lookup_key = f"{entry.get('name')}@{entry.get('price')}"
        test_info = level_tests.get(lookup_key, {})
        entry["tests_12h"] = test_info.get("tests_12h")
        entry["first_test_min_ago"] = test_info.get("first_test_min_ago")
        entry["avg_bounce_pct"] = test_info.get("avg_bounce_pct")

    context = context_builder.build(
        symbol=symbol,
        price=price,
        indicators_by_timeframe=indicators_by_timeframe,
        account_positions=account_positions,
        orderbook_features=orderbook_features,
        orderbook_dynamics=orderbook_dynamics,
        open_interest=open_interest,
        open_interest_trend=open_interest_trend,
        long_short_ratio=long_short_ratio,
        funding=funding,
        basis=basis,
        cross_exchange_funding=cross_exchange_funding,
        funding_spread=funding_spread,
        options_iv=options_iv,
        stats_24h=ticker_24h,
        recent_4h_range=recent_4h_range,
        volume_change=volume_change,
        volume_profile=volume_profile,
        trade_flow=trade_flow,
        liquidation_heatmap=liquidation_heatmap,
        daily_anchors=daily_anchors,
        external_drivers=external_data,
        chart_files={},
    )
    position_sizing = feature_extractor.extract_position_sizing(
        current_price=price,
        indicators_by_timeframe=indicators_by_timeframe,
        account_positions=account_positions,
    )
    signal_score = feature_extractor.calculate_signal_score(
        indicators_by_timeframe=indicators_by_timeframe,
        orderbook_features=orderbook_features,
        orderbook_dynamics=orderbook_dynamics,
        open_interest_trend=open_interest_trend,
        long_short_ratio=long_short_ratio,
        trade_flow=trade_flow,
        funding=funding,
        basis=basis,
    )

    context["session_context"] = session_context
    context["deployment_context"] = deployment_context
    context["position_sizing"] = position_sizing
    context["signal_score"] = signal_score
    context["summary_files"] = {}

    # ── Candle structure for key timeframes (4h / 1h) ────────────────────────
    context["candle_structure"] = feature_extractor.extract_candle_structure(
        candles_by_timeframe=klines_by_timeframe,
        timeframes=("4h", "1h"),
        n=3,
    )

    primary_tf_for_price = timeframes[0]
    current_price_for_spot = indicators_by_timeframe[primary_tf_for_price]["price"]
    context["spot_perp"] = feature_extractor.extract_spot_perp_features(
        spot_trades=spot_trades,
        spot_ticker=spot_ticker,
        perp_price=current_price_for_spot,
        perp_funding_rate=float(funding.get("funding_rate", 0)),
    )
    context["cross_exchange_oi"] = feature_extractor.extract_cross_exchange_oi(
        binance_oi=open_interest,
        okx_oi=okx_oi,
        bybit_oi=bybit_oi,
        current_price=price,
    )
    context["transition"] = feature_extractor.extract_transition_features(
        open_interest_trend=open_interest_trend,
        basis=basis,
        funding=funding,
        trade_flow=trade_flow,
        spot_perp=context.get("spot_perp", {}),
        candles_5m=klines_by_timeframe.get("5m", []),
    )
    context["raw_appendix"] = _build_raw_appendix(context, klines_by_timeframe)

    if include_charts and chart_generator_cls is not None:
        chart_generator = chart_generator_cls(output_dir=charts_path, bars_by_timeframe=CHART_BARS)
        chart_files = chart_generator.generate(symbol, klines_by_timeframe, context=context)
        spot_perp_path = charts_path / f"{symbol}_spot_perp.png"
        sp_result = chart_generator.generate_spot_perp_chart(
            symbol=symbol,
            perp_klines=klines_by_timeframe.get("1h", []),
            spot_klines=spot_klines_1h,
            output_path=spot_perp_path,
            context=context,
        )
        if sp_result:
            chart_files["spot_perp"] = sp_result
        context["chart_files"] = chart_files

    # Stash raw 1h klines for analysis_history outcome resolution (not written to AI prompt)
    context["_klines_1h_raw"] = klines_by_timeframe.get("1h", [])

    return context


def _sanitize_context_for_ai_output(context: Dict) -> Dict:
    """Prepare context for the AI prompt.

    Keeps all interpretive fields (trend, state, divergence, etc.) so the
    prompt generator can reference them.  Only strips bulky internal data
    that would waste tokens without helping the AI (raw series, history
    arrays, cvd_path, etc.).
    """
    sanitized = copy.deepcopy(context)
    sanitized["summary_files"] = {}
    sanitized.pop("_klines_1h_raw", None)  # internal only, not for AI prompt

    for metrics in sanitized.get("timeframes", {}).values():
        if not isinstance(metrics, dict):
            continue
        metrics.pop("bar_state", None)

    oi_trend = sanitized.get("open_interest_trend", {})
    if isinstance(oi_trend, dict):
        for period_data in oi_trend.get("periods", {}).values():
            if isinstance(period_data, dict):
                period_data.pop("series", None)

    for ratio_block_key in ("global_account", "top_trader_position"):
        block = sanitized.get("long_short_ratio", {}).get(ratio_block_key)
        if isinstance(block, dict):
            block.pop("history", None)

    trade_flow = sanitized.get("trade_flow", {})
    if isinstance(trade_flow, dict):
        trade_flow.pop("cvd_path", None)
        pld = trade_flow.get("price_level_delta", {})
        if isinstance(pld, dict):
            pld.pop("all_bins", None)

    return sanitized


def _cleanup_alternate_prompt_output(report_path: Path) -> None:
    try:
        is_default_prompt = report_path.resolve() == REPORT_FILE.resolve()
    except OSError:
        is_default_prompt = report_path == REPORT_FILE
    if not is_default_prompt:
        return

    stale_names = ["btc_user_prompt.txt", "btc_report.txt"]
    for name in stale_names:
        old = report_path.with_name(name)
        if old == report_path or not old.exists():
            continue
        try:
            old.unlink()
        except OSError:
            pass


def write_outputs(
    context: Dict,
    context_path: Path,
    report_path: Path,
    report_mode: str = "raw_first",
) -> Dict:
    context_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ai_context = _sanitize_context_for_ai_output(context)

    with context_path.open("w", encoding="utf-8") as f:
        json.dump(ai_context, f, ensure_ascii=False, indent=2)

    gen = PromptGenerator()

    data_prompt = gen.build(ai_context, report_mode=report_mode, include_instructions=False)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(data_prompt)
    _cleanup_alternate_prompt_output(report_path)

    system_path = report_path.with_name(SYSTEM_PROMPT_FILE.name)
    system_text = gen.build_system_prompt(stage="decision", include_vision=False)
    with system_path.open("w", encoding="utf-8") as f:
        f.write(system_text)

    total = len(data_prompt) + len(system_text)
    cn_chars = sum(1 for ch in data_prompt if "\u4e00" <= ch <= "\u9fff")
    en_chars = len(data_prompt) - cn_chars
    est_tokens = int(cn_chars / 3.5 + en_chars / 4)
    logger.info(
        "output: system=%d chars -> %s  |  data=%d chars (~%d tokens) -> %s",
        len(system_text), system_path.name,
        len(data_prompt), est_tokens, report_path.name,
    )

    return ai_context

def write_summary_output(context: Dict, context_path: Path) -> Path:
    summary_path = SUMMARY_FILE
    if summary_path.parent != context_path.parent:
        summary_path = context_path.parent / summary_path.name

    summary = SummaryTableGenerator().build(context)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary)
    return summary_path


def _resolve_related_output_path(anchor_path: Path, default_path: Path) -> Path:
    if anchor_path.parent == default_path.parent:
        return default_path
    return anchor_path.parent / default_path.name


def _parse_research_handoff(text: str) -> Dict[str, Any]:
    def _repair_json_brackets(source: str) -> str:
        repaired: list[str] = []
        stack: list[str] = []
        in_string = False
        escape = False
        for ch in source:
            if in_string:
                repaired.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                repaired.append(ch)
                continue

            if ch == "{":
                stack.append("}")
                repaired.append(ch)
                continue
            if ch == "[":
                stack.append("]")
                repaired.append(ch)
                continue
            if ch in "}]":
                if stack:
                    expected = stack.pop()
                    repaired.append(expected)
                else:
                    repaired.append(ch)
                continue
            repaired.append(ch)

        while stack:
            repaired.append(stack.pop())
        return "".join(repaired)

    raw = (text or "").strip()
    candidates: list[str] = []
    if raw:
        candidates.append(raw)
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1).strip())
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(raw[start : end + 1].strip())

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                repaired = _repair_json_brackets(candidate)
                obj = json.loads(repaired)
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            return obj

    return {
        "stage": "research",
        "parse_error": "invalid_json",
        "raw_text": raw,
    }


def _load_ai_advisor():
    try:
        try:
            from .advisor import AIAdvisor
        except ImportError:
            from advisor import AIAdvisor
        return AIAdvisor
    except ImportError:
        return None


def _ordered_chart_items_for_ai(
    context: Dict,
    stage: str = "research",
) -> list[tuple[str, str]]:
    """(标签, 路径) 顺序与多周期由大到小，便于模型读图。"""
    if stage == "decision":
        order = ("4h", "1h", "15m", "spot_perp")
    else:
        order = ("4h", "1h", "15m", "5m", "spot_perp")
    cf = context.get("chart_files") or {}
    items: list[tuple[str, str]] = []
    for k in order:
        if k in cf and cf[k]:
            label = "现货vs永续" if k == "spot_perp" else k
            items.append((label, str(cf[k])))
    return items


def _run_ai_analysis(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    base_url: str | None = None,
    chart_items: list[tuple[str, str]] | None = None,
    stage_name: str = "decision",
) -> str:
    AIAdvisor = _load_ai_advisor()
    if AIAdvisor is None:
        raise RuntimeError("advisor module could not be loaded")
    advisor = AIAdvisor(api_key=api_key, model=model, base_url=base_url)
    n_img = 0
    if chart_items:
        from pathlib import Path as _P

        for _, p in chart_items:
            pp = _P(p)
            if pp.is_file() and pp.suffix.lower() == ".png":
                n_img += 1
    token_est = advisor.estimate_tokens(system_prompt + "\n\n" + user_prompt, num_images=n_img)
    print(f"estimated {stage_name} prompt tokens (含图约估): ~{token_est}")
    analysis = advisor.analyze_with_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        chart_items=chart_items or None,
    )
    if not analysis or not analysis.strip():
        raise RuntimeError("AI analysis returned empty content; output file was not updated")
    return analysis


def _send_pushplus(context: Dict, analysis_text: str) -> None:
    token = os.getenv(PUSHPLUS_TOKEN_ENV, "").strip()
    if not token:
        logger.warning("PushPlus skipped: %s not set", PUSHPLUS_TOKEN_ENV)
        return
    try:
        try:
            from .advisor.pushplus_notifier import PushPlusNotifier, _is_wait_decision, _full_content
        except ImportError:
            from advisor.pushplus_notifier import PushPlusNotifier, _is_wait_decision, _full_content
        try:
            from .advisor.smart_ai_scheduler import analysis_has_open_position
        except ImportError:
            from advisor.smart_ai_scheduler import analysis_has_open_position
        notifier = PushPlusNotifier(token=token)
        price = context.get("price", 0)
        symbol = context.get("symbol", "BTCUSDT")
        # 持仓管理场景：有持仓就推送，无论是 hold/reduce/add
        if analysis_has_open_position(analysis_text):
            body = _full_content(analysis_text, max_len=4000)
            sent = notifier.send(f"📋 {symbol} 持仓更新 @{price:.0f}", body, template="markdown")
            if sent:
                logger.info("PushPlus position update sent to WeChat")
            else:
                logger.warning("PushPlus push FAILED — position update delivery failed")
            return
        # 空仓场景：仅有新开单方案时推送
        if _is_wait_decision(analysis_text):
            logger.info("PushPlus skipped: AI decision is wait/hold (no actionable setup)")
            return
        sent = notifier.send_trade_signal(analysis_text, context)
        if sent:
            logger.info("PushPlus trade signal sent to WeChat")
        else:
            logger.warning("PushPlus push FAILED — signal was actionable but delivery failed")
    except Exception as exc:
        logger.error("PushPlus notification error: %s", exc)


def _send_telegram(context: Dict, analysis_text: str = None) -> None:
    bot_token = _sanitize_env_credential(os.getenv(TELEGRAM_BOT_TOKEN_ENV))
    chat_id = os.getenv(TELEGRAM_CHAT_ID_ENV, "").strip()
    if not bot_token or not chat_id:
        logger.warning(
            "Telegram notification skipped: %s and/or %s not set",
            TELEGRAM_BOT_TOKEN_ENV, TELEGRAM_CHAT_ID_ENV,
        )
        return
    try:
        try:
            from .advisor.telegram_notifier import TelegramNotifier
        except ImportError:
            from advisor.telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
        signal_score = context.get("signal_score", {})
        ok = notifier.send_analysis_summary(
            signal_score=signal_score,
            analysis_text=analysis_text,
        )
        if ok:
            logger.info("Telegram notification sent")
        else:
            logger.warning("Telegram notification may have failed")
    except Exception as exc:
        logger.error("Telegram notification error: %s", exc)


def _resolve_analysis_path(context_path: Path) -> Path:
    if context_path.parent == AI_ANALYSIS_FILE.parent:
        return AI_ANALYSIS_FILE
    return context_path.parent / AI_ANALYSIS_FILE.name


def _resolve_research_prompt_path(report_path: Path) -> Path:
    return _resolve_related_output_path(report_path, RESEARCH_PROMPT_FILE)


def _resolve_research_system_path(report_path: Path) -> Path:
    return _resolve_related_output_path(report_path, RESEARCH_SYSTEM_FILE)


def _resolve_decision_prompt_path(report_path: Path) -> Path:
    return _resolve_related_output_path(report_path, DECISION_PROMPT_FILE)


def _resolve_decision_system_path(report_path: Path) -> Path:
    return _resolve_related_output_path(report_path, DECISION_SYSTEM_FILE)


def _resolve_research_analysis_path(context_path: Path) -> Path:
    return _resolve_related_output_path(context_path, AI_RESEARCH_FILE)


def _resolve_research_handoff_path(context_path: Path) -> Path:
    return _resolve_related_output_path(context_path, RESEARCH_HANDOFF_FILE)


def _resolve_decision_analysis_path(context_path: Path) -> Path:
    return _resolve_related_output_path(context_path, AI_DECISION_FILE)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTCUSDT perpetual market context analyzer")
    parser.add_argument("--symbol", default=SYMBOL, help="Trading symbol, default BTCUSDT")
    parser.add_argument(
        "--timeframes",
        default=",".join(TIMEFRAMES),
        help="Comma-separated kline intervals, e.g. 15m,1h,4h",
    )
    parser.add_argument("--kline-limit", type=int, default=KLINE_LIMIT, help="Kline size per timeframe")
    parser.add_argument("--depth-limit", type=int, default=DEPTH_LIMIT, help="Depth levels per side")
    parser.add_argument(
        "--agg-trades-limit",
        type=int,
        default=AGG_TRADES_LIMIT,
        help="AggTrades sample size for CVD/Delta calculation",
    )
    parser.add_argument(
        "--agg-trades-window-minutes",
        type=int,
        default=AGG_TRADES_WINDOW_MINUTES,
        help="Time window (minutes) for aggTrades collection; 0=use count-based limit only",
    )
    parser.add_argument(
        "--large-trade-quantile",
        type=float,
        default=LARGE_TRADE_QUANTILE,
        help="Quantile threshold for large trades (0~1)",
    )
    parser.add_argument("--oi-period", default=OI_PERIOD, help="Open-interest history period (e.g. 5m,15m,1h)")
    parser.add_argument("--oi-limit", type=int, default=OI_LIMIT, help="Open-interest history points")
    parser.add_argument(
        "--long-short-period",
        default=LONG_SHORT_PERIOD,
        help="Long/short ratio period (e.g. 5m,15m,1h)",
    )
    parser.add_argument(
        "--long-short-limit",
        type=int,
        default=LONG_SHORT_LIMIT,
        help="Long/short ratio history points",
    )
    parser.add_argument(
        "--orderbook-dynamic-samples",
        type=int,
        default=ORDERBOOK_DYNAMIC_SAMPLES,
        help="Number of depth snapshots for add/cancel dynamics",
    )
    parser.add_argument(
        "--orderbook-dynamic-interval",
        type=float,
        default=ORDERBOOK_DYNAMIC_INTERVAL,
        help="Seconds between depth snapshots",
    )
    parser.add_argument(
        "--volume-profile-window",
        type=int,
        default=VOLUME_PROFILE_WINDOW,
        help="How many candles to use for volume profile",
    )
    parser.add_argument(
        "--volume-profile-bins",
        type=int,
        default=VOLUME_PROFILE_BINS,
        help="Price bins for volume profile histogram",
    )
    parser.add_argument("--context-file", default=None, help="Optional custom context json path")
    parser.add_argument(
        "--report-file",
        "--prompt-file",
        dest="report_file",
        default=None,
        help="Optional custom report/prompt txt path",
    )
    parser.add_argument(
        "--report-mode",
        dest="report_mode",
        default=REPORT_MODE,
        choices=["raw_first", "full_debug"],
        help=(
            "Report generation mode: 'raw_first' (default) outputs only raw facts with no "
            "directional labels; 'full_debug' appends derived/score/deployment fields as an appendix."
        ),
    )
    parser.add_argument("--chart-dir", default=None, help="Optional custom chart output directory")
    parser.add_argument(
        "--include-summary",
        action="store_true",
        help="Generate the optional human-readable summary markdown file",
    )
    parser.add_argument("--no-charts", action="store_true", help="Disable kline screenshot generation")
    parser.add_argument(
        "--include-account",
        action="store_true",
        help=(
            "Force include current futures positions from Binance private read-only API. "
            f"Requires env vars {BINANCE_API_KEY_ENV}/{BINANCE_API_SECRET_ENV}."
        ),
    )
    parser.add_argument(
        "--no-account",
        action="store_true",
        help="Disable account position collection even if API credentials are present.",
    )
    parser.add_argument(
        "--auto-analyze",
        action="store_true",
        help=(
            "Send the generated prompt to an AI provider for analysis. "
            f"Requires env var {OPENAI_API_KEY_ENV} (openai) or {DEEPSEEK_API_KEY_ENV} (deepseek)."
        ),
    )
    parser.add_argument(
        "--ai-provider",
        default="openai",
        choices=["openai", "deepseek"],
        help="AI provider for --auto-analyze (default: openai / ChatGPT API)",
    )
    parser.add_argument(
        "--ai-model",
        default=None,
        help=(
            f"Model override for --auto-analyze. "
            f"Defaults: openai={OPENAI_MODEL}, deepseek={DEEPSEEK_MODEL}"
        ),
    )
    parser.add_argument(
        "--no-ai-charts",
        action="store_true",
        help="--auto-analyze 时不把 K 线 PNG 传给模型（默认会传图，需 vision 模型如 gpt-4o-mini）",
    )
    parser.add_argument(
        "--loop",
        dest="loop_sec",
        nargs="?",
        const=300,
        default=None,
        type=int,
        metavar="SEC",
        help="循环简写：等同 --watch SEC；只写 --loop 默认 300 秒（会覆盖已写的 --watch）。",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help=(
            "一键循环监控：打开 --auto-analyze --smart --pushplus --cache-ttl 30，"
            "默认每 600 秒（10 分钟）一轮；可用 --loop 300 改成 5 分钟等。"
        ),
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="每 N 秒重新跑一轮（0=只跑一趟）。与 --loop 二选一即可，--loop 优先覆盖本项。",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate an HTML dashboard report in the output directory",
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=0,
        metavar="SECONDS",
        help="Cache API responses for N seconds (useful with --watch to reduce redundant calls, e.g. --cache-ttl 30)",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help=(
            "Send analysis summary to Telegram. "
            f"Requires env vars {TELEGRAM_BOT_TOKEN_ENV} and {TELEGRAM_CHAT_ID_ENV}."
        ),
    )
    parser.add_argument(
        "--pushplus",
        action="store_true",
        help=(
            "Send trade signals to WeChat via PushPlus. "
            f"Requires env var {PUSHPLUS_TOKEN_ENV}. "
            "Only sends when AI produces an actionable setup (not wait)."
        ),
    )
    parser.add_argument(
        "--smart",
        action="store_true",
        help=(
            "Smart analysis mode: skip AI call when market context hasn't "
            "changed materially since last analysis. Saves API cost with "
            "--watch. Falls back to full analysis every ~20 min regardless."
        ),
    )
    return parser.parse_args(argv)


def _apply_cli_shortcuts(args: argparse.Namespace) -> None:
    """--loop 先定间隔；--monitor 再补全 AI/推送/缓存，并在未指定间隔时默认 600s（10 分钟）。"""
    ls = getattr(args, "loop_sec", None)
    if ls is not None:
        args.watch = max(1, int(ls))
    if getattr(args, "monitor", False):
        args.auto_analyze = True
        args.smart = True
        args.pushplus = True
        if not args.ai_model and args.ai_provider == "openai":
            args.ai_model = OPENAI_MODEL
        if args.cache_ttl <= 0:
            args.cache_ttl = 30.0
        if args.watch <= 0:
            args.watch = 600


def _run_once(args: argparse.Namespace) -> int:
    """Execute a single collect -> analyze cycle. Returns 0 on success."""
    timeframes = _parse_timeframes(args.timeframes)
    if args.kline_limit <= 0:
        raise ValueError("kline-limit must be greater than 0")
    if args.depth_limit <= 0:
        raise ValueError("depth-limit must be greater than 0")
    if args.agg_trades_limit <= 0:
        raise ValueError("agg-trades-limit must be greater than 0")
    if not 0 <= args.large_trade_quantile <= 1:
        raise ValueError("large-trade-quantile must be between 0 and 1")
    if args.oi_limit <= 0:
        raise ValueError("oi-limit must be greater than 0")
    if args.long_short_limit <= 0:
        raise ValueError("long-short-limit must be greater than 0")
    if args.orderbook_dynamic_samples <= 0:
        raise ValueError("orderbook-dynamic-samples must be greater than 0")
    if args.orderbook_dynamic_interval < 0:
        raise ValueError("orderbook-dynamic-interval must be >= 0")
    if args.volume_profile_window <= 0:
        raise ValueError("volume-profile-window must be greater than 0")
    if args.volume_profile_bins <= 0:
        raise ValueError("volume-profile-bins must be greater than 0")
    if args.include_account and args.no_account:
        raise ValueError("include-account and no-account cannot both be set")

    context_path, report_path, charts_path = _resolve_output_files(
        args.context_file,
        args.report_file,
        args.chart_dir,
    )

    api_key = _sanitize_env_credential(os.getenv(BINANCE_API_KEY_ENV))
    api_secret = _sanitize_env_credential(os.getenv(BINANCE_API_SECRET_ENV))
    if _looks_like_placeholder_credential(api_key) or _looks_like_placeholder_credential(api_secret):
        logger.warning(
            "detected placeholder %s/%s; account position collection disabled",
            BINANCE_API_KEY_ENV, BINANCE_API_SECRET_ENV,
        )
        api_key = None
        api_secret = None
    has_api_credentials = bool(api_key and api_secret)
    include_account = not args.no_account and (args.include_account or has_api_credentials)
    if include_account and (not api_key or not api_secret):
        logger.warning(
            "account collection enabled but usable %s/%s not found; "
            "account position data will be unavailable",
            BINANCE_API_KEY_ENV, BINANCE_API_SECRET_ENV,
        )

    # ── Risk monitor: check trading discipline rules before market data fetch ──
    pushplus_token = os.getenv(PUSHPLUS_TOKEN_ENV, "").strip()
    if has_api_credentials and pushplus_token:
        try:
            try:
                from .advisor.risk_monitor import RiskMonitor
            except ImportError:
                from advisor.risk_monitor import RiskMonitor
            _risk_collector = BinanceFuturesCollector(api_key=api_key, api_secret=api_secret)
            RiskMonitor().check_and_alert(_risk_collector, pushplus_token)
        except Exception as _e:
            logger.warning("risk monitor skipped: %s", _e)

    chart_generator_cls = None
    chart_warning = None
    include_charts = not args.no_charts
    if include_charts:
        chart_generator_cls, chart_warning = _load_chart_generator()
        if chart_generator_cls is None:
            include_charts = False
            logger.warning("%s", chart_warning)

    context = build_market_context(
        symbol=args.symbol,
        timeframes=timeframes,
        kline_limit=args.kline_limit,
        depth_limit=args.depth_limit,
        agg_trades_limit=args.agg_trades_limit,
        agg_trades_window_minutes=args.agg_trades_window_minutes,
        large_trade_quantile=args.large_trade_quantile,
        oi_period=args.oi_period,
        oi_limit=args.oi_limit,
        long_short_period=args.long_short_period,
        long_short_limit=args.long_short_limit,
        orderbook_dynamic_samples=args.orderbook_dynamic_samples,
        orderbook_dynamic_interval=args.orderbook_dynamic_interval,
        volume_profile_window=args.volume_profile_window,
        volume_profile_bins=args.volume_profile_bins,
        include_account=include_account,
        api_key=api_key,
        api_secret=api_secret,
        charts_path=charts_path,
        include_charts=include_charts,
        chart_generator_cls=chart_generator_cls,
        cache_ttl=args.cache_ttl,
    )
    # ── Analysis history: update outcomes using already-fetched 1h klines ────
    try:
        try:
            from .advisor.analysis_history import AnalysisHistory
        except ImportError:
            from advisor.analysis_history import AnalysisHistory
        _history = AnalysisHistory()
        _history.update_outcomes(context.get("_klines_1h_raw", []))
        context["prior_decisions"] = _history.get_context_block()
    except Exception as _e:
        logger.warning("analysis_history update skipped: %s", _e)
        context["prior_decisions"] = ""

    ai_context = write_outputs(context, context_path, report_path, report_mode=args.report_mode)
    if args.include_summary:
        summary_path = write_summary_output(context, context_path)
        with context_path.open("w", encoding="utf-8") as f:
            ai_context["summary_files"] = {"overview": str(summary_path.resolve())}
            json.dump(ai_context, f, ensure_ascii=False, indent=2)
    logger.info("context written to %s", context_path)
    logger.info("data prompt written to %s", report_path)
    logger.info("system prompt written to %s", report_path.with_name(SYSTEM_PROMPT_FILE.name))
    if args.include_summary:
        logger.info("overview summary written to %s", summary_path)
    if context.get("chart_files"):
        logger.info("charts:")
        for timeframe, file_path in context["chart_files"].items():
            logger.info("  %s: %s", timeframe, file_path)

    analysis = ""
    skip_ai = False
    if args.auto_analyze:
        try:
            try:
                from .advisor.smart_ai_scheduler import (
                    SKIPS_BEFORE_FORCE,
                    analysis_has_actionable_signal,
                    analysis_has_open_position,
                    load_scheduler_state,
                    save_scheduler_state,
                )
            except ImportError:
                from advisor.smart_ai_scheduler import (
                    SKIPS_BEFORE_FORCE,
                    analysis_has_actionable_signal,
                    analysis_has_open_position,
                    load_scheduler_state,
                    save_scheduler_state,
                )
        except ImportError:
            load_scheduler_state = save_scheduler_state = None  # type: ignore
            analysis_has_actionable_signal = lambda t: False  # type: ignore
            analysis_has_open_position = lambda t: False  # type: ignore
            SKIPS_BEFORE_FORCE = 3  # type: ignore

        sched = load_scheduler_state() if load_scheduler_state else {}
        fast_ai = bool(sched.get("fast_ai_mode"))
        n_skip = int(sched.get("consecutive_skips", 0))
        force_after_skips = bool(args.smart and n_skip >= SKIPS_BEFORE_FORCE)

        if args.smart:
            if fast_ai:
                logger.info(
                    "smart: 快速分析模式（约每 4 分钟）直至主结论 wait — 本轮必跑 AI",
                )
                skip_ai = False
            elif force_after_skips:
                logger.info(
                    "smart: 已连续跳过 %d 次 AI（≥%d），本轮强制执行",
                    n_skip,
                    SKIPS_BEFORE_FORCE,
                )
                skip_ai = False
            else:
                try:
                    try:
                        from .advisor.change_detector import ChangeDetector
                    except ImportError:
                        from advisor.change_detector import ChangeDetector
                    detector = ChangeDetector()
                    should, reasons = detector.should_analyze(context)
                    if should:
                        logger.info("smart mode: AI triggered — %s", "; ".join(reasons))
                    else:
                        logger.info("smart mode: no material change, skipping AI call")
                        skip_ai = True
                except Exception as exc:
                    logger.warning("smart mode detection failed, running AI anyway: %s", exc)

        if args.smart and skip_ai and save_scheduler_state:
            sched["consecutive_skips"] = n_skip + 1
            save_scheduler_state(sched)
            logger.info(
                "smart scheduler: 连续跳过 %d/%d 次后下次将强制 AI",
                sched["consecutive_skips"],
                SKIPS_BEFORE_FORCE,
            )

        if not skip_ai:
            provider = args.ai_provider
            if provider == "deepseek":
                ai_api_key = _sanitize_env_credential(os.getenv(DEEPSEEK_API_KEY_ENV))
                key_env_name = DEEPSEEK_API_KEY_ENV
                default_model = DEEPSEEK_MODEL
                base_url = DEEPSEEK_BASE_URL
            else:
                ai_api_key = _sanitize_env_credential(os.getenv(OPENAI_API_KEY_ENV))
                key_env_name = OPENAI_API_KEY_ENV
                default_model = OPENAI_MODEL
                base_url = None

            if not ai_api_key:
                logger.error("--auto-analyze requires %s to be set", key_env_name)
                return 1
            model = args.ai_model or default_model
            research_chart_items = None
            decision_chart_items = None
            if not args.no_ai_charts:
                if provider == "deepseek":
                    logger.info(
                        "DeepSeek 仅支持纯文本，不传 K 线图（避免 400）。"
                        "看图请改用: --ai-provider openai --ai-model gpt-4o-mini"
                    )
                else:
                    research_chart_items = _ordered_chart_items_for_ai(context, stage="research")
                    decision_chart_items = _ordered_chart_items_for_ai(context, stage="decision")
                    if research_chart_items:
                        logger.info(
                            "research 阶段将附带 K 线图: %s",
                            ", ".join(f"{a}→{Path(b).name}" for a, b in research_chart_items),
                        )
                    else:
                        logger.info(
                            "research 阶段: 无可用图表（可装 matplotlib 或去掉 --no-charts）"
                        )
                    if decision_chart_items:
                        logger.info(
                            "decision 阶段将附带关键图表: %s",
                            ", ".join(f"{a}→{Path(b).name}" for a, b in decision_chart_items),
                        )
            gen = PromptGenerator()

            research_system = gen.build_system_prompt(
                stage="research",
                include_vision=bool(research_chart_items),
            )
            research_prompt = gen.build_research_prompt(ai_context, include_instructions=False)
            research_system_path = _resolve_research_system_path(report_path)
            research_prompt_path = _resolve_research_prompt_path(report_path)
            research_system_path.write_text(research_system, encoding="utf-8")
            research_prompt_path.write_text(research_prompt, encoding="utf-8")
            logger.info("sending research prompt to %s/%s for analysis...", provider, model)
            research_text = _run_ai_analysis(
                ai_api_key,
                model,
                system_prompt=research_system,
                user_prompt=research_prompt,
                base_url=base_url,
                chart_items=research_chart_items,
                stage_name="research",
            )
            research_analysis_path = _resolve_research_analysis_path(context_path)
            research_analysis_path.parent.mkdir(parents=True, exist_ok=True)
            research_analysis_path.write_text(research_text, encoding="utf-8")
            logger.info("AI research written to %s", research_analysis_path)

            research_handoff = _parse_research_handoff(research_text)
            research_handoff_text = json.dumps(research_handoff, ensure_ascii=False, indent=2)
            research_handoff_path = _resolve_research_handoff_path(context_path)
            research_handoff_path.write_text(research_handoff_text, encoding="utf-8")
            logger.info("AI research handoff written to %s", research_handoff_path)

            decision_system = gen.build_system_prompt(
                stage="decision",
                include_vision=bool(decision_chart_items),
            )
            decision_prompt = gen.build_decision_prompt(
                ai_context,
                research_handoff=research_handoff_text,
                report_mode=args.report_mode,
                include_instructions=False,
            )
            decision_system_path = _resolve_decision_system_path(report_path)
            decision_prompt_path = _resolve_decision_prompt_path(report_path)
            decision_system_path.write_text(decision_system, encoding="utf-8")
            decision_prompt_path.write_text(decision_prompt, encoding="utf-8")
            report_path.with_name(SYSTEM_PROMPT_FILE.name).write_text(decision_system, encoding="utf-8")
            logger.info("sending decision prompt to %s/%s for analysis...", provider, model)
            analysis = _run_ai_analysis(
                ai_api_key,
                model,
                system_prompt=decision_system,
                user_prompt=decision_prompt,
                base_url=base_url,
                chart_items=decision_chart_items,
                stage_name="decision",
            )
            decision_analysis_path = _resolve_decision_analysis_path(context_path)
            decision_analysis_path.parent.mkdir(parents=True, exist_ok=True)
            decision_analysis_path.write_text(analysis, encoding="utf-8")
            logger.info("AI decision written to %s", decision_analysis_path)
            analysis_path = _resolve_analysis_path(context_path)
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_path.write_text(analysis, encoding="utf-8")
            logger.info("AI analysis written to %s", analysis_path)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_analysis_path = analysis_path.with_name(
                f"{analysis_path.stem}_{ts}{analysis_path.suffix}"
            )
            ts_analysis_path.write_text(analysis, encoding="utf-8")
            logger.info("AI analysis archived to %s", ts_analysis_path.name)
            print("\n" + "=" * 60)
            print(analysis)
            print("=" * 60)

            # Record this analysis call for future bias calibration
            try:
                try:
                    from .advisor.analysis_history import AnalysisHistory as _AH
                except ImportError:
                    from advisor.analysis_history import AnalysisHistory as _AH
                _AH().record(analysis, float(context.get("price", 0)))
            except Exception as _e:
                logger.warning("analysis_history record skipped: %s", _e)

            if args.smart:
                try:
                    try:
                        from .advisor.change_detector import ChangeDetector as _CD
                    except ImportError:
                        from advisor.change_detector import ChangeDetector as _CD
                    _CD().save_state(context, analysis_text=analysis)
                except Exception:
                    pass
            if save_scheduler_state and load_scheduler_state:
                sched2 = load_scheduler_state()
                sched2["consecutive_skips"] = 0
                has_signal = analysis_has_actionable_signal(analysis)
                has_position = analysis_has_open_position(analysis)
                if has_signal:
                    sched2["fast_ai_mode"] = True
                    logger.info(
                        "smart scheduler: 检测到可执行方案 → 接下来约每 4 分钟分析，直到 wait",
                    )
                elif has_position:
                    sched2["fast_ai_mode"] = True
                    logger.info(
                        "smart scheduler: 检测到持仓中 → 继续每 4 分钟分析，直到持仓平仓",
                    )
                else:
                    was_fast = sched2.get("fast_ai_mode")
                    sched2["fast_ai_mode"] = False
                    if was_fast and fast_ai:
                        logger.info(
                            "smart scheduler: 空仓且无挂单方案 → 退出快速分析，恢复常规定时",
                        )
                save_scheduler_state(sched2)

    if args.html:
        try:
            try:
                from .reports.html_report import HtmlReportGenerator
            except ImportError:
                from reports.html_report import HtmlReportGenerator
            html_content = HtmlReportGenerator().build(
                context, analysis_text=analysis if args.auto_analyze else "",
            )
            html_path = context_path.with_suffix(".html")
            html_path.write_text(html_content, encoding="utf-8")
            logger.info("HTML dashboard written to %s", html_path)
        except Exception as exc:
            logger.warning("HTML report generation failed: %s", exc)

    if args.telegram:
        _send_telegram(context, analysis_text=analysis if args.auto_analyze else None)

    if args.pushplus and analysis:
        _send_pushplus(context, analysis)

    return 0


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    _setup_logging()
    logger.info("running under Python executable: %s", sys.executable)
    args = parse_args(argv if argv is not None else sys.argv[1:])
    _load_local_env_file(Path(__file__).resolve().parent / ".env.local")
    _apply_cli_shortcuts(args)
    if getattr(args, "monitor", False):
        logger.info(
            "monitor 模式: auto-analyze + smart + pushplus, cache_ttl=%s, watch=%ss",
            args.cache_ttl,
            args.watch,
        )

    watch_interval = max(0, args.watch)
    iteration = 0
    while True:
        iteration += 1
        if watch_interval:
            logger.info("=" * 20 + " iteration #%d " + "=" * 20, iteration)
        try:
            rc = _run_once(args)
            if rc != 0 and not watch_interval:
                return rc
        except Exception as exc:
            logger.error("failed to build market context: %s", exc)
            if not watch_interval:
                return 1

        if not watch_interval:
            return 0

        try:
            try:
                from .advisor.smart_ai_scheduler import (
                    FAST_POLL_SECONDS,
                    load_scheduler_state as _load_sched,
                )
            except ImportError:
                from advisor.smart_ai_scheduler import (
                    FAST_POLL_SECONDS,
                    load_scheduler_state as _load_sched,
                )
            st = _load_sched()
            use_fast = bool(
                args.auto_analyze and st.get("fast_ai_mode") and watch_interval > 0
            )
            sleep_sec = FAST_POLL_SECONDS if use_fast else watch_interval
        except Exception:
            sleep_sec = watch_interval
        if sleep_sec != watch_interval:
            logger.info(
                "[watch] 快速分析中: %ds 后下一轮 (常规定时 %ds)… Ctrl+C 停止",
                sleep_sec,
                watch_interval,
            )
        else:
            logger.info("[watch] sleeping %ds until next run... (Ctrl+C to stop)", sleep_sec)
        try:
            time.sleep(sleep_sec)
        except KeyboardInterrupt:
            logger.info("[watch] stopped by user")
            return 0


if __name__ == "__main__":
    _load_env_local()
    raise SystemExit(main())
