import argparse
import concurrent.futures
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

logger = logging.getLogger("btc_context")

try:
    from .collectors import BinanceFuturesCollector
    from .config import (
        AGG_TRADES_LIMIT, AI_ANALYSIS_FILE, BINANCE_API_KEY_ENV, BINANCE_API_SECRET_ENV,
        CHART_BARS, CHART_DIR, CONTEXT_FILE, DEPTH_LIMIT, KLINE_LIMIT, LARGE_TRADE_QUANTILE,
        LONG_SHORT_LIMIT, LONG_SHORT_PERIOD, OI_LIMIT, OI_PERIOD, OPENAI_API_KEY_ENV,
        OPENAI_MODEL, ORDERBOOK_DYNAMIC_INTERVAL, ORDERBOOK_DYNAMIC_SAMPLES, REPORT_FILE,
        REPORT_MODE, SUMMARY_FILE, SYMBOL, TELEGRAM_BOT_TOKEN_ENV, TELEGRAM_CHAT_ID_ENV,
        TIMEFRAMES, VOLUME_PROFILE_BINS, VOLUME_PROFILE_WINDOW,
    )
    from .context import MarketContextBuilder
    from .features import FeatureExtractor
    from .indicators import IndicatorEngine
    from .reports import PromptGenerator, SummaryTableGenerator
except ImportError:
    from collectors import BinanceFuturesCollector
    from config import (
        AGG_TRADES_LIMIT, AI_ANALYSIS_FILE, BINANCE_API_KEY_ENV, BINANCE_API_SECRET_ENV,
        CHART_BARS, CHART_DIR, CONTEXT_FILE, DEPTH_LIMIT, KLINE_LIMIT, LARGE_TRADE_QUANTILE,
        LONG_SHORT_LIMIT, LONG_SHORT_PERIOD, OI_LIMIT, OI_PERIOD, OPENAI_API_KEY_ENV,
        OPENAI_MODEL, ORDERBOOK_DYNAMIC_INTERVAL, ORDERBOOK_DYNAMIC_SAMPLES, REPORT_FILE,
        REPORT_MODE, SUMMARY_FILE, SYMBOL, TELEGRAM_BOT_TOKEN_ENV, TELEGRAM_CHAT_ID_ENV,
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


def _make_timestamped_path(base_path: Path) -> Path:
    """Insert a UTC timestamp before the file extension: foo.json -> foo_20260311_143000.json"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return base_path.with_name(f"{base_path.stem}_{ts}{base_path.suffix}")


def _resolve_output_files(
    context_file: Optional[str],
    report_file: Optional[str],
    chart_dir: Optional[str],
    history: bool = False,
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

    if history:
        context_path = _make_timestamped_path(context_path)
        report_path = _make_timestamped_path(report_path)

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

def build_market_context(
    symbol: str,
    timeframes: Sequence[str],
    kline_limit: int,
    depth_limit: int,
    agg_trades_limit: int,
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
    engine = IndicatorEngine()
    feature_extractor = FeatureExtractor()
    context_builder = MarketContextBuilder()

    # ── Phase 1: fire all independent network calls in parallel ──────────
    all_kline_intervals = list(dict.fromkeys(
        list(timeframes) + [tf for tf in ("5m", "15m", "1h") if tf not in timeframes]
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
        fut_agg_trades = pool.submit(collector.get_agg_trades, symbol, agg_trades_limit)
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
    )

    if include_charts and chart_generator_cls is not None:
        chart_generator = chart_generator_cls(output_dir=charts_path, bars_by_timeframe=CHART_BARS)
        chart_files = chart_generator.generate(symbol, klines_by_timeframe, context=context)
        spot_perp_path = charts_path / f"{symbol}_spot_perp.png"
        sp_result = chart_generator.generate_spot_perp_chart(
            symbol=symbol,
            perp_klines=klines_by_timeframe.get("1h", []),
            spot_klines=spot_klines_1h,
            output_path=spot_perp_path,
        )
        if sp_result:
            chart_files["spot_perp"] = sp_result
        context["chart_files"] = chart_files

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

    alternate_prompt_path = report_path.with_name("btc_user_prompt.txt")
    if alternate_prompt_path == report_path or not alternate_prompt_path.exists():
        return
    try:
        alternate_prompt_path.unlink()
    except OSError:
        return


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

    prompt = PromptGenerator().build(ai_context, report_mode=report_mode)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(prompt)
    _cleanup_alternate_prompt_output(report_path)

    cn_chars = sum(1 for ch in prompt if "\u4e00" <= ch <= "\u9fff")
    en_chars = len(prompt) - cn_chars
    est_tokens = int(cn_chars / 3.5 + en_chars / 4)
    logger.info("prompt size: %d chars, ~%d tokens (estimate)", len(prompt), est_tokens)

    return ai_context

def write_summary_output(context: Dict, context_path: Path) -> Path:
    summary_path = SUMMARY_FILE
    if summary_path.parent != context_path.parent:
        summary_path = context_path.parent / summary_path.name

    summary = SummaryTableGenerator().build(context)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary)
    return summary_path


def _load_ai_advisor():
    try:
        try:
            from .advisor import AIAdvisor
        except ImportError:
            from advisor import AIAdvisor
        return AIAdvisor
    except ImportError:
        return None


def _run_ai_analysis(api_key: str, model: str, prompt_text: str) -> str:
    AIAdvisor = _load_ai_advisor()
    if AIAdvisor is None:
        raise RuntimeError("advisor module could not be loaded")
    advisor = AIAdvisor(api_key=api_key, model=model)
    token_est = advisor.estimate_tokens(prompt_text)
    print(f"estimated prompt tokens: ~{token_est}")
    return advisor.analyze(prompt_text)


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
        "--history",
        action="store_true",
        help="Append UTC timestamp to output filenames to preserve historical runs",
    )
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
            "Send the generated prompt to OpenAI and write the AI analysis to file. "
            f"Requires env var {OPENAI_API_KEY_ENV}."
        ),
    )
    parser.add_argument(
        "--ai-model",
        default=OPENAI_MODEL,
        help=f"OpenAI model to use for --auto-analyze (default: {OPENAI_MODEL})",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Re-run every N seconds (0 = single run). E.g. --watch 300 for 5-minute intervals.",
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
    return parser.parse_args(argv)


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

    use_history = args.history or args.watch > 0
    context_path, report_path, charts_path = _resolve_output_files(
        args.context_file,
        args.report_file,
        args.chart_dir,
        history=use_history,
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
    ai_context = write_outputs(context, context_path, report_path, report_mode=args.report_mode)
    if args.include_summary:
        summary_path = write_summary_output(context, context_path)
        with context_path.open("w", encoding="utf-8") as f:
            ai_context["summary_files"] = {"overview": str(summary_path.resolve())}
            json.dump(ai_context, f, ensure_ascii=False, indent=2)
    logger.info("context written to %s", context_path)
    logger.info("report written to %s", report_path)
    if args.include_summary:
        logger.info("overview summary written to %s", summary_path)
    if context.get("chart_files"):
        logger.info("charts:")
        for timeframe, file_path in context["chart_files"].items():
            logger.info("  %s: %s", timeframe, file_path)

    analysis = ""
    if args.auto_analyze:
        openai_key = _sanitize_env_credential(os.getenv(OPENAI_API_KEY_ENV))
        if not openai_key:
            logger.error("--auto-analyze requires %s to be set", OPENAI_API_KEY_ENV)
            return 1
        logger.info("sending prompt to %s for analysis...", args.ai_model)
        prompt_text = report_path.read_text(encoding="utf-8")
        analysis = _run_ai_analysis(openai_key, args.ai_model, prompt_text)
        analysis_path = _resolve_analysis_path(context_path)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(analysis, encoding="utf-8")
        logger.info("AI analysis written to %s", analysis_path)
        print("\n" + "=" * 60)
        print(analysis)
        print("=" * 60)

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

        logger.info("[watch] sleeping %ds until next run... (Ctrl+C to stop)", watch_interval)
        try:
            time.sleep(watch_interval)
        except KeyboardInterrupt:
            logger.info("[watch] stopped by user")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
