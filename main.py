import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

if __package__:
    from .collectors import BinanceFuturesCollector
    from .config import (
        AGG_TRADES_LIMIT,
        BINANCE_API_KEY_ENV,
        BINANCE_API_SECRET_ENV,
        CHART_BARS,
        CHART_DIR,
        CONTEXT_FILE,
        DEPTH_LIMIT,
        KLINE_LIMIT,
        LARGE_TRADE_QUANTILE,
        LONG_SHORT_LIMIT,
        LONG_SHORT_PERIOD,
        OI_LIMIT,
        OI_PERIOD,
        ORDERBOOK_DYNAMIC_INTERVAL,
        ORDERBOOK_DYNAMIC_SAMPLES,
        REPORT_FILE,
        SUMMARY_FILE,
        SYMBOL,
        TIMEFRAMES,
        VOLUME_PROFILE_BINS,
        VOLUME_PROFILE_WINDOW,
    )
    from .context import MarketContextBuilder
    from .features import FeatureExtractor
    from .indicators import IndicatorEngine
    from .reports import PromptGenerator, SummaryTableGenerator
else:
    from collectors import BinanceFuturesCollector
    from config import (
        AGG_TRADES_LIMIT,
        BINANCE_API_KEY_ENV,
        BINANCE_API_SECRET_ENV,
        CHART_BARS,
        CHART_DIR,
        CONTEXT_FILE,
        DEPTH_LIMIT,
        KLINE_LIMIT,
        LARGE_TRADE_QUANTILE,
        LONG_SHORT_LIMIT,
        LONG_SHORT_PERIOD,
        OI_LIMIT,
        OI_PERIOD,
        ORDERBOOK_DYNAMIC_INTERVAL,
        ORDERBOOK_DYNAMIC_SAMPLES,
        REPORT_FILE,
        SUMMARY_FILE,
        SYMBOL,
        TIMEFRAMES,
        VOLUME_PROFILE_BINS,
        VOLUME_PROFILE_WINDOW,
    )
    from context import MarketContextBuilder
    from features import FeatureExtractor
    from indicators import IndicatorEngine
    from reports import PromptGenerator, SummaryTableGenerator


def _load_chart_generator():
    try:
        if __package__:
            from .charts import KlineChartGenerator
        else:
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
) -> Dict:
    collector = BinanceFuturesCollector(api_key=api_key, api_secret=api_secret)
    engine = IndicatorEngine()
    feature_extractor = FeatureExtractor()
    context_builder = MarketContextBuilder()

    klines_by_timeframe = collector.get_multi_klines(symbol, list(timeframes), kline_limit)
    support_timeframes = ("5m", "15m", "1h")
    missing_support = [timeframe for timeframe in support_timeframes if timeframe not in klines_by_timeframe]
    if missing_support:
        klines_by_timeframe.update(
            collector.get_multi_klines(
                symbol,
                missing_support,
                max(kline_limit, 120),
            )
        )

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

    if not include_account:
        account_positions = {
            "available": False,
            "reason": "disabled_by_flag",
            "active_positions_count": 0,
            "active_positions": [],
            "symbol_position": None,
        }
    else:
        account_positions = collector.get_account_positions(symbol=symbol)

    orderbook = collector.get_orderbook(symbol, depth_limit)
    orderbook_features = feature_extractor.extract_orderbook_features(orderbook)
    agg_trades = collector.get_agg_trades(symbol, agg_trades_limit)
    trade_flow = feature_extractor.extract_trade_flow_features(agg_trades, large_trade_quantile)
    orderbook_snapshots = collector.get_orderbook_snapshots(
        symbol=symbol,
        limit=depth_limit,
        samples=orderbook_dynamic_samples,
        interval_seconds=orderbook_dynamic_interval,
    )
    if not orderbook_snapshots:
        orderbook_snapshots = [orderbook]
    orderbook_dynamics = feature_extractor.extract_orderbook_dynamics(orderbook_snapshots, trades=agg_trades)

    open_interest = collector.get_open_interest(symbol)
    volume_change = feature_extractor.extract_volume_change(klines_by_timeframe)
    oi_histories = {
        period: collector.get_open_interest_hist(symbol, period=period, limit=oi_limit)
        for period in ("5m", "15m", "1h")
    }
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

    global_long_short = collector.get_global_long_short_ratio(
        symbol=symbol,
        period=long_short_period,
        limit=long_short_limit,
    )
    top_trader_ratio = collector.get_top_trader_long_short_ratio(
        symbol=symbol,
        period=long_short_period,
        limit=long_short_limit,
    )
    long_short_ratio = feature_extractor.extract_long_short_ratio(
        global_account_ratio=global_long_short,
        top_trader_ratio=top_trader_ratio,
    )

    funding = collector.get_funding(symbol)
    basis = feature_extractor.extract_basis(funding)
    cross_exchange_funding = collector.get_cross_exchange_funding(
        symbol=symbol,
        binance_funding_rate=funding.get("funding_rate"),
    )
    funding_spread = feature_extractor.extract_cross_exchange_funding_spread(cross_exchange_funding)
    options_iv = feature_extractor.extract_options_iv_placeholder()

    ticker_24h = collector.get_ticker_24h(symbol)
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
    force_orders = collector.get_force_orders(symbol, limit=100)
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
    context["session_context"] = session_context
    context["deployment_context"] = deployment_context
    context["summary_files"] = {}

    if include_charts and chart_generator_cls is not None:
        chart_generator = chart_generator_cls(output_dir=charts_path, bars_by_timeframe=CHART_BARS)
        context["chart_files"] = chart_generator.generate(symbol, klines_by_timeframe, context=context)

    return context


def _sanitize_context_for_ai_output(context: Dict) -> Dict:
    sanitized = copy.deepcopy(context)
    sanitized.pop("deployment_context", None)
    sanitized.pop("market_structure", None)
    sanitized["summary_files"] = {}

    for metrics in sanitized.get("timeframes", {}).values():
        if not isinstance(metrics, dict):
            continue
        metrics.pop("features", None)
        rsi = metrics.get("rsi")
        if isinstance(rsi, dict):
            rsi.pop("state", None)
            rsi.pop("divergence", None)

    orderbook_dynamics = sanitized.get("orderbook_dynamics", {})
    if isinstance(orderbook_dynamics, dict):
        orderbook_dynamics.pop("spoofing_risk", None)
        orderbook_dynamics.pop("wall_behavior", None)

    open_interest_trend = sanitized.get("open_interest_trend", {})
    if isinstance(open_interest_trend, dict):
        for key in (
            "trend",
            "latest_state",
            "latest_interpretation",
            "composite_signal",
            "volume_oi_cvd_state",
        ):
            open_interest_trend.pop(key, None)
        for period_data in open_interest_trend.get("periods", {}).values():
            if not isinstance(period_data, dict):
                continue
            for key in ("trend", "latest_state", "latest_interpretation"):
                period_data.pop(key, None)
            for point in period_data.get("series", []):
                if not isinstance(point, dict):
                    continue
                point.pop("state", None)
                point.pop("interpretation", None)

    long_short_ratio = sanitized.get("long_short_ratio", {})
    if isinstance(long_short_ratio, dict):
        long_short_ratio.pop("overall_crowding", None)
        for subkey in ("global_account", "top_trader_position"):
            ratio_block = long_short_ratio.get(subkey)
            if isinstance(ratio_block, dict):
                ratio_block.pop("crowding", None)

    basis = sanitized.get("basis", {})
    if isinstance(basis, dict):
        basis.pop("structure", None)

    funding_spread = sanitized.get("funding_spread", {})
    if isinstance(funding_spread, dict):
        funding_spread.pop("signal", None)

    trade_flow = sanitized.get("trade_flow", {})
    if isinstance(trade_flow, dict):
        trade_flow.pop("large_trade_direction", None)
        for window in trade_flow.get("windows", {}).values():
            if isinstance(window, dict):
                window.pop("large_trade_direction", None)
        for layer in trade_flow.get("aggressor_layers", {}).values():
            if isinstance(layer, dict):
                layer.pop("large_trade_direction", None)
        for cluster in trade_flow.get("large_trade_clusters", []):
            if isinstance(cluster, dict):
                cluster.pop("dominant_side", None)

    liquidation_heatmap = sanitized.get("liquidation_heatmap", {})
    if isinstance(liquidation_heatmap, dict):
        liquidation_heatmap.pop("confidence", None)
        liquidation_heatmap.pop("model_assumptions", None)
        for zone in liquidation_heatmap.get("zones", []):
            if not isinstance(zone, dict):
                continue
            zone.pop("confidence", None)
            zone.pop("assumption", None)
            zone.pop("estimated_pressure", None)

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


def write_outputs(context: Dict, context_path: Path, report_path: Path) -> Dict:
    context_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ai_context = _sanitize_context_for_ai_output(context)

    with context_path.open("w", encoding="utf-8") as f:
        json.dump(ai_context, f, ensure_ascii=False, indent=2)

    prompt = PromptGenerator().build(ai_context)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(prompt)
    _cleanup_alternate_prompt_output(report_path)
    return ai_context

def write_summary_output(context: Dict, context_path: Path) -> Path:
    summary_path = SUMMARY_FILE
    if summary_path.parent != context_path.parent:
        summary_path = context_path.parent / summary_path.name

    summary = SummaryTableGenerator().build(context)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary)
    return summary_path


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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    # indicate which Python interpreter is running; helps avoid mismatched envs
    print(f"[info] running under Python executable: {sys.executable}")
    args = parse_args(argv if argv is not None else sys.argv[1:])
    _load_local_env_file(Path(__file__).resolve().parent / ".env.local")

    try:
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
            print(
                f"warning: detected placeholder {BINANCE_API_KEY_ENV}/{BINANCE_API_SECRET_ENV}; "
                "account position collection disabled",
                file=sys.stderr,
            )
            api_key = None
            api_secret = None
        has_api_credentials = bool(api_key and api_secret)
        include_account = not args.no_account and (args.include_account or has_api_credentials)
        if include_account and (not api_key or not api_secret):
            print(
                f"warning: account collection enabled but usable {BINANCE_API_KEY_ENV}/{BINANCE_API_SECRET_ENV} "
                "not found; account position data will be unavailable",
                file=sys.stderr,
            )

        chart_generator_cls = None
        chart_warning = None
        include_charts = not args.no_charts
        if include_charts:
            chart_generator_cls, chart_warning = _load_chart_generator()
            if chart_generator_cls is None:
                include_charts = False
                print(f"warning: {chart_warning}", file=sys.stderr)

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
        )
        ai_context = write_outputs(context, context_path, report_path)
        if args.include_summary:
            summary_path = write_summary_output(context, context_path)
            with context_path.open("w", encoding="utf-8") as f:
                ai_context["summary_files"] = {"overview": str(summary_path.resolve())}
                json.dump(ai_context, f, ensure_ascii=False, indent=2)
        print(f"context written to {context_path}")
        print(f"report written to {report_path}")
        if args.include_summary:
            print(f"overview summary written to {summary_path}")
        if context.get("chart_files"):
            print("charts:")
            for timeframe, file_path in context["chart_files"].items():
                print(f"  {timeframe}: {file_path}")
        return 0
    except Exception as exc:
        print(f"failed to build market context: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
