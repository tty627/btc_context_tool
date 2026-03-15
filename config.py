from pathlib import Path

BASE_URL = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
TIMEFRAMES = ("15m", "1h", "4h")

KLINE_LIMIT = 200
DEPTH_LIMIT = 20
REQUEST_TIMEOUT = 10
BINANCE_API_KEY_ENV = "BINANCE_API_KEY"
BINANCE_API_SECRET_ENV = "BINANCE_API_SECRET"

# Proxy (optional). When set, all collector HTTP requests go through it (e.g. to avoid 451).
# Use HTTPS_PROXY or HTTP_PROXY; both are read by the collector.
HTTPS_PROXY_ENV = "HTTPS_PROXY"
HTTP_PROXY_ENV = "HTTP_PROXY"

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.3
OPENAI_MAX_TOKENS = 4096

TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CONTEXT_FILE = OUTPUT_DIR / "btc_context.json"
REPORT_FILE = OUTPUT_DIR / "btc_report.txt"
SUMMARY_FILE = OUTPUT_DIR / "btc_summary.md"
CHART_DIR = OUTPUT_DIR / "charts"
AI_ANALYSIS_FILE = OUTPUT_DIR / "btc_ai_analysis.md"

CHART_BARS = {
    "4h": 120,
    "1h": 120,
    "15m": 120,
    "5m": 120,
}

AGG_TRADES_LIMIT = 3000
LARGE_TRADE_QUANTILE = 0.9

OI_PERIOD = "5m"
OI_LIMIT = 30
LONG_SHORT_PERIOD = "5m"
LONG_SHORT_LIMIT = 30
ORDERBOOK_DYNAMIC_SAMPLES = 30
ORDERBOOK_DYNAMIC_INTERVAL = 0.2
VOLUME_PROFILE_WINDOW = 72
VOLUME_PROFILE_BINS = 24

# Report generation mode:
#   "raw_first"  — main report contains only raw facts (default, recommended for LLM input)
#   "full_debug" — appends derived/hypothesis/score fields after the main report
REPORT_MODE = "raw_first"
