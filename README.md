# BTC Context Tool

Market context report builder for the BTCUSDT perpetual contract using Binance REST APIs. Collects klines, orderbook data, trades, computes indicators/features, optionally renders charts, and can send the result directly to OpenAI for an AI-generated trading plan.

## Features

* Multi-timeframe technical indicators: EMA, MACD, KDJ, RSI, **ATR**, **Bollinger Bands**, **VWAP**
* Open-interest trend with price-OI state labelling
* Global/top-trader long-short crowding ratios
* Multi-snapshot orderbook dynamics (wall pull/add, spoofing risk)
* Volume profile (POC/HVN/LVN), session profiles, anchored VWAP profiles
* Basis (mark-index) and cross-exchange funding spread (Binance/Bybit/OKX)
* Trade flow analysis: CVD, delta, large-trade clusters, absorption zones
* Liquidation heatmap (from force orders or model estimate)
* Optional private read-only account positions
* **Concurrent API requests** — data collection runs in parallel via thread pool
* **Direct OpenAI integration** — `--auto-analyze` sends the prompt to GPT and writes the analysis
* **Watch mode** — `--watch N` re-runs every N seconds for continuous monitoring
* **Historical output** — `--history` timestamps output filenames for post-trade review
* **Structured logging** with Python `logging` module

## Prerequisites

* Python 3.8+ (3.11+ recommended)
* Dependencies: `pip install -r requirements.txt`

| Package | Purpose |
|---------|---------|
| `matplotlib>=3.0` | Chart generation (optional — script works without it) |
| `openai>=1.0` | AI analysis via `--auto-analyze` (optional) |
| `httpx>=0.27` | Faster HTTP with connection pooling (optional — falls back to urllib) |

## Usage

```bash
# install dependencies
python3 -m pip install -r requirements.txt

# basic run (writes to output/ directory)
python3 main.py

# one-click AI trading plan
export OPENAI_API_KEY="sk-..."
python3 main.py --auto-analyze

# use a specific model
python3 main.py --auto-analyze --ai-model gpt-4o-mini

# continuous monitoring every 5 minutes
python3 main.py --watch 300

# continuous monitoring with AI analysis
python3 main.py --watch 300 --auto-analyze

# preserve historical outputs (timestamped filenames)
python3 main.py --history

# disable charts
python3 main.py --no-charts

# tune market microstructure windows
python3 main.py --oi-period 5m --oi-limit 30 \
    --long-short-period 5m --long-short-limit 30 \
    --orderbook-dynamic-samples 3 --orderbook-dynamic-interval 0.6 \
    --volume-profile-window 72 --volume-profile-bins 24

# include Binance futures positions (read-only API)
export BINANCE_API_KEY="your_read_only_key"
export BINANCE_API_SECRET="your_read_only_secret"
python3 main.py

# or store credentials in project-local .env.local
cat > .env.local <<'EOF'
export BINANCE_API_KEY="your_read_only_key"
export BINANCE_API_SECRET="your_read_only_secret"
export OPENAI_API_KEY="sk-..."
EOF
chmod 600 .env.local
python3 main.py --auto-analyze

# custom output paths
python3 main.py --context-file ~/mycontext.json --report-file ~/report.txt
```

## Output Files

| File | Description |
|------|-------------|
| `output/btc_context.json` | Raw JSON market context |
| `output/btc_report.txt` | Structured prompt for AI trading assistant |
| `output/btc_ai_analysis.md` | AI-generated trading plan (with `--auto-analyze`) |
| `output/btc_summary.md` | Human-readable summary (with `--include-summary`) |
| `output/charts/*.png` | Kline charts (if matplotlib installed) |

With `--history` or `--watch`, output filenames include UTC timestamps, e.g. `btc_context_20260311_143000.json`.

## Project Structure

```
├── main.py                   # Entry point and orchestration
├── config.py                 # Central configuration
├── collectors/
│   └── binance_collector.py  # Binance/Bybit/OKX REST API client
├── indicators/
│   └── engine.py             # EMA, MACD, KDJ, RSI, ATR, Bollinger, VWAP
├── features/                 # Feature extraction (split into domain modules)
│   ├── _base.py              # Shared utilities
│   ├── technical.py          # Trend/momentum classification
│   ├── orderbook.py          # Orderbook features and dynamics
│   ├── volume.py             # Volume profile, session profiles
│   ├── derivatives.py        # OI trend, long/short ratio, basis, funding
│   ├── trade_flow.py         # CVD, delta, large trades, absorption
│   ├── liquidation.py        # Liquidation heatmap
│   ├── session.py            # Session context, funding countdown
│   ├── deployment.py         # Deployment assessment scoring
│   └── extractor.py          # Facade composing all mixins
├── context/
│   └── builder.py            # Assembles final context dict
├── reports/
│   ├── prompt_generator.py   # Builds Chinese prompt for AI
│   └── summary_table.py      # Markdown summary table
├── advisor/
│   └── ai_advisor.py         # OpenAI API integration
└── charts/
    └── kline_chart.py        # Matplotlib chart generation
```

## Private API Note

When `BINANCE_API_KEY` and `BINANCE_API_SECRET` are present, account positions
are collected automatically. Use `--include-account` to force on or `--no-account` to disable.
Use a key with **read-only permissions** and never hardcode secrets in source files.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use the helper script: `source setup_env.sh`

## Notes

* Public market data endpoints do not require an API key.
* All API calls run concurrently for faster data collection.
* The prompt generator summarizes time-series data to reduce token usage.
* ATR is used to provide quantitative stop-loss distance suggestions.
