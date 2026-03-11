# BTC Context Tool

Simple script for building a market context report for the BTCUSDT perpetual contract using Binance's REST APIs.  It collects klines, orderbook data, trades, etc., computes indicators/features, and optionally renders kline charts.

Latest context enrichments include:
* Open-interest trend (instead of only OI absolute value)
* Global/top-trader long-short crowding ratios
* Multi-snapshot orderbook dynamics (wall pull/add, spoofing risk)
* Volume profile (POC/HVN/LVN)
* Basis (mark-index) and cross-exchange funding spread
* RSI + basic RSI divergence signal
* Optional private read-only account positions

## Prerequisites

* Python 3.8+ (3.11 tested).  You should run the script using the *same* interpreter
  that you use to install dependencies – otherwise you may encounter the
  familiar warning about `matplotlib` being missing.
* Dependencies are listed in `requirements.txt`; install them in your chosen
  environment with `pip install -r requirements.txt`.

> The only non‑stdlib requirement is `matplotlib` which is used for chart
> generation.  If it is not installed the script will still run but charts will
> be disabled and you'll see a warning indicating which Python executable was
> used.

## Usage

```bash
# install dependencies
python3 -m pip install -r requirements.txt

# run with defaults (writes to output/ directory)
python3 main.py

# disable charts explicitly
python3 main.py --no-charts

# tune newly added market microstructure windows
python3 main.py --oi-period 5m --oi-limit 30 \
    --long-short-period 5m --long-short-limit 30 \
    --orderbook-dynamic-samples 3 --orderbook-dynamic-interval 0.6 \
    --volume-profile-window 72 --volume-profile-bins 24

# include your Binance futures positions (read-only API, auto-enabled when env vars exist)
export BINANCE_API_KEY="your_read_only_key"
export BINANCE_API_SECRET="your_read_only_secret"
python3 main.py

# or store credentials in project-local .env.local (auto-loaded by main.py)
cat > .env.local <<'EOF'
export BINANCE_API_KEY="your_read_only_key"
export BINANCE_API_SECRET="your_read_only_secret"
EOF
chmod 600 .env.local
python3 main.py

# explicitly disable account collection if needed
python3 main.py --no-account

# specify custom paths
python3 main.py --symbol BTCUSDT --timeframes 15m,1h,4h \
    --kline-limit 200 --depth-limit 20 \
    --context-file ~/mycontext.json --report-file ~/report.txt
```

Output files are produced under `output/` by default:

* `btc_context.json` – raw JSON market context
* `btc_report.txt` – neutral user prompt that constrains answer format without injecting directional bias
* `output/charts/...` – generated PNG charts (if `matplotlib` is installed)

### Private API note

When `BINANCE_API_KEY` and `BINANCE_API_SECRET` are present, account positions
are collected automatically from `/fapi/v2/positionRisk`.
You can force on with `--include-account` or disable with `--no-account`.
Use a key with read-only permissions and never hardcode secrets in source files.
`main.py` also auto-loads `.env.local` in the project directory (without
overriding already-exported shell variables).

Troubleshooting (account_positions `reason`):

* `code=-2014`:
  API key format invalid. Check whether key/secret were pasted with extra spaces
  or quotes, and make sure you are using Binance Futures mainnet credentials.
* `code=-2015`:
  Key/permissions/IP invalid. Ensure Futures read permission is enabled and IP
  whitelist settings match your current network.
* `missing_api_credentials`:
  Export both `BINANCE_API_KEY` and `BINANCE_API_SECRET`, or run with
  `--no-account`.

### Environment notes

To avoid confusion between multiple Pythons (system, Homebrew, conda, venv,
etc.) you can create an isolated environment inside the repo.  A helper
script is provided:

```bash
# run once, or whenever you want to refresh the environment
source setup_env.sh

# afterwards just activate
source .venv/bin/activate
```

Alternatively, manually:

```bash
python -m venv .venv      # or use conda: conda create -n btccontext python
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Once activated the same `python` binary will be used for every subsequent
command.  The script itself will print the interpreter on startup so you can
verify which one is running:

```text
$ python main.py
[info] running under Python executable: /path/to/venv/bin/python
```

Alternatively, activate your conda environment before installation/running:

```bash
conda activate base   # or the environment of your choice
python main.py
```

The important part is to be consistent – installing with one interpreter and
running with another is what previously caused the exit code 1 and the missing
`matplotlib` warning.

## Notes

* Public market data endpoints do not require an API key.
* Private account data is auto-enabled when read-only API credentials are present.
* You can import and use the library modules directly in your own code if desired.
