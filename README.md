# BTC Context Tool

基于 Binance REST API 的 BTCUSDT 永续合约市场数据面板工具。采集 K 线、盘口、成交流、衍生品等多维数据，计算技术指标与微观结构特征，可选生成图表，并支持直接调用 OpenAI 进行 AI 行情分析。

> **设计理念**：报告采用 `raw_first` 模式，只输出原始事实和数据质量信息，不在主报告中预设交易方向，最大化下游 LLM 的自主判读空间，消除锚定效应。

## 功能特性

- **多周期技术指标**：EMA、MACD、KDJ、RSI、ATR、布林带、VWAP
- **成交流分析**：CVD、Delta、大单聚集区、吸收区、价格分层 Delta（Footprint 风格）
- **盘口动态**：多快照盘口分析（挂单撤单速率、墙撤离/吸收/扫单事件）
- **量价结构**：成交量分布（POC/HVN/LVN）、session 分时 VP、锚定 AVWAP
- **衍生品数据**：OI 趋势（5m/15m/1h 分段）、全局/顶级交易者多空比、资金费率
- **跨所数据**：Binance/Bybit/OKX 的 OI 及资金费率对比
- **现货 vs 合约**：现货 CVD、basis bps、现货-合约价差
- **K 线图表**：4h / 1h / 15m / 5m Delta 面板图 + 现货合约对比图
- **并发采集**：所有 API 请求通过线程池并行执行
- **AI 分析集成**：`--auto-analyze` 将报告发送至 GPT，输出行情分析
- **持续监控**：`--watch N` 每 N 秒重新运行一次
- **历史存档**：`--history` 自动为输出文件加时间戳

## 环境要求

- Python 3.8+（推荐 3.11+）
- 依赖安装：`pip install -r requirements.txt`

| 依赖包 | 用途 |
|--------|------|
| `matplotlib>=3.0` | 图表生成（可选，缺失时跳过图表） |
| `openai>=1.0` | AI 分析，`--auto-analyze` 时需要（可选） |
| `httpx>=0.27` | 更快的 HTTP 连接池（可选，缺失时回退到 urllib） |

## 快速开始

```bash
# 安装依赖
python3 -m pip install -r requirements.txt

# 基础运行（输出到 output/ 目录）
python3 main.py

# 一键 AI 行情分析
export OPENAI_API_KEY="sk-..."
python3 main.py --auto-analyze

# 指定模型
python3 main.py --auto-analyze --ai-model gpt-4o-mini

# 持续监控（每 5 分钟）
python3 main.py --watch 300

# 持续监控 + AI 分析
python3 main.py --watch 300 --auto-analyze

# 保留历史输出（文件名带时间戳）
python3 main.py --history

# 禁用图表
python3 main.py --no-charts

# raw_first 模式（默认，仅原始数据，无方向性先验）
python3 main.py --report-mode raw_first

# full_debug 模式（主报告 + 附录：信号评分、部署评估等派生字段）
python3 main.py --report-mode full_debug

# 调整微观结构采集参数
python3 main.py --oi-period 5m --oi-limit 30 \
    --long-short-period 5m --long-short-limit 30 \
    --orderbook-dynamic-samples 30 --orderbook-dynamic-interval 0.2 \
    --volume-profile-window 72 --volume-profile-bins 24

# 包含 Binance 合约持仓（只读 API）
export BINANCE_API_KEY="your_read_only_key"
export BINANCE_API_SECRET="your_read_only_secret"
python3 main.py

# 将凭证存入本地 .env.local（不入版本控制）
cat > .env.local <<'EOF'
export BINANCE_API_KEY="your_read_only_key"
export BINANCE_API_SECRET="your_read_only_secret"
export OPENAI_API_KEY="sk-..."
EOF
chmod 600 .env.local
source .env.local && python3 main.py --auto-analyze

# 自定义输出路径
python3 main.py --context-file ~/mycontext.json --report-file ~/report.txt
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `output/btc_context.json` | 原始市场上下文 JSON（完整数据，含所有派生字段） |
| `output/btc_report.txt` | 结构化市场数据面板（发送给 AI 的主报告） |
| `output/btc_ai_analysis.md` | AI 行情分析（使用 `--auto-analyze` 时生成） |
| `output/btc_summary.md` | 人类可读摘要表（使用 `--include-summary` 时生成） |
| `output/charts/*.png` | K 线图表（需安装 matplotlib） |

使用 `--history` 或 `--watch` 时，文件名会带 UTC 时间戳，例如 `btc_context_20260312_071309.json`。

## 报告模式说明

`btc_report.txt` 支持两种生成模式，通过 `config.py` 中的 `REPORT_MODE` 或 `--report-mode` 参数控制：

| 模式 | 说明 |
|------|------|
| `raw_first`（默认） | 主报告只含原始事实：数据质量、指标数值、盘口、衍生品、持仓。不含任何方向性标签、信号评分或交易计划 |
| `full_debug` | 在主报告后追加附录，包含派生字段：信号评分、部署偏向、state_tags、plan_zones、市场结构标签等 |

> 主报告中被移除的锚定字段包括：`primary_bias`、`transition_state`、`composite_score`、`state_tags`、`plan_zones`、`market_structure bearish/bullish 标签`、`OI interpretation`、`L/S crowding`、`spot_perp interpretation` 等。这些字段仍保存在 `btc_context.json` 中，仅在 `raw_first` 模式下不进入主报告。

## 项目结构

```
├── main.py                   # 入口与整体调度
├── config.py                 # 全局配置（含 REPORT_MODE）
├── collectors/
│   └── binance_collector.py  # Binance/Bybit/OKX REST API 客户端
├── indicators/
│   └── engine.py             # EMA、MACD、KDJ、RSI、ATR、布林带、VWAP
├── features/                 # 特征提取（按领域拆分）
│   ├── _base.py              # 公共工具
│   ├── technical.py          # 趋势/动量分类
│   ├── orderbook.py          # 盘口特征与动态
│   ├── volume.py             # 量价结构、session 分析
│   ├── derivatives.py        # OI、多空比、basis、资金费率
│   ├── trade_flow.py         # CVD、Delta、大单、Footprint
│   ├── spot_perp.py          # 现货 vs 合约对比
│   ├── liquidation.py        # 清算热力图（模型估算）
│   ├── session.py            # Session 上下文、费率倒计时
│   ├── deployment.py         # 部署评估评分（附录用）
│   └── extractor.py          # 聚合所有 mixin 的外观类
├── context/
│   └── builder.py            # 组装最终 context dict
├── reports/
│   ├── prompt_generator.py   # raw_first 数据面板生成器
│   └── summary_table.py      # Markdown 摘要表
├── advisor/
│   └── ai_advisor.py         # OpenAI API 集成
└── charts/
    └── kline_chart.py        # matplotlib 图表生成
```

## API 密钥说明

- **公开市场数据**端点无需 API Key。
- 当环境变量中存在 `BINANCE_API_KEY` 和 `BINANCE_API_SECRET` 时，自动采集持仓数据。
- 使用 `--include-account` 强制开启，`--no-account` 强制关闭。
- 请使用**只读权限**的 API Key，不要在源码中硬编码密钥。

## 环境初始化

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

或使用辅助脚本：`source setup_env.sh`

## 注意事项

- 所有 API 请求并发执行，采集速度快，通常在 5~10 秒内完成。
- `btc_context.json` 保存完整原始数据，可供自定义后处理或调试。
- `btc_report.txt` 默认为纯数据面板，适合直接作为 LLM prompt 使用。
- 30m/1h K 线 Delta 需要 Binance 返回 `taker_buy_base` 字段；若不可用，报告中会标注 `kline_flow: unavailable`。
