# BTC Context Tool

基于 Binance REST API 的 BTCUSDT 永续合约市场数据面板工具。采集 K 线、盘口、成交流、衍生品等多维数据，计算技术指标与微观结构特征，可选生成图表，并支持直接调用 OpenAI 进行 AI 行情分析。

> **设计理念**：默认采用 `raw_first` + **双阶段 AI**。第一阶段 `research` 读取原始事实、图表和高价值原始附录，生成交易员 handoff；第二阶段 `decision` 基于 handoff + 关键原始证据给出最终开仓/持仓处理。目标是**高赔率优先、允许低胜率、错了快砍**，而不是追求分析报告的形式完整。

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
- **双阶段 AI 决策**：`research -> decision` 两段式调用；`research` 生成结构化 handoff，`decision` 结合 handoff、关键原始附录和关键图表做最终拍板
- **高赔率决策目标**：优先找“位置干净 + 失效清晰 + 盈亏比不对称”的机会；脏中位 / RR 一般直接 `WAIT`；允许小亏损，不鼓励扛单或希望式持仓
- **AI 分析集成**：默认 **OpenAI + gpt-5**；多模态传 K 线 PNG。`OPENAI_REASONING_EFFORT` 可调推理档位；`research` 默认带全量关键图，`decision` 默认带关键高周期图。若主模型不可用会自动回退到 `gpt-5-mini -> gpt-4o -> gpt-4o-mini`
- **AI 历史上下文**：每次 AI 分析后记录方向/入场/止损，下次运行时将近期预测与实际价格结果注入 prompt 末尾，帮助 AI 识别并纠正系统性方向偏差（`output/.analysis_history.json`）
- **交易风控监控**：`--monitor` 模式每轮自动检查 4 条交易纪律规则，触发即微信推送提醒（需配置 `BINANCE_API_KEY` + `PUSHPLUS_TOKEN`）
- **ETF 资金流**：BTC 现货 ETF 日净流量（SoSoValue 主源 / coinglass 备用）+ 恐慌贪婪指数
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

# 一键 AI 行情分析（双阶段：research -> decision）
export OPENAI_API_KEY="sk-..."
python3 main.py --auto-analyze

# 默认 openai + gpt-5；更快但通常更弱的示例：
python3 main.py --auto-analyze --ai-model gpt-4o-mini

# 只传文字、不传 K 线图
python3 main.py --auto-analyze --no-ai-charts

# 持续监控（每 5 分钟）
python3 main.py --watch 300
# 简写：只写 --loop 即每 300 秒一轮；--loop 600 即 10 分钟
python3 main.py --loop
python3 main.py --loop 600

# 一键循环监控（AI + 智能省调用 + PushPlus + 缓存 30s，默认每 10 分钟）
python3 main.py --monitor
python3 main.py --monitor --loop 300   # 改回 5 分钟一轮

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

# 通过代理访问 Binance（避免 451 地域限制）
export HTTPS_PROXY=http://127.0.0.1:10808
python3 main.py
# 若仍 451：多半是代理出口在受限地区（如美国）。在 v2rayN 中换为非美节点（如香港/新加坡/日本）再试。
# 若代理是 SOCKS5：export HTTPS_PROXY=socks5://127.0.0.1:10808，并安装 pip install httpx-socks

# 自定义输出路径
python3 main.py --context-file ~/mycontext.json --report-file ~/report.txt
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `output/btc_context.json` | 市场上下文 JSON（含派生字段与 `raw_appendix`） |
| `output/btc_prompt.txt` | 兼容/手工使用的单文件数据面板 |
| `output/btc_system.txt` | 兼容/手工使用的系统提示；在自动分析中会被覆盖为当前 decision system |
| `output/btc_research_system.txt` | `research` 阶段 system prompt |
| `output/btc_research_prompt.txt` | `research` 阶段 user prompt（含 `RAW_APPENDIX`） |
| `output/btc_ai_research.md` | `research` 阶段原始输出 |
| `output/btc_research_handoff.json` | `research` 阶段交给 `decision` 的结构化 handoff |
| `output/btc_decision_system.txt` | `decision` 阶段 system prompt |
| `output/btc_decision_prompt.txt` | `decision` 阶段 user prompt |
| `output/btc_ai_decision.md` | `decision` 阶段原始输出 |
| `output/btc_ai_analysis.md` | 最终 AI 结果的兼容文件（与 `btc_ai_decision.md` 同步） |
| `output/btc_summary.md` | 人类可读摘要表（使用 `--include-summary` 时生成） |
| `output/charts/*.png` | K 线图表（需安装 matplotlib） |

使用 `--history` 或 `--watch` 时，文件名会带 UTC 时间戳，例如 `btc_context_20260312_071309.json`。

**`--smart` 调度（与 `--monitor` 联动）**：连续 **3** 次因「无实质变化」跳过 AI 后，下一轮 **强制** 调用 AI；若本轮分析出现 **可执行挂单方案**（非 wait），则改为约 **每 4 分钟** 一轮 AI，直到结论回到 **wait/无方案**。状态文件：`output/.smart_ai_scheduler.json`。

**默认覆盖**：不带 `--history` 时，同一路径下的 `btc_context.json`、`btc_prompt.txt`、`btc_research_*`、`btc_decision_*`、`btc_ai_analysis.md`、`charts/*.png` 等每次运行会被**直接覆盖**为最新一轮快照。

## AI 决策流程

```mermaid
flowchart TD
  marketData[MarketContext] --> rawAppendix[RawAppendix]
  marketData --> basePanel[RawFirstPanel]
  rawAppendix --> researchPrompt[ResearchPrompt]
  basePanel --> researchPrompt
  charts[KeyCharts] --> researchPrompt
  researchPrompt --> researchModel[ResearchStage]
  researchModel --> handoff[ResearchHandoffJson]
  basePanel --> decisionPrompt[DecisionPrompt]
  rawAppendix --> decisionPrompt
  handoff --> decisionPrompt
  charts --> decisionPrompt
  decisionPrompt --> decisionModel[DecisionStage]
  decisionModel --> finalDecision[FinalDecision]
```

- `research`：更像微观结构研究员，只负责提炼“现在最值得做的一条路、哪里失效、哪里别碰”
- `decision`：更像交易员，只负责开仓/持仓处理，不追求格式完整，而追求**执行质量**
- `decision` 输出目标是“交易备忘录”，不是分析报告
- `decision` 的机器可读字段保留最小集合：`primary_decision`、`execution_mode`、`trade_plan / wait_plan`、`position_action`、`hard_invalidation / stop_loss`、`summary`

### 把 output 交给网页版 ChatGPT / Claude

1. **若你要模拟自动分析的最终决策阶段**：优先使用 `output/btc_decision_system.txt` + `output/btc_decision_prompt.txt`。
2. **若你要手动复现完整双阶段**：
   先用 `output/btc_research_system.txt` + `output/btc_research_prompt.txt` 生成研究结果；
   再把研究结果整理成 `btc_research_handoff.json` 风格，配合 `btc_decision_system.txt` + `btc_decision_prompt.txt` 做最终决策。
3. **图表**：上传 `output/charts/` 下 `BTCUSDT_4h.png`、`BTCUSDT_1h.png`、`BTCUSDT_15m.png`，若有则再加 `BTCUSDT_spot_perp.png`。`research` 阶段可加 `5m`；`decision` 阶段通常只需要关键图。
4. **不必上传**：`btc_ai_analysis.md` / `btc_ai_decision.md` 是**上次**自动分析结果，不是输入；`btc_context.json` 体积较大，一般只在你想让模型查原始字段时才需要。

## 报告模式说明

`btc_prompt.txt` 支持两种生成模式，通过 `config.py` 中的 `REPORT_MODE` 或 `--report-mode` 参数控制：

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
│   ├── prompt_generator.py   # research/decision 双阶段提示词生成器
│   └── summary_table.py      # Markdown 摘要表
├── advisor/
│   ├── ai_advisor.py         # OpenAI API 集成
│   ├── analysis_parser.py    # 最终输出解析（信号/持仓/等待识别）
│   ├── analysis_history.py   # AI 预测历史追踪与偏差校准注入
│   ├── risk_monitor.py       # 交易风控监控（日亏限额/复仇冷静/频率过高等）
│   ├── smart_ai_scheduler.py # 智能 AI 调度（信号驱动快速轮询）
│   ├── change_detector.py    # 市场变化检测（避免重复 AI 调用）
│   └── pushplus_notifier.py  # PushPlus 微信推送
└── charts/
    └── kline_chart.py        # matplotlib 图表生成
```

## 交易风控监控

`--monitor` 模式下，每轮行情采集前自动检查以下 4 条交易纪律规则，触发时通过 PushPlus 推送微信提醒：

| 规则 | 触发条件 | 说明 |
|------|----------|------|
| 日亏损限额 | 今日净亏 > 40 USDT | 每超出 10 USDT 再推一次 |
| 复仇冷静期 | 60 分钟内连续 2 笔亏损 | 提醒暂停 60 分钟 |
| 大赢次日警告 | 昨日净盈 > 50 USDT | 每日首次运行时推一次 |
| 频率过高 | 1 小时内成交 ≥ 5 次 | 每小时最多推一次 |

需要设置 `BINANCE_API_KEY` + `BINANCE_API_SECRET`（只读权限）+ `PUSHPLUS_TOKEN`。状态文件：`output/.risk_monitor_state.json`。

## AI 历史上下文

每次 `--auto-analyze` 完成后，程序自动将**最终 decision 阶段**的方向、止损、置信度存入 `output/.analysis_history.json`。下次运行时，利用**已拉取的 1h K 线**（无额外 API 调用）回填 1h/4h 实际价格结果，并将近 5 条记录注入 `decision` prompt 末尾：

```
=== PRIOR_DECISIONS_CONTEXT ===
[校准参考 — 请在完成 PHASE A 独立判断后再查阅此段，勿将历史决策作为当次方向依据]
03-19 14:00  BTC@84200  结论:开多  stop:83200  → 1h:83500(-700)[MISS]  4h:82800(-1400)[MISS]
03-18 22:00  BTC@85100  结论:等待                → 1h:pending  4h:pending
recent_direction_bias: long_heavy (2多/0空/2次)
```

AI 可据此识别系统性方向偏差，在连续误判后主动调整分析角度。

## API 密钥说明

- **公开市场数据**端点无需 API Key。
- 当环境变量中存在 `BINANCE_API_KEY` 和 `BINANCE_API_SECRET` 时，自动采集持仓数据及风控监控。
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
- `btc_context.json` 保存完整上下文与 `raw_appendix`，可供自定义后处理或调试。
- `btc_prompt.txt` / `btc_system.txt` 主要用于兼容旧流程或手工粘贴；自动分析时优先参考 `btc_research_*` 与 `btc_decision_*`。
- 双阶段 `gpt-5` 方案更偏“质量优先”而非“速度优先”；若你更看重速度，可手动降模型或降低 `OPENAI_REASONING_EFFORT`。
- 30m/1h K 线 Delta 需要 Binance 返回 `taker_buy_base` 字段；若不可用，报告中会标注 `kline_flow: unavailable`。
