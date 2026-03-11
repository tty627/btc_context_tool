"""Generate a self-contained HTML dashboard from market context."""

from typing import Dict


class HtmlReportGenerator:
    def build(self, context: Dict, analysis_text: str = "") -> str:
        price = context.get("price", 0)
        symbol = context.get("symbol", "BTCUSDT")
        generated_at = context.get("generated_at", "")
        signal = context.get("signal_score", {})
        composite = signal.get("composite_score", 50)
        bias = signal.get("bias", "neutral")
        strength = signal.get("strength", "neutral")
        components = signal.get("components", {})
        sizing = context.get("position_sizing", {})
        session = context.get("session_context", {})
        deploy = context.get("deployment_context", {})
        stats = context.get("stats_24h", {})

        bias_color = {"bullish": "#22c55e", "bearish": "#ef4444", "neutral": "#eab308"}.get(bias, "#6b7280")
        gauge_rotation = (composite / 100) * 180 - 90

        component_bars = ""
        for name, comp in components.items():
            score = comp.get("score", 50)
            color = "#22c55e" if score >= 60 else "#ef4444" if score <= 40 else "#eab308"
            component_bars += f"""
            <div class="comp-row">
              <span class="comp-label">{name}</span>
              <div class="comp-bar-bg"><div class="comp-bar" style="width:{score}%;background:{color}"></div></div>
              <span class="comp-val">{score}</span>
            </div>"""

        tf_rows = ""
        for tf, metrics in context.get("timeframes", {}).items():
            ema = metrics.get("ema", {})
            rsi = metrics.get("rsi", {})
            atr = metrics.get("atr", {})
            features = metrics.get("features", {})
            trend = features.get("trend", "-")
            trend_cls = "bull" if trend == "bullish" else "bear" if trend == "bearish" else "neut"
            tf_rows += f"""
            <tr>
              <td>{tf}</td>
              <td class="{trend_cls}">{trend}</td>
              <td>{features.get('momentum','-')}</td>
              <td>{round(float(rsi.get('14',0)),1)}</td>
              <td>{round(float(atr.get('atr',0)),1)}</td>
              <td>{round(float(atr.get('atr_pct',0)),2)}%</td>
              <td>{round(float(ema.get('7',0)),1)}</td>
              <td>{round(float(ema.get('25',0)),1)}</td>
            </tr>"""

        sizing_html = ""
        if sizing.get("available"):
            ref_long = sizing.get("reference_levels", {}).get("long", {})
            ref_short = sizing.get("reference_levels", {}).get("short", {})
            sizing_html = f"""
            <div class="card">
              <h3>Position Sizing (1% risk, 10x)</h3>
              <div class="kv">ATR({sizing.get('atr_timeframe','')}): <b>{sizing.get('atr')}</b> | SL distance: <b>{sizing.get('sl_distance')}</b> ({sizing.get('sl_pct')}%)</div>
              <div class="kv">Position: <b>{sizing.get('position_size_usdt')}U</b> ({sizing.get('position_size_btc')} BTC) | Margin: {sizing.get('margin_required')}U ({sizing.get('margin_usage_pct')}%)</div>
              <table>
                <tr><th></th><th>Stop Loss</th><th>TP1 (2R)</th><th>TP2 (3R)</th></tr>
                <tr><td>Long</td><td>{ref_long.get('stop_loss')}</td><td>{ref_long.get('tp1')}</td><td>{ref_long.get('tp2')}</td></tr>
                <tr><td>Short</td><td>{ref_short.get('stop_loss')}</td><td>{ref_short.get('tp1')}</td><td>{ref_short.get('tp2')}</td></tr>
              </table>
            </div>"""

        analysis_html = ""
        if analysis_text:
            escaped = analysis_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            analysis_html = f"""
            <div class="card">
              <h3>AI Trading Plan</h3>
              <pre class="analysis">{escaped}</pre>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{symbol} Market Dashboard</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0f172a; color:#e2e8f0; padding:16px; }}
  .header {{ text-align:center; margin-bottom:24px; }}
  .header h1 {{ font-size:28px; color:#f8fafc; }}
  .header .meta {{ color:#94a3b8; font-size:14px; margin-top:4px; }}
  .price {{ font-size:40px; font-weight:700; color:#f8fafc; margin:8px 0; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
  .card {{ background:#1e293b; border-radius:12px; padding:20px; }}
  .card h3 {{ color:#94a3b8; font-size:14px; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; }}
  .score-big {{ font-size:48px; font-weight:700; text-align:center; }}
  .badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; }}
  .comp-row {{ display:flex; align-items:center; gap:8px; margin:6px 0; }}
  .comp-label {{ width:90px; font-size:13px; color:#94a3b8; }}
  .comp-bar-bg {{ flex:1; height:8px; background:#334155; border-radius:4px; overflow:hidden; }}
  .comp-bar {{ height:100%; border-radius:4px; transition:width .3s; }}
  .comp-val {{ width:30px; text-align:right; font-size:13px; font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }}
  th {{ text-align:left; color:#94a3b8; padding:6px 8px; border-bottom:1px solid #334155; }}
  td {{ padding:6px 8px; border-bottom:1px solid #1e293b; }}
  .bull {{ color:#22c55e; font-weight:600; }}
  .bear {{ color:#ef4444; font-weight:600; }}
  .neut {{ color:#eab308; }}
  .kv {{ font-size:13px; color:#cbd5e1; margin:4px 0; }}
  .kv b {{ color:#f8fafc; }}
  .analysis {{ white-space:pre-wrap; font-size:13px; line-height:1.6; color:#cbd5e1; max-height:600px; overflow-y:auto; }}
  .stat-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
  .stat {{ }}
  .stat-label {{ font-size:11px; color:#64748b; }}
  .stat-value {{ font-size:16px; font-weight:600; color:#f1f5f9; }}
</style>
</head>
<body>
<div class="header">
  <h1>{symbol} Market Dashboard</h1>
  <div class="meta">{generated_at}</div>
  <div class="price">${price:,.2f}</div>
</div>

<div class="grid">
  <div class="card">
    <h3>Signal Score</h3>
    <div class="score-big" style="color:{bias_color}">{composite}</div>
    <div style="text-align:center;margin:8px 0">
      <span class="badge" style="background:{bias_color}20;color:{bias_color}">{strength}</span>
    </div>
    {component_bars}
  </div>

  <div class="card">
    <h3>24h Statistics</h3>
    <div class="stat-grid">
      <div class="stat"><div class="stat-label">24h High</div><div class="stat-value">${float(stats.get('high_price',0)):,.2f}</div></div>
      <div class="stat"><div class="stat-label">24h Low</div><div class="stat-value">${float(stats.get('low_price',0)):,.2f}</div></div>
      <div class="stat"><div class="stat-label">24h Change</div><div class="stat-value">{stats.get('price_change_percent',0)}%</div></div>
      <div class="stat"><div class="stat-label">24h Volume</div><div class="stat-value">{float(stats.get('quote_volume',0)):,.0f}U</div></div>
      <div class="stat"><div class="stat-label">Session</div><div class="stat-value">{session.get('current_session','?').upper()}</div></div>
      <div class="stat"><div class="stat-label">Funding CD</div><div class="stat-value">{session.get('funding_countdown_label','?')}</div></div>
      <div class="stat"><div class="stat-label">Deploy Score</div><div class="stat-value">{deploy.get('deployment_score','?')} ({deploy.get('deployment_score_value','?')})</div></div>
      <div class="stat"><div class="stat-label">Bias</div><div class="stat-value">{deploy.get('primary_bias','?')}</div></div>
    </div>
  </div>

  <div class="card" style="grid-column:1/-1">
    <h3>Multi-Timeframe Technical</h3>
    <table>
      <tr><th>TF</th><th>Trend</th><th>Momentum</th><th>RSI</th><th>ATR</th><th>ATR%</th><th>EMA7</th><th>EMA25</th></tr>
      {tf_rows}
    </table>
  </div>

  {sizing_html}
  {analysis_html}
</div>
</body>
</html>"""
