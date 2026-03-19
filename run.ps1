# BTC Context Tool 运行脚本 — 从 .env.local 加载密钥（勿提交 .env.local）
$envFile = Join-Path $PSScriptRoot ".env.local"
if (-not (Test-Path $envFile)) {
    Write-Error "缺少 .env.local，请复制并填写密钥。"
    exit 1
}
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -match '^\s*#' -or $line -eq '') { return }
    if ($line -match '^\s*export\s+(\w+)=(.*)$') {
        $name = $Matches[1]
        $val = $Matches[2].Trim().Trim('"').Trim("'")
        Set-Item -Path "env:$name" -Value $val
    }
}

# 运行示例（任选其一）：
# python main.py
# 默认每 10 分钟一轮；要 5 分钟请加 --loop 300
python main.py --monitor
