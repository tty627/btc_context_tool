"""一次性 PushPlus 连通测试：python scripts/test_pushplus.py"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 加载 .env.local
env_path = ROOT / ".env.local"
if env_path.is_file():
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

import httpx  # noqa: E402

token = os.getenv("PUSHPLUS_TOKEN", "").strip()
if not token:
    print("未设置 PUSHPLUS_TOKEN（检查 .env.local）")
    sys.exit(1)

proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
url = "https://www.pushplus.plus/send"
payload = {
    "token": token,
    "title": "BTC Context 连通测试",
    "content": "PushPlus 通道正常。若微信收到本条说明配置 OK。",
    "template": "txt",
}
kw = {"json": payload, "timeout": 30}
if proxy:
    kw["proxy"] = proxy
    print("使用代理:", proxy)
try:
    r = httpx.post(url, **kw)
except Exception as e:
    print("请求失败:", e)
    sys.exit(1)
print("HTTP 状态:", r.status_code)
try:
    data = r.json()
    print("PushPlus code:", data.get("code"), "| msg:", data.get("msg"))
    sys.exit(0 if data.get("code") == 200 else 1)
except json.JSONDecodeError:
    print(r.text[:400])
    sys.exit(1)
