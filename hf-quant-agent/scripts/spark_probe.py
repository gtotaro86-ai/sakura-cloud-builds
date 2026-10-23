"""One local completion against the loopback Spark server. No secrets."""

import json
import time
import urllib.request

payload = {
    "model": "spark-x2.6-4b",
    "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    "http://127.0.0.1:8088/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
started = time.time()
with urllib.request.urlopen(request, timeout=180) as response:
    body = json.loads(response.read().decode())
elapsed = time.time() - started
usage = body.get("usage") or {}
message = ((body.get("choices") or [{}])[0].get("message") or {})
print(json.dumps({
    "seconds": round(elapsed, 2),
    "usage": usage,
    "timings": body.get("timings"),
    "content": (message.get("content") or "")[:500],
    "reasoning": (message.get("reasoning_content") or "")[:300],
}, ensure_ascii=False))
