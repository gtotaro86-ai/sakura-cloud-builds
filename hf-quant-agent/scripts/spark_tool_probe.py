"""Ask the local Spark server for one tool call. Records the raw reply. No secrets."""

import json
import urllib.request

payload = {
    "model": "spark-x2.6-4b",
    "messages": [
        {"role": "user", "content": "Call inspect_hf_repo for https://huggingface.co/gpt2 and do not answer in prose."}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "inspect_hf_repo",
            "description": "Inspect a Hugging Face model repository.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    }],
    "tool_choice": "required",
    "max_tokens": 80,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"},
}
request = urllib.request.Request(
    "http://127.0.0.1:8088/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    body = json.loads(response.read().decode())
message = ((body.get("choices") or [{}])[0].get("message") or {})
print(json.dumps({
    "tool_calls": message.get("tool_calls"),
    "content": (message.get("content") or "")[:800],
    "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
    "timings": body.get("timings"),
}, ensure_ascii=False))
