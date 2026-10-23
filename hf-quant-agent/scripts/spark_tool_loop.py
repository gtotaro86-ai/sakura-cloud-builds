"""One real Spark tool round-trip. The tool reads public HF metadata only."""

import json
import urllib.parse
import urllib.request

SPARK = "http://127.0.0.1:8088/v1/chat/completions"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "inspect_hf_repo",
        "description": "Read public Hugging Face model metadata. Do not execute repository code.",
        "parameters": {
            "type": "object",
            "properties": {"repo_id": {"type": "string"}},
            "required": ["repo_id"],
        },
    },
}]


def post(payload: dict) -> dict:
    request = urllib.request.Request(
        SPARK,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


def inspect(repo_id: str) -> dict:
    if not repo_id or repo_id.count("/") > 1 or ".." in repo_id or " " in repo_id:
        return {"error": "invalid repo id"}
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(repo_id)
    request = urllib.request.Request(url, headers={"User-Agent": "hf-quant-agent"})
    with urllib.request.urlopen(request, timeout=30) as response:
        info = json.loads(response.read().decode())
    return {
        "repo_id": info.get("id"),
        "sha": info.get("sha"),
        "pipeline_tag": info.get("pipeline_tag"),
        "license": (info.get("cardData") or {}).get("license"),
        "private": info.get("private"),
        "trust_remote_code_executed": False,
    }


def main() -> None:
    first = post({
        "model": "spark-x2.6-4b",
        "messages": [{"role": "user", "content": "Call inspect_hf_repo for repo_id gpt2."}],
        "tools": TOOLS,
        "tool_choice": "required",
        "max_tokens": 60,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    message = (first.get("choices") or [{}])[0].get("message") or {}
    calls = message.get("tool_calls") or []
    if not calls:
        print(json.dumps({"ok": False, "reason": "no tool call", "content": message.get("content")}, ensure_ascii=False))
        return
    call = calls[0]
    args = json.loads(call["function"]["arguments"])
    result = inspect(args.get("repo_id", ""))
    second = post({
        "model": "spark-x2.6-4b",
        "messages": [
            {"role": "user", "content": "Call inspect_hf_repo for repo_id gpt2."},
            message,
            {"role": "tool", "tool_call_id": call.get("id"), "content": json.dumps(result)},
        ],
        "tools": TOOLS,
        "max_tokens": 80,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    final = (second.get("choices") or [{}])[0].get("message") or {}
    print(json.dumps({
        "ok": True,
        "tool": call["function"]["name"],
        "arguments": args,
        "tool_result": result,
        "final": (final.get("content") or "")[:500],
        "final_tool_calls": final.get("tool_calls"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
