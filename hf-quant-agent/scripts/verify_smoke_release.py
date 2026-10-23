"""Check the public smoke repo. Does not upload or requantize."""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hfqa.publish import release_gaps

REPO = "webmp3/hfqa-smoke-Qwen2.5-0.5B-Instruct-requant-GGUF"
EXPECTED = {
    "qwen2.5-0.5b-instruct-Q4_K_M-requant.gguf": 491400032,
    "qwen2.5-0.5b-instruct-Q5_K_M-requant.gguf": 522186592,
    "README.md": (ROOT / "data" / "smoke" / "README.md").stat().st_size,
}


def main() -> None:
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{REPO}",
        headers={"User-Agent": "hf-quant-agent"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        info = json.loads(response.read().decode())
    names = {item.get("rfilename") for item in info.get("siblings") or []}
    remote = {}
    for name in EXPECTED:
        if name not in names:
            continue
        head = urllib.request.Request(
            f"https://huggingface.co/{REPO}/resolve/main/{name}",
            method="HEAD",
            headers={"User-Agent": "hf-quant-agent"},
        )
        with urllib.request.urlopen(head, timeout=60) as response:
            length = response.headers.get("content-length")
            remote[name] = int(length) if length else None
    gaps = release_gaps(remote, EXPECTED)
    report = {
        "repo": REPO,
        "url": f"https://huggingface.co/{REPO}",
        "sha": info.get("sha"),
        "remote_files": sorted(name for name in remote if name),
        "gaps": gaps,
        "gguf_architecture": (info.get("gguf") or {}).get("architecture"),
        "complete": gaps["complete"],
    }
    out = ROOT / "data" / "smoke" / "release-status.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
