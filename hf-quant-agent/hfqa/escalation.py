"""Optional DeepSeek escalation. Disabled and budgeted by default. No call without a key and an explicit enable flag."""

from __future__ import annotations

import json
from pathlib import Path

# Public list prices checked conceptually at integration time. The client re-reads
# the configured table and refuses a call when the estimate plus reserve exceeds the cap.
# These are planning rates, not an invoice. Update them before enabling the fallback.
DEFAULT_RATES = {
    "deepseek-chat": {"input_per_million": 0.14, "output_per_million": 0.28},
    "deepseek-reasoner": {"input_per_million": 0.55, "output_per_million": 2.19},
}
PREFERRED_MODEL = "deepseek-chat"


class BudgetExceeded(RuntimeError):
    pass


class EscalationDisabled(RuntimeError):
    pass


class BudgetManager:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({
                "enabled": False,
                "require_confirmation": True,
                "per_job_usd": 0.50,
                "monthly_usd": 3.00,
                "spent_month_usd": 0.0,
                "month": "",
                "reserve_ratio": 0.25,
                "model": PREFERRED_MODEL,
                "rates": DEFAULT_RATES,
            }, indent=2), encoding="utf-8")
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def estimate(self, input_tokens: int, output_tokens: int, model: str | None = None) -> float:
        model = model or self.data.get("model") or PREFERRED_MODEL
        rates = (self.data.get("rates") or DEFAULT_RATES).get(model) or DEFAULT_RATES[PREFERRED_MODEL]
        raw = (input_tokens / 1_000_000) * rates["input_per_million"] + (output_tokens / 1_000_000) * rates["output_per_million"]
        return round(raw * (1 + float(self.data.get("reserve_ratio") or 0)), 6)

    def authorize(self, job_spent: float, estimate: float, confirmed: bool = False) -> dict:
        if not self.data.get("enabled"):
            raise EscalationDisabled("DeepSeek fallback is disabled")
        if self.data.get("require_confirmation") and not confirmed:
            raise EscalationDisabled("manual confirmation required before a paid call")
        projected_job = job_spent + estimate
        projected_month = float(self.data.get("spent_month_usd") or 0) + estimate
        if projected_job > float(self.data["per_job_usd"]) or projected_month > float(self.data["monthly_usd"]):
            raise BudgetExceeded(
                f"estimate {estimate} would exceed job {self.data['per_job_usd']} or month {self.data['monthly_usd']}"
            )
        return {"authorized": True, "estimate_usd": estimate, "projected_job": projected_job, "projected_month": projected_month}

    def record(self, actual_usd: float) -> None:
        # Conservative: if the provider later revises upward, the reserve already covered a margin.
        self.data["spent_month_usd"] = round(float(self.data.get("spent_month_usd") or 0) + actual_usd, 6)
        self.save()


def build_diagnosis(error: str, files: list[dict], attempts: int) -> dict:
    return {
        "attempts": attempts,
        "error": error[:4000],
        "files": [{"path": item.get("path"), "excerpt": (item.get("excerpt") or "")[:2000]} for item in files[:6]],
        "secrets_included": False,
        "instruction": "Return a minimal patch and a test. Do not request credentials.",
    }


class DeepSeekClient:
    def __init__(self, budget: BudgetManager, transport=None):
        self.budget = budget
        self.transport = transport  # callable(payload) -> dict, used for the mock test

    def complete(self, diagnosis: dict, job_spent: float, confirmed: bool = False) -> dict:
        estimate = self.budget.estimate(input_tokens=2500, output_tokens=1500)
        auth = self.budget.authorize(job_spent, estimate, confirmed=confirmed)
        if self.transport is None:
            raise EscalationDisabled("no DeepSeek transport configured; refusing a live call")
        response = self.transport({"model": self.budget.data.get("model"), "diagnosis": diagnosis})
        actual = float(response.get("cost_usd") or estimate)
        self.budget.record(actual)
        return {"authorization": auth, "response": response, "cost_usd": actual, "mock": bool(response.get("mock"))}
