"""The runner re-reads the pinned source's license on the Hub. A vendor license (`other`) that Oracle approved is carried in
the manifest, because approvals live only in Oracle's data directory; the runner honours it only while the live license
still carries the very same name."""
from __future__ import annotations


def runner_accepts(manifest_decision, live: dict) -> bool:
    if ((live or {}).get('license_decision') or {}).get('redistributable') is True:
        return True
    pinned = manifest_decision if isinstance(manifest_decision, dict) else {}
    name = str(pinned.get('license_name') or '').strip().lower()
    if pinned.get('redistributable') is not True or not name:
        return False
    if not (pinned.get('operator_approved') is True or pinned.get('automatic') is True):
        return False
    return (live or {}).get('license') == 'other' and str((live or {}).get('license_name') or '').strip().lower() == name
