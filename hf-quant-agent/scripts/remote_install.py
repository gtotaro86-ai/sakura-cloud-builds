#!/usr/bin/env python3
"""Finish the Oracle install. Does not print the API token."""

import os
import secrets
import shutil
import subprocess
import time

ROOT = "/opt/hf-quant-agent"
TOKEN = "/etc/hf-quant-agent/api.token"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    run(["id", "hfquant"]) if shutil.which("id") else None
    try:
        subprocess.run(["id", "hfquant"], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        run(["useradd", "--system", "--create-home", "--home-dir", "/var/lib/hf-quant-agent", "--shell", "/usr/sbin/nologin", "hfquant"])
    for path in ("/var/lib/hf-quant-agent", "/var/lib/hf-quant-agent/data", "/var/lib/hf-quant-agent/work", "/etc/hf-quant-agent"):
        os.makedirs(path, exist_ok=True)
    if not os.path.exists(TOKEN) or os.path.getsize(TOKEN) < 16:
        with open(TOKEN, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(32))
    os.chmod(TOKEN, 0o640)
    shutil.chown(TOKEN, user="root", group="hfquant")
    for path in ("/var/lib/hf-quant-agent", "/var/lib/hf-quant-agent/data", "/var/lib/hf-quant-agent/work"):
        shutil.chown(path, user="hfquant", group="hfquant")
        os.chmod(path, 0o750)
    if not os.path.exists(ROOT + "/.venv/bin/pip"):
        run(["python3", "-m", "venv", ROOT + "/.venv"])
        run([ROOT + "/.venv/bin/pip", "install", "-r", ROOT + "/requirements.txt"])
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "enable", "--now", "hf-quant-agent"])
    time.sleep(2)
    print(subprocess.check_output(["systemctl", "is-active", "hf-quant-agent", "AdGuardHome", "unbound", "tailscaled"], text=True))


if __name__ == "__main__":
    main()
