"""Quick Telegram connectivity check for ClinicAI."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_PROXY,
    TELEGRAM_TRUST_ENV,
)


def main() -> int:
    if not TELEGRAM_BOT_TOKEN:
        print("FAIL: TELEGRAM_BOT_TOKEN is missing in .env")
        return 1

    proxy_mode = TELEGRAM_PROXY or ("system/env" if TELEGRAM_TRUST_ENV else "direct")
    print(f"Mode: {proxy_mode}")

    try:
        with httpx.Client(
            timeout=20.0,
            trust_env=TELEGRAM_TRUST_ENV,
            proxy=TELEGRAM_PROXY,
        ) as client:
            me = client.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe")
            me.raise_for_status()
            bot = me.json().get("result", {})
            print(f"OK: bot @{bot.get('username', '?')} ({bot.get('first_name', '')})")

            webhook = client.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getWebhookInfo"
            )
            webhook.raise_for_status()
            info = webhook.json().get("result", {})
            url = info.get("url") or ""
            if url:
                print(f"WARN: webhook is set to {url!r} — polling will not receive updates")
            else:
                print("OK: no webhook (polling mode is fine)")

            updates = client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                json={"timeout": 0, "limit": 1},
            )
            updates.raise_for_status()
            pending = len(updates.json().get("result", []))
            print(f"OK: getUpdates works (sample pending={pending})")
    except httpx.HTTPError as exc:
        print(f"FAIL: cannot reach Telegram API ({exc})")
        if TELEGRAM_PROXY:
            print("Hint: remove TELEGRAM_PROXY from .env if nothing listens on that port.")
        elif not TELEGRAM_TRUST_ENV:
            print("Hint: if Telegram is blocked on your network, enable VPN or set TELEGRAM_PROXY.")
        return 1

    print("\nIf the bot still does not reply:")
    print("1. Stop ALL python main.py processes (only one instance allowed).")
    print("2. Run: python main.py")
    print("3. Send /start and watch the terminal for 'Incoming /start from ...'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
