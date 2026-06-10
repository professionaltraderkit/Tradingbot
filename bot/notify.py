"""Telegram delivery for the daily reports. Falls back to printing to the
console when TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not configured.
"""

import logging
import os

import requests

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN") or None
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID") or None

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        """Send a message; returns True if delivered to Telegram."""
        if not self.telegram_enabled:
            print("\n" + "=" * 60)
            print(text)
            print("=" * 60 + "\n")
            return False
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=15,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.error("Telegram send failed: %s", exc)
            print("\n[Telegram delivery failed — report below]\n" + text + "\n")
            return False
