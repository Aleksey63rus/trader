"""Уведомления в MAX (maxapi). Без токена — только лог."""
from __future__ import annotations

import logging
import os

log = logging.getLogger("investor.notifications")


async def send_max_message(text: str, user_id: str | None = None, token: str | None = None) -> bool:
    """
    Отправка сообщения. Если нет MAX_BOT_TOKEN — пишем в лог и возвращаем False.
    """
    token = token or os.getenv("MAX_BOT_TOKEN")
    user_id = user_id or os.getenv("MAX_USER_ID")
    if not token or not user_id:
        log.info("[MAX disabled] %s", text[:500])
        return False
    try:
        from maxapi import Bot  # type: ignore

        bot = Bot(token)
        uid = int(user_id) if str(user_id).isdigit() else user_id
        await bot.send_message(user_id=uid, text=text)  # type: ignore[attr-defined]
        return True
    except Exception as e:  # noqa: BLE001
        log.exception("MAX send failed: %s", e)
        return False


def notify_sync_stub(text: str) -> None:
    """Синхронная заглушка для вызовов из не-async кода."""
    log.info("[notify] %s", text[:500])
