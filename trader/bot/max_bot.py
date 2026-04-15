"""
Уведомления через мессенджер MAX (VK).
Библиотека: maxapi (https://pypi.org/project/maxapi/)
Документация API: https://dev.max.ru/docs-api

Установка: pip install maxapi
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.database import get_setting, log_alert, mark_alert_sent, get_pending_alerts

# ── Шаблоны сообщений (строгий деловой стиль) ─────────────────────────────

def fmt_entry(ticker: str, entry_px: float, shares: float,
              strategy: str, sl_px: float = 0.0) -> str:
    cost = entry_px * shares
    sl_str = f"\nСтоп-лосс:  {sl_px:.2f} руб." if sl_px else ""
    return (
        f"ВХОД В ПОЗИЦИЮ\n"
        f"{'─'*28}\n"
        f"Инструмент: {ticker}\n"
        f"Стратегия:  {strategy}\n"
        f"Цена входа: {entry_px:.2f} руб.\n"
        f"Объём:      {shares:.4f} шт. / {cost:,.0f} руб."
        f"{sl_str}"
    )


def fmt_exit(ticker: str, entry_px: float, exit_px: float,
             pnl_rub: float, pnl_pct: float, reason: str, hold_days: float) -> str:
    sign = "+" if pnl_rub >= 0 else ""
    status = "ПРИБЫЛЬ" if pnl_rub >= 0 else "УБЫТОК"
    reason_ru = {
        "TIME":       "Тайм-аут (макс. срок)",
        "SL":         "Стоп-лосс",
        "SL_FRACTAL": "Фракт. стоп-лосс",
        "FORCED":     "Принудительное закрытие",
    }.get(reason, reason)
    return (
        f"ВЫХОД ИЗ ПОЗИЦИИ — {status}\n"
        f"{'─'*28}\n"
        f"Инструмент: {ticker}\n"
        f"Вход / выход: {entry_px:.2f} / {exit_px:.2f} руб.\n"
        f"Причина: {reason_ru}\n"
        f"Удержание: {hold_days:.0f} дн.\n"
        f"Результат: {sign}{pnl_rub:,.0f} руб. ({sign}{pnl_pct:.1f}%)"
    )


def fmt_drawdown_warning(current_dd: float, capital: float) -> str:
    return (
        f"ПРЕДУПРЕЖДЕНИЕ — ПРОСАДКА\n"
        f"{'─'*28}\n"
        f"Текущая просадка: {current_dd:.1f}%\n"
        f"Капитал: {capital:,.0f} руб."
    )


def fmt_circuit_breaker(n_losses: int) -> str:
    return (
        f"ОСТАНОВКА ТОРГОВЛИ\n"
        f"{'─'*28}\n"
        f"Подряд убыточных сделок: {n_losses}\n"
        f"Требуется ручная проверка и перезапуск."
    )


def fmt_error(module: str, message: str) -> str:
    return (
        f"СИСТЕМНАЯ ОШИБКА\n"
        f"{'─'*28}\n"
        f"Модуль: {module}\n"
        f"Сообщение: {message}"
    )


# ── Отправка через MAX API ──────────────────────────────────────────────────

def _parse_user_id(raw: str) -> int:
    """
    Извлекает числовой user_id из разных форматов MAX:
      - "631107420238"        → 631107420238
      - "id631107420238_1_bot"→ 631107420238
      - "id631107420238"      → 631107420238
    """
    import re
    raw = str(raw).strip()
    m = re.search(r'\d+', raw)
    if not m:
        raise ValueError(f"Не удалось извлечь user_id из: {raw!r}")
    return int(m.group())


async def _send_max(token: str, user_id: str, text: str) -> bool:
    """
    Отправляет сообщение через MAX Bot API (platform-api.max.ru).
    Документация: https://dev.max.ru/docs-api/methods/POST/messages

    Актуальный формат (2025+):
      POST https://platform-api.max.ru/messages?user_id={id}
      Authorization: {token}
      Content-Type: application/json
      Body: {"text": "..."}
    """
    try:
        import aiohttp
        uid = _parse_user_id(user_id)
        url = f"https://platform-api.max.ru/messages?user_id={uid}"
        headers = {
            "Authorization": token,
            "Content-Type":  "application/json",
        }
        payload = {"text": text}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                body = {}
                try:
                    body = await r.json()
                except Exception:
                    pass
                ok = r.status in (200, 201)
                if not ok:
                    import logging
                    logging.warning(
                        "MAX API error: status=%s body=%s", r.status, body
                    )
                return ok
    except Exception as e:
        import logging
        logging.warning("MAX _send_max exception: %s", e)
        return False


async def send_alert(event_type: str, message: str, ticker: str | None = None) -> bool:
    """Логирует алерт в БД и отправляет в MAX."""
    level   = get_setting("max_alert_level", "all")
    # Фильтрация по уровню
    if level == "trades"   and event_type not in ("ENTRY", "EXIT"):
        log_alert(event_type, message, ticker)
        return False
    if level == "critical" and event_type not in ("CIRCUIT_BREAKER", "ERROR", "DRAWDOWN"):
        log_alert(event_type, message, ticker)
        return False

    alert_id = log_alert(event_type, message, ticker)
    token   = get_setting("max_token", "")
    user_id = get_setting("max_user_id", "")

    if not token or not user_id:
        return False  # не настроено

    ok = await _send_max(token, user_id, message)
    if ok:
        mark_alert_sent(alert_id)
    return ok


def send_alert_sync(event_type: str, message: str, ticker: str | None = None) -> bool:
    """Синхронная обёртка для send_alert."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return False
        return loop.run_until_complete(send_alert(event_type, message, ticker))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(send_alert(event_type, message, ticker))
        loop.close()
        return result


async def flush_pending_alerts() -> int:
    """Повторная отправка неотправленных алертов при старте."""
    token   = get_setting("max_token", "")
    user_id = get_setting("max_user_id", "")
    if not token or not user_id:
        return 0
    pending = get_pending_alerts()
    sent = 0
    for a in pending:
        ok = await _send_max(token, user_id, a["message"])
        if ok:
            mark_alert_sent(a["id"])
            sent += 1
    return sent
