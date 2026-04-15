"""
Telegram-бот для оповещений о торговых событиях.
Отправляет уведомления при входе, выходе по TP/SL, drawdown-предупреждениях.
"""
from __future__ import annotations
import asyncio
import aiohttp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.database import get_setting, log_alert, mark_alert_sent, get_pending_alerts

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


async def _send_message(token: str, chat_id: str, text: str) -> bool:
    url = TELEGRAM_API.format(token=token)
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:
                return r.status == 200
    except Exception:
        return False


def format_trade_enter(ticker: str, entry_px: float, shares: float,
                       strategy: str, sl_px: float = 0.0) -> str:
    cost = entry_px * shares
    sl_str = f"\n🛑 SL: <b>{sl_px:.2f} ₽</b>" if sl_px else ""
    return (
        f"📈 <b>ВХОД В СДЕЛКУ</b>\n"
        f"Инструмент: <b>{ticker}</b>\n"
        f"Стратегия: {strategy}\n"
        f"Цена входа: <b>{entry_px:.2f} ₽</b>\n"
        f"Количество: {shares:.4f} шт\n"
        f"Сумма: <b>{cost:,.0f} ₽</b>"
        f"{sl_str}"
    )


def format_trade_exit(ticker: str, entry_px: float, exit_px: float,
                      pnl_rub: float, pnl_pct: float, reason: str,
                      hold_days: float) -> str:
    icon = "✅" if pnl_rub > 0 else "❌"
    reason_ru = {"SL": "Стоп-лосс", "SL_FRACTAL": "Фракт. SL",
                 "TIME": "Тайм-аут", "FORCED": "Принудит."}.get(reason, reason)
    return (
        f"{icon} <b>ВЫХОД ИЗ СДЕЛКИ</b>\n"
        f"Инструмент: <b>{ticker}</b>\n"
        f"Цена входа: {entry_px:.2f} ₽ → Выход: <b>{exit_px:.2f} ₽</b>\n"
        f"Причина: {reason_ru}\n"
        f"П/У: <b>{pnl_rub:+,.0f} ₽ ({pnl_pct:+.1f}%)</b>\n"
        f"Удержание: {hold_days:.0f} дн"
    )


def format_drawdown_warning(current_dd: float, capital: float) -> str:
    return (
        f"⚠️ <b>ПРЕДУПРЕЖДЕНИЕ О ПРОСАДКЕ</b>\n"
        f"Текущая просадка: <b>{current_dd:.1f}%</b>\n"
        f"Капитал: {capital:,.0f} ₽"
    )


async def send_alert(event_type: str, message: str, ticker: str | None = None) -> bool:
    """Логирует алерт и отправляет его в Telegram."""
    alert_id = log_alert(event_type, message, ticker)
    token   = get_setting("telegram_token", "")
    chat_id = get_setting("telegram_chat_id", "")
    if not token or not chat_id:
        return False  # токен не настроен
    ok = await _send_message(token, chat_id, message)
    if ok:
        mark_alert_sent(alert_id)
    return ok


def send_alert_sync(event_type: str, message: str, ticker: str | None = None) -> bool:
    """Синхронная обёртка для отправки алертов."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return False  # нельзя запустить синхронно внутри async
        return loop.run_until_complete(send_alert(event_type, message, ticker))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(send_alert(event_type, message, ticker))
        loop.close()
        return result


async def flush_pending_alerts() -> int:
    """Отправляет все неотправленные алерты (retry при запуске)."""
    token   = get_setting("telegram_token", "")
    chat_id = get_setting("telegram_chat_id", "")
    if not token or not chat_id:
        return 0
    pending = get_pending_alerts()
    sent = 0
    for alert in pending:
        ok = await _send_message(token, chat_id, alert["message"])
        if ok:
            mark_alert_sent(alert["id"])
            sent += 1
    return sent
