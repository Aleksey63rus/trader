"""
Тарифы БКС — фондовый рынок MOEX (упрощённо для бэктеста).

Источник: тариф «Инвестор» (PDF клиента), «Трейдер» — типовые ставки.
Шорт: открытие = продажа бумаги (комиссия как у продажи), закрытие = выкуп (как покупка).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BCSTariff:
    """Параметры для симуляции комиссий и проскальзывания."""
    id: str
    name: str
    commission_buy: float   # доля от оборота
    commission_sell: float
    slippage: float         # на каждую сторону (модель)
    monthly_fee_rub: float  # фикс в месяц при наличии сделок (Трейдер)
    notes: str = ""

    def round_trip_long_pct(self) -> float:
        """Полный круг лонга: покупка + продажа (только комиссии, без слипп)."""
        return (self.commission_buy + self.commission_sell) * 100

    def round_trip_short_pct(self) -> float:
        """Полный круг шорта: продажа (вход) + покупка (выход)."""
        return (self.commission_sell + self.commission_buy) * 100


# ── Тариф «Инвестор» (как в PDF пользователя) ────────────────────────────────
INVESTOR = BCSTariff(
    id="investor",
    name="БКС «Инвестор»",
    commission_buy=0.0,
    commission_sell=0.0030,   # 0.30% с продажи
    slippage=0.0010,          # 0.10% модельное проскальзывание MOEX
    monthly_fee_rub=0.0,
    notes="Покупка 0%, продажа 0.30%. Без абонплаты.",
)

# ── Тариф «Трейдер» — типовая сетка (средний уровень ~0.02% с каждой стороны) ─
# Уточняйте в договоре: ставка зависит от оборота/активов (0.01%–0.03%).
TRADER_MID = BCSTariff(
    id="trader_mid",
    name="БКС «Трейдер» (средняя ставка)",
    commission_buy=0.0002,    # 0.02%
    commission_sell=0.0002,
    slippage=0.0008,          # чуть ниже при активной торговле
    monthly_fee_rub=299.0,
    notes="~0.02% на покупку и на продажу + 299 ₽/мес при сделках в месяце (модель).",
)

ALL_TARIFFS = (INVESTOR, TRADER_MID)


def get_tariff(tariff_id: str) -> BCSTariff:
    for t in ALL_TARIFFS:
        if t.id == tariff_id:
            return t
    return INVESTOR
