"""
core.risk — Профессиональный риск-менеджмент.

Реализует 5 уровней защиты капитала (как в проп-конторах и брокерских домах):

  1. Position Sizing     — фиксированный % риска капитала на сделку
  2. Kelly Criterion     — оптимальный % риска на основе WR и avg W/L
  3. Daily Loss Limit    — стоп торговли при превышении дневного убытка
  4. Max Drawdown Guard  — уменьшение лота при просадке счёта
  5. Correlation Filter  — запрет одновременного открытия коррелирующих позиций

Принципы:
  - Риск 1-2% капитала на сделку (не более)
  - Дневной лимит убытков: 3% капитала
  - При просадке 10% → лот × 0.5
  - При просадке 15% → торговля заморожена
  - Одновременно максимум 3 открытых позиции
  - Нельзя открывать тикеры из одной корреляционной группы одновременно
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Корреляционные группы (MOEX, исторически сильно коррелируют)
# Открываем не более 1 тикера из каждой группы одновременно
# ══════════════════════════════════════════════════════════════════════════════
CORRELATION_GROUPS: list[set[str]] = [
    {"SBER", "SBERP", "T", "VTBR"},              # банки
    {"LKOH", "ROSN", "NVTK", "GAZP", "SNGS", "SNGSP"},  # нефть и газ
    {"NLMK", "MTLR", "CHMF", "MAGN", "RUAL", "GMKN"},   # металлургия
    {"MGNT", "X5"},                               # ритейл
    {"OZPH"},                                     # фарма
    {"YDEX", "OZON"},                             # технологии
    {"TATN", "TATNP"},                            # Татнефть об/преф
    {"TGKA", "IRAO"},                             # электроэнергетика
    {"ALRS", "PLZL"},                             # добыча (алмазы/золото)
    {"PHOR"},                                     # удобрения
    {"AFLT"},                                     # авиация
    {"MTSS"},                                     # телеком
]


# ══════════════════════════════════════════════════════════════════════════════
# Параметры риск-менеджмента
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RiskParams:
    """Конфигурация риск-менеджмента."""

    # --- Position sizing ---
    risk_pct: float = 0.01          # базовый риск на сделку (1% капитала)
    max_risk_pct: float = 0.02      # максимальный риск на сделку (2%)
    min_risk_pct: float = 0.005     # минимальный риск (0.5%, при просадке)

    # --- Portfolio limits ---
    max_positions: int = 3          # максимум открытых позиций одновременно
    max_sector_exposure: int = 1    # максимум позиций из одной корр. группы

    # --- Daily loss limit (как в FTMO) ---
    daily_loss_limit_pct: float = 0.03   # 3% от капитала = стоп дня

    # --- Drawdown control (как у проп-трейдеров) ---
    dd_reduce_threshold: float = 0.10    # при просадке 10% → лот × 0.5
    dd_halt_threshold: float = 0.15      # при просадке 15% → стоп торговли
    dd_lot_multiplier: float = 0.5       # множитель лота при просадке

    # --- Kelly ---
    kelly_fraction: float = 0.25    # дробный Kelly (25% от полного)
    kelly_max: float = 0.02         # кэп Kelly на 2% капитала

    # --- Trailing stop ---
    use_trailing_stop: bool = True
    trailing_atr_mult: float = 2.5  # трейлинг = цена - ATR × mult


# ══════════════════════════════════════════════════════════════════════════════
# Состояние риска (обновляется каждый день симуляции)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RiskState:
    """Текущее состояние риск-системы портфеля."""

    capital: float                  # текущий капитал
    peak_capital: float             # максимум капитала за всё время
    daily_start_capital: float      # капитал на начало текущего дня
    open_tickers: set = field(default_factory=set)   # открытые тикеры

    # Статистика для Kelly
    total_wins: int   = 0
    total_losses: int = 0
    sum_win_pct: float = 0.0        # сумма % выигрышей
    sum_loss_pct: float = 0.0       # сумма % проигрышей

    # Счётчики нарушений
    daily_halt_count: int = 0       # сколько дней было остановлено
    dd_halt_count: int    = 0       # сколько раз срабатывал DD-стоп

    @property
    def drawdown_pct(self) -> float:
        """Текущая просадка от пика (0.0 = нет просадки, 0.15 = 15%)."""
        if self.peak_capital <= 0:
            return 0.0
        return max(0.0, (self.peak_capital - self.capital) / self.peak_capital)

    @property
    def daily_loss_pct(self) -> float:
        """Убыток за текущий день."""
        if self.daily_start_capital <= 0:
            return 0.0
        loss = self.daily_start_capital - self.capital
        return max(0.0, loss / self.daily_start_capital)

    @property
    def open_count(self) -> int:
        return len(self.open_tickers)

    def update_peak(self):
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

    def update_kelly_stats(self, pnl_pct: float):
        if pnl_pct > 0:
            self.total_wins  += 1
            self.sum_win_pct += pnl_pct
        else:
            self.total_losses  += 1
            self.sum_loss_pct  += abs(pnl_pct)


# ══════════════════════════════════════════════════════════════════════════════
# Основной риск-менеджер
# ══════════════════════════════════════════════════════════════════════════════
class RiskManager:
    """
    Центральный риск-менеджер портфеля.

    Использование:
        rm = RiskManager(initial_capital=100_000)

        # В начале каждого дня:
        rm.on_day_start()

        # Перед открытием позиции:
        decision = rm.can_open(ticker, sl_pct)
        if decision.allowed:
            lot_rub = decision.position_size_rub

        # После закрытия позиции:
        rm.on_trade_closed(ticker, pnl_rub, pnl_pct)
    """

    def __init__(self, initial_capital: float, params: Optional[RiskParams] = None):
        self.params = params or RiskParams()
        self.state  = RiskState(
            capital            = initial_capital,
            peak_capital       = initial_capital,
            daily_start_capital= initial_capital,
        )

    # ── День ──────────────────────────────────────────────────────────────────
    def on_day_start(self):
        """Вызывать в начале каждого торгового дня."""
        self.state.daily_start_capital = self.state.capital
        self.state.update_peak()

    # ── Решение об открытии позиции ──────────────────────────────────────────
    def can_open(self, ticker: str, sl_pct: float) -> "OpenDecision":
        """
        Проверяет все условия риска и возвращает решение.

        Args:
            ticker:  тикер инструмента
            sl_pct:  расстояние до SL в % (например 0.05 = 5%)

        Returns:
            OpenDecision с полями: allowed, reason, position_size_rub, risk_pct
        """
        p = self.params
        s = self.state

        # 1. Лимит открытых позиций
        if s.open_count >= p.max_positions:
            return OpenDecision(False, f"MAX_POSITIONS ({p.max_positions})")

        # 2. Дневной лимит убытков
        if s.daily_loss_pct >= p.daily_loss_limit_pct:
            s.daily_halt_count += 1
            return OpenDecision(False,
                f"DAILY_LOSS_LIMIT ({s.daily_loss_pct*100:.1f}% >= "
                f"{p.daily_loss_limit_pct*100:.0f}%)")

        # 3. Просадка → стоп
        if s.drawdown_pct >= p.dd_halt_threshold:
            s.dd_halt_count += 1
            return OpenDecision(False,
                f"DD_HALT ({s.drawdown_pct*100:.1f}% >= "
                f"{p.dd_halt_threshold*100:.0f}%)")

        # 4. Корреляционный фильтр
        corr_block = self._correlation_blocked(ticker)
        if corr_block:
            return OpenDecision(False, f"CORRELATION ({corr_block})")

        # 5. Расчёт размера позиции
        base_risk_pct = self._effective_risk_pct()
        kelly_risk    = self._kelly_risk_pct()
        risk_pct      = min(base_risk_pct, kelly_risk, p.max_risk_pct)
        risk_pct      = max(risk_pct, p.min_risk_pct)

        if sl_pct <= 0:
            return OpenDecision(False, "SL_ZERO")

        # Размер позиции = капитал × риск_пct / sl_pct
        position_size_rub = s.capital * risk_pct / sl_pct

        # Кэп: не более 30% капитала в одну позицию
        position_size_rub = min(position_size_rub, s.capital * 0.30)

        return OpenDecision(
            allowed          = True,
            reason           = "OK",
            position_size_rub= round(position_size_rub, 2),
            risk_pct_used    = risk_pct,
        )

    def on_position_opened(self, ticker: str):
        self.state.open_tickers.add(ticker)

    def on_trade_closed(self, ticker: str, pnl_rub: float, pnl_pct: float):
        """Обновить состояние после закрытия сделки."""
        self.state.capital += pnl_rub
        self.state.open_tickers.discard(ticker)
        self.state.update_peak()
        self.state.update_kelly_stats(pnl_pct)

    # ── Вспомогательные методы ────────────────────────────────────────────────
    def _effective_risk_pct(self) -> float:
        """Снижает риск при просадке."""
        p = self.params
        s = self.state
        dd = s.drawdown_pct

        if dd >= p.dd_reduce_threshold:
            # При просадке 10-15% → риск × 0.5
            return p.risk_pct * p.dd_lot_multiplier
        return p.risk_pct

    def _kelly_risk_pct(self) -> float:
        """
        Дробный Kelly Criterion.
        Kelly% = WR - (1-WR) / (avg_win / avg_loss)
        Используем 25% от полного Kelly (более консервативно).
        """
        p = self.params
        s = self.state

        n_wins   = s.total_wins
        n_losses = s.total_losses
        total    = n_wins + n_losses

        # Недостаточно статистики — используем базовый риск
        if total < 10:
            return p.risk_pct

        wr       = n_wins / total
        avg_win  = s.sum_win_pct / n_wins   if n_wins   > 0 else 0.0
        avg_loss = s.sum_loss_pct / n_losses if n_losses > 0 else 1.0

        if avg_loss <= 0:
            return p.risk_pct

        win_loss_ratio = avg_win / avg_loss
        kelly_full = wr - (1 - wr) / win_loss_ratio

        if kelly_full <= 0:
            # Отрицательный Kelly = стратегия убыточна → минимальный риск
            return p.min_risk_pct

        kelly_fractional = kelly_full * p.kelly_fraction
        return min(kelly_fractional / 100, p.kelly_max)  # kelly в % → доля

    def _correlation_blocked(self, ticker: str) -> Optional[str]:
        """
        Возвращает название блокирующего тикера если
        уже открыта позиция из той же корреляционной группы.
        """
        for group in CORRELATION_GROUPS:
            if ticker in group:
                blocking = group & self.state.open_tickers
                if blocking:
                    return ", ".join(blocking)
        return None

    # ── Отчёт о состоянии риска ───────────────────────────────────────────────
    def summary(self) -> str:
        s = self.state
        p = self.params
        kelly = self._kelly_risk_pct()
        lines = [
            f"  Капитал:         {s.capital:>12,.0f} руб.",
            f"  Пик капитала:    {s.peak_capital:>12,.0f} руб.",
            f"  Просадка:        {s.drawdown_pct*100:>11.1f}%",
            f"  Дн. убыток:      {s.daily_loss_pct*100:>11.1f}%",
            f"  Открытых поз.:   {s.open_count:>12d}",
            f"  Kelly риск:      {kelly*100:>11.2f}%",
            f"  Базовый риск:    {self._effective_risk_pct()*100:>11.2f}%",
            f"  Стопов за день:  {s.daily_halt_count:>12d}",
            f"  DD-стопов:       {s.dd_halt_count:>12d}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Решение об открытии позиции (возвращается can_open)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class OpenDecision:
    allowed:           bool
    reason:            str   = "OK"
    position_size_rub: float = 0.0
    risk_pct_used:     float = 0.0

    def __bool__(self):
        return self.allowed
