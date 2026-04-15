"""
Бэктест-движок на исторических данных (CSV или DataFrame).

Логика:
1. Для каждого окна свечей запускаем swing_detector → wave_analyzer.
2. При нахождении восходящего импульса: ждём пробой high₃ как сигнал входа.
3. Открываем позицию с SL = low₄ и TP = entry + (high₅ - low₀) * 1.618.
4. Моделируем комиссию и проскальзывание.
5. Выводим сводные метрики.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from analysis.indicators import add_indicators
from analysis.swing_detector import find_swings
from analysis.wave_analyzer import find_impulse
from config import (
    ATR_MULTIPLIER,
    ATR_PERIOD,
    COMMISSION_RATE,
    FIB_TOLERANCE,
    LOOKBACK,
    RISK_PERCENT,
    SLIPPAGE_TICKS,
)


@dataclass
class BacktestTrade:
    entry_idx: int
    entry_price: float
    stop_loss: float
    take_profit: float
    direction: str
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    commission: float = 0.0
    pnl: float = 0.0
    status: str = "OPEN"

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    trades: list[BacktestTrade]
    equity_curve: pd.Series
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int

    def summary(self) -> str:
        lines = [
            f"Сделок всего  : {self.total_trades}",
            f"Прибыльных    : {self.winning_trades} ({self.win_rate * 100:.1f}%)",
            f"Убыточных     : {self.losing_trades}",
            f"Доходность    : {self.total_return * 100:.2f}%",
            f"Max Drawdown  : {self.max_drawdown * 100:.2f}%",
            f"Sharpe Ratio  : {self.sharpe_ratio:.3f}",
        ]
        return "\n".join(lines)


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Загружает CSV с ММВБ (Финам-формат или generic OHLCV).

    Поддерживает два варианта:
    - Финам с разделителем «;»: <TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>
      дата DD/MM/YY или YYYYMMDD, время HHMMSS
    - Generic CSV с запятой: datetime,open,high,low,close,volume
    """
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    sep = ";" if raw.count(";") > raw.count(",") else ","

    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower().replace("<", "").replace(">", "") for c in df.columns]

    # Финам-формат: колонки date и time (возможно ticker, per, vol)
    if "date" in df.columns and "time" in df.columns:
        date_str = df["date"].astype(str).str.strip()
        time_str = df["time"].astype(str).str.zfill(6)

        # Определяем формат даты: DD/MM/YY или YYYYMMDD
        sample = date_str.iloc[0]
        if "/" in sample:
            # DD/MM/YY → DD/MM/YYYY (Финам экспортирует двузначный год)
            df["datetime"] = pd.to_datetime(
                date_str + " " + time_str,
                format="%d/%m/%y %H%M%S",
                errors="coerce",
            )
        else:
            df["datetime"] = pd.to_datetime(
                date_str + " " + time_str,
                format="%Y%m%d %H%M%S",
                errors="coerce",
            )
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        raise ValueError("CSV не содержит колонок datetime или (date + time)")

    # Нормализуем названия: vol → volume
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})

    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.dropna()
    return df


class BacktestEngine:
    """
    Пошаговый бэктест: анализируем историю слева направо,
    НЕ заглядывая в будущее (no look-ahead bias).

    min_window : минимальное количество свечей для первого анализа.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        risk_percent: float = RISK_PERCENT,
        commission_rate: float = COMMISSION_RATE,
        slippage_ticks: int = SLIPPAGE_TICKS,
        lookback: int = LOOKBACK,
        atr_period: int = ATR_PERIOD,
        atr_multiplier: float = ATR_MULTIPLIER,
        fib_tolerance: float = FIB_TOLERANCE,
        min_window: int = 30,
        signal_window: int = 200,   # сколько свечей смотрим назад для анализа свингов
        bars_since_limit: int = 48, # сигнал устаревает через N свечей после p5 (48ч = ~5 дней)
        min_confidence: float = 0.25,
    ) -> None:
        self.df = add_indicators(df, atr_period=atr_period).reset_index(drop=False)
        self.capital = initial_capital
        self.risk_percent = risk_percent
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.fib_tolerance = fib_tolerance
        self.min_window = min_window

        self.signal_window = signal_window
        self.bars_since_limit = bars_since_limit
        self.min_confidence = min_confidence

        self._trades: list[BacktestTrade] = []
        self._equity: list[float] = []
        self._open_trade: Optional[BacktestTrade] = None
        self._last_signal_idx: int = -1     # защита от дублирования сигналов
        self._used_entries: set[float] = set()  # уже использованные уровни входа

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        df = self.df
        n = len(df)
        equity = self.capital

        for i in range(self.min_window, n):
            row = df.iloc[i]
            high_i = float(row["high"])
            low_i = float(row["low"])
            close_i = float(row["close"])

            # 1. Проверяем открытую позицию
            if self._open_trade is not None:
                trade = self._open_trade
                if trade.direction == "UP":
                    if low_i <= trade.stop_loss:
                        equity = self._close_trade(trade, trade.stop_loss, i, equity, "SL")
                    elif high_i >= trade.take_profit:
                        equity = self._close_trade(trade, trade.take_profit, i, equity, "TP")
                # DOWN-позиции не используются (только лонг)

            # 2. Если нет открытой позиции — ищем сигнал
            if self._open_trade is None and i != self._last_signal_idx:
                window_df = df.iloc[max(0, i - self.signal_window) : i + 1].copy()
                signal = self._find_signal(window_df, i)
                if signal is not None:
                    entry, sl, tp, direction = signal
                    # Защита от повторного входа на один и тот же уровень
                    entry_key = round(entry, 4)
                    if entry_key in self._used_entries:
                        signal = None
                if signal is not None:
                    entry, sl, tp, direction = signal
                    slippage = self._get_tick_size(df) * self.slippage_ticks
                    entry_actual = entry + slippage if direction == "UP" else entry - slippage

                    trade = BacktestTrade(
                        entry_idx=i,
                        entry_price=entry_actual,
                        stop_loss=sl,
                        take_profit=tp,
                        direction=direction,
                    )
                    commission = entry_actual * self.commission_rate
                    trade.commission += commission
                    equity -= commission
                    self._open_trade = trade
                    self._last_signal_idx = i
                    self._used_entries.add(round(entry, 4))

            self._equity.append(equity)

        # Закрываем незакрытые позиции по последней цене
        if self._open_trade is not None:
            last_close = float(df.iloc[-1]["close"])
            equity = self._close_trade(self._open_trade, last_close, n - 1, equity, "END")

        return self._compute_result(equity)

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _find_signal(
        self, window_df: pd.DataFrame, current_idx: int
    ) -> Optional[tuple[float, float, float, str]]:
        """
        Возвращает (entry, sl, tp, direction) или None.

        Логика сигнала:
        - Находим все завершённые импульсы в окне.
        - Ищем импульс, у которого p5 (последняя точка) НЕ является
          последней свечой окна (т.е. структура завершилась раньше).
        - Сигнал BUY: текущая цена закрытия пробивает high₃ импульса вверх.
        - Вход: open следующей свечи (имитируем лимитную заявку по high₃).
        - SL: low₄ минус буфер 5% от (high₃ - low₄).
        - TP: entry + (high₅ - low₀) * 1.618.
        """
        try:
            swings = find_swings(
                window_df,
                lookback=self.lookback,
                atr_period=self.atr_period,
                atr_multiplier=self.atr_multiplier,
            )
            if len(swings) < 6:
                return None

            impulses = find_impulse(swings, fib_tolerance=self.fib_tolerance)
            if not impulses:
                return None

            current_close = float(window_df.iloc[-1]["close"])
            current_high = float(window_df.iloc[-1]["high"])
            current_low = float(window_df.iloc[-1]["low"])
            n_rows = len(window_df)

            for imp in impulses:
                if imp.confidence_score < self.min_confidence:
                    continue

                p = imp.points
                # p5 должна быть в истории (не последняя строка окна),
                # чтобы не было look-ahead bias
                if p[5].idx >= n_rows - 1:
                    continue

                if imp.direction == "UP":
                    high3 = p[3].price
                    low4 = p[4].price
                    low0 = p[0].price
                    high5 = p[5].price

                    # Сигнал: текущая свеча закрылась выше high₃
                    if current_close <= high3:
                        continue

                    # Дополнительная проверка: сигнал не устарел
                    bars_since_p5 = n_rows - 1 - p[5].idx
                    if bars_since_p5 > self.bars_since_limit:
                        continue

                    entry = high3
                    sl = low4 - max((high3 - low4) * 0.05, 0.001 * high3)
                    tp = entry + (high5 - low0) * 1.618
                    if tp <= entry or sl >= entry:
                        continue
                    return entry, sl, tp, "UP"

                # DOWN-сигналы не используются (только лонг)

        except Exception:
            return None

        return None

    def _close_trade(
        self,
        trade: BacktestTrade,
        exit_price: float,
        idx: int,
        equity: float,
        reason: str,
    ) -> float:
        slippage = self._get_tick_size(self.df) * self.slippage_ticks
        if trade.direction == "UP":
            exit_actual = exit_price - slippage
            pnl = exit_actual - trade.entry_price
        else:
            exit_actual = exit_price + slippage
            pnl = trade.entry_price - exit_actual

        commission = exit_actual * self.commission_rate
        trade.exit_idx = idx
        trade.exit_price = exit_actual
        trade.exit_reason = reason
        trade.commission += commission
        trade.pnl = pnl - trade.commission
        trade.status = "CLOSED"

        equity += trade.pnl
        self._trades.append(trade)
        self._open_trade = None
        return equity

    def _get_tick_size(self, df: pd.DataFrame) -> float:
        avg_price = float(df["close"].mean())
        return avg_price * 0.0001  # ~0.01% от цены как тик по умолчанию

    def _compute_result(self, final_equity: float) -> BacktestResult:
        initial = self.capital
        equity_series = pd.Series(self._equity, name="equity")

        total_return = (final_equity - initial) / initial if initial > 0 else 0.0

        # Max drawdown
        roll_max = equity_series.cummax()
        drawdown = (equity_series - roll_max) / roll_max
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Sharpe (дневные изменения капитала)
        returns = equity_series.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * math.sqrt(252))
        else:
            sharpe = 0.0

        closed = [t for t in self._trades if t.status == "CLOSED"]
        wins = [t for t in closed if t.is_win]
        losses = [t for t in closed if not t.is_win]
        win_rate = len(wins) / len(closed) if closed else 0.0

        return BacktestResult(
            trades=closed,
            equity_curve=equity_series,
            total_return=round(total_return, 6),
            max_drawdown=round(max_dd, 6),
            sharpe_ratio=round(sharpe, 4),
            win_rate=round(win_rate, 4),
            total_trades=len(closed),
            winning_trades=len(wins),
            losing_trades=len(losses),
        )
