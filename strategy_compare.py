"""
Сравнение торговых стратегий на исторических данных MOEX.

Стратегии:
  1. Elliott Wave  — текущая реализация (lookback=7, ATR×0.5)
  2. EMA Cross     — пересечение EMA20 / EMA50 (классика тренда)
  3. RSI Mean Rev  — покупка на перепроданности RSI<30, выход RSI>70
  4. Breakout ATR  — пробой максимума N свечей + ATR-стоп
  5. MACD Signal   — сигнальная линия MACD пересекает нулевую снизу вверх

Только длинные позиции (LONG-only), одна позиция одновременно.
Комиссия 0.05% за сделку, проскальзывание 0.01% цены.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Загрузка данных
# --------------------------------------------------------------------------

def load_csv(path: str | Path) -> pd.DataFrame:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    sep = ";" if raw.count(";") > raw.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower().replace("<", "").replace(">", "") for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        date_str = df["date"].astype(str).str.strip()
        time_str = df["time"].astype(str).str.zfill(6)
        sample = date_str.iloc[0]
        fmt = "%d/%m/%y %H%M%S" if "/" in sample else "%Y%m%d %H%M%S"
        df["datetime"] = pd.to_datetime(date_str + " " + time_str, format=fmt, errors="coerce")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float).dropna()
    return df

# --------------------------------------------------------------------------
# Базовые индикаторы
# --------------------------------------------------------------------------

def atr(df, period=14):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(df, fast=12, slow=26, signal=9):
    m = ema(df["close"], fast) - ema(df["close"], slow)
    s = ema(m, signal)
    return m, s

# --------------------------------------------------------------------------
# Движок бэктеста (универсальный)
# --------------------------------------------------------------------------

COMMISSION = 0.0005   # 0.05%
SLIPPAGE   = 0.0001   # 0.01%

@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    sl: float
    tp: float
    exit_idx: int = 0
    exit_price: float = 0.0
    reason: str = ""

    @property
    def pnl(self):
        return self.exit_price - self.entry_price - (self.entry_price + self.exit_price) * COMMISSION

    @property
    def win(self):
        return self.pnl > 0

@dataclass
class Result:
    name: str
    trades: list[Trade] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    initial: float = 100_000.0

    @property
    def n(self): return len(self.trades)

    @property
    def wins(self): return sum(1 for t in self.trades if t.win)

    @property
    def win_rate(self): return self.wins / self.n if self.n else 0.0

    @property
    def total_pnl(self): return sum(t.pnl for t in self.trades)

    @property
    def total_return(self): return self.total_pnl / self.initial

    @property
    def max_dd(self):
        eq = pd.Series(self.equity)
        if len(eq) < 2: return 0.0
        roll = eq.cummax()
        dd = (eq - roll) / roll
        return float(dd.min())

    @property
    def sharpe(self):
        eq = pd.Series(self.equity)
        r = eq.pct_change().dropna()
        if len(r) < 2 or r.std() == 0: return 0.0
        # Годовой: дневные данные ~252, часовые ~252*7
        return float(r.mean() / r.std() * math.sqrt(252))

    @property
    def avg_win(self):
        wins = [t.pnl for t in self.trades if t.win]
        return sum(wins)/len(wins) if wins else 0.0

    @property
    def avg_loss(self):
        losses = [t.pnl for t in self.trades if not t.win]
        return sum(losses)/len(losses) if losses else 0.0

    @property
    def profit_factor(self):
        g = sum(t.pnl for t in self.trades if t.win)
        l = abs(sum(t.pnl for t in self.trades if not t.win))
        return g / l if l > 0 else float("inf")

    def row(self):
        return {
            "Стратегия": self.name,
            "Сделок": self.n,
            "WR%": f"{self.win_rate*100:.0f}%",
            "Net P&L": f"{self.total_pnl:+.0f}",
            "Return%": f"{self.total_return*100:.1f}%",
            "MaxDD%": f"{self.max_dd*100:.1f}%",
            "Sharpe": f"{self.sharpe:.2f}",
            "PF": f"{self.profit_factor:.2f}" if self.profit_factor != float('inf') else "inf",
            "Avg Win": f"{self.avg_win:+.1f}",
            "Avg Loss": f"{self.avg_loss:+.1f}",
        }


def run_engine(df: pd.DataFrame, signals_fn, name: str, initial=100_000.0) -> Result:
    """
    signals_fn(df) -> pd.DataFrame с колонками: signal(bool), sl(float), tp(float)
    signal=True означает сигнал на покупку на открытии следующей свечи.
    """
    sig_df = signals_fn(df)
    result = Result(name=name, initial=initial)
    equity = initial
    open_trade: Optional[Trade] = None
    prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    n = len(df)

    for i in range(1, n):
        # Обновляем equity даже без позиции
        result.equity.append(equity)

        # Проверяем открытую позицию
        if open_trade is not None:
            # Сначала проверяем SL/TP на текущей свече
            lo, hi = lows[i], highs[i]
            closed = False
            if lo <= open_trade.sl:
                ep = open_trade.sl * (1 - SLIPPAGE)
                open_trade.exit_idx = i
                open_trade.exit_price = ep
                open_trade.reason = "SL"
                equity += open_trade.pnl
                result.trades.append(open_trade)
                open_trade = None
                closed = True
            elif hi >= open_trade.tp:
                ep = open_trade.tp * (1 - SLIPPAGE)
                open_trade.exit_idx = i
                open_trade.exit_price = ep
                open_trade.reason = "TP"
                equity += open_trade.pnl
                result.trades.append(open_trade)
                open_trade = None
                closed = True

        # Ищем новый сигнал (если нет позиции и на предыдущей свече был сигнал)
        if open_trade is None:
            prev = i - 1
            if prev >= 0 and sig_df["signal"].iloc[prev]:
                sl = float(sig_df["sl"].iloc[prev])
                tp = float(sig_df["tp"].iloc[prev])
                entry = opens[i] * (1 + SLIPPAGE)
                if sl < entry < tp:  # валидный сигнал
                    open_trade = Trade(entry_idx=i, entry_price=entry, sl=sl, tp=tp)

    # Закрываем по последней цене
    if open_trade is not None:
        ep = prices[-1] * (1 - SLIPPAGE)
        open_trade.exit_idx = n - 1
        open_trade.exit_price = ep
        open_trade.reason = "END"
        equity += open_trade.pnl
        result.trades.append(open_trade)

    return result


# ==========================================================================
# СТРАТЕГИИ
# ==========================================================================

# --------------------------------------------------------------------------
# 1. EMA Cross: покупка при EMA20 > EMA50 (пересечение снизу вверх)
#    SL: последний локальный минимум (ATR-буфер)
#    TP: entry + 2 * (entry - SL)  → R:R = 1:2
# --------------------------------------------------------------------------
def strat_ema_cross(df: pd.DataFrame) -> pd.DataFrame:
    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    at = atr(df, 14)
    cross_up = (e20 > e50) & (e20.shift(1) <= e50.shift(1))
    sl_price = df["low"].rolling(10).min() - at * 0.5
    tp_price = df["close"] + 2 * (df["close"] - sl_price)
    return pd.DataFrame({"signal": cross_up, "sl": sl_price, "tp": tp_price})


# --------------------------------------------------------------------------
# 2. RSI Mean Reversion: покупка при RSI < 30 и разворот вверх
#    SL: low свечи сигнала − ATR
#    TP: entry + 2 × ATR
# --------------------------------------------------------------------------
def strat_rsi_reversal(df: pd.DataFrame) -> pd.DataFrame:
    r = rsi(df, 14)
    at = atr(df, 14)
    signal = (r < 30) & (r > r.shift(1))        # RSI < 30 и начал расти
    sl_price = df["low"] - at
    tp_price = df["close"] + 2 * at
    return pd.DataFrame({"signal": signal, "sl": sl_price, "tp": tp_price})


# --------------------------------------------------------------------------
# 3. Breakout N-bar: покупка при пробое максимума последних N свечей
#    SL: low последних N свечей − ATR×0.5
#    TP: entry + 2 × (entry − SL)
# --------------------------------------------------------------------------
def strat_breakout(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    at = atr(df, 14)
    high_n = df["high"].rolling(n).max().shift(1)   # shift: не заглядываем вперёд
    low_n  = df["low"].rolling(n).min().shift(1)
    signal = df["close"] > high_n
    sl_price = low_n - at * 0.5
    tp_price = df["close"] + 2 * (df["close"] - sl_price)
    # Убираем сигналы вплотную друг к другу (не ранее 5 свечей после предыдущего)
    sig = signal.copy()
    last = -999
    for i in range(len(sig)):
        if sig.iloc[i]:
            if i - last < 5:
                sig.iloc[i] = False
            else:
                last = i
    return pd.DataFrame({"signal": sig, "sl": sl_price, "tp": tp_price})


# --------------------------------------------------------------------------
# 4. MACD: пересечение MACD > Signal снизу вверх (оба ниже нуля → разворот)
#    SL: low за 10 свечей − ATR×0.3
#    TP: entry + 2.5 × (entry − SL)
# --------------------------------------------------------------------------
def strat_macd(df: pd.DataFrame) -> pd.DataFrame:
    m, s = macd(df)
    at = atr(df, 14)
    cross_up = (m > s) & (m.shift(1) <= s.shift(1))
    signal = cross_up & (m < 0)   # пересечение в зоне ниже нуля
    sl_price = df["low"].rolling(10).min() - at * 0.3
    tp_price = df["close"] + 2.5 * (df["close"] - sl_price)
    return pd.DataFrame({"signal": signal, "sl": sl_price, "tp": tp_price})


# --------------------------------------------------------------------------
# 5. Supertrend: классический индикатор тренда
#    Покупка при смене тренда вниз→вверх
#    SL: нижняя линия Supertrend
#    TP: entry + 3 × ATR
# --------------------------------------------------------------------------
def strat_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    at = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * at
    lower = hl2 - mult * at

    close = df["close"].values
    n = len(close)
    trend = np.ones(n)        # 1 = up, -1 = down
    final_upper = upper.values.copy()
    final_lower = lower.values.copy()

    for i in range(1, n):
        # Нижняя полоса (поддержка при up-тренде)
        if lower.iloc[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower.iloc[i]
        else:
            final_lower[i] = final_lower[i-1]

        # Верхняя полоса (сопротивление при down-тренде)
        if upper.iloc[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper.iloc[i]
        else:
            final_upper[i] = final_upper[i-1]

        # Тренд
        if trend[i-1] == -1 and close[i] > final_upper[i-1]:
            trend[i] = 1
        elif trend[i-1] == 1 and close[i] < final_lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    trend_s = pd.Series(trend, index=df.index)
    lower_s = pd.Series(final_lower, index=df.index)

    signal = (trend_s == 1) & (trend_s.shift(1) == -1)
    at14 = atr(df, 14)
    sl_price = lower_s - at14 * 0.2
    tp_price = df["close"] + 3 * at14
    return pd.DataFrame({"signal": signal, "sl": sl_price, "tp": tp_price})


# --------------------------------------------------------------------------
# 6. Donchian Breakout (Turtle Trading): покупка при пробое max за N свечей,
#    выход при пробое min за N/2 свечей или TP 2×ATR
# --------------------------------------------------------------------------
def strat_donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    at = atr(df, 14)
    # Сигнал: close пробивает max(high) за последние N свечей
    dc_high = df["high"].rolling(n).max().shift(1)
    dc_low  = df["low"].rolling(n // 2).min().shift(1)
    signal = df["close"] > dc_high
    # SL: Donchian нижняя линия (N/2)
    sl_price = dc_low - at * 0.3
    tp_price = df["close"] + 2.5 * at
    # Дедупликация сигналов (не чаще раз в 3 свечи)
    sig = signal.copy()
    last = -999
    for i in range(len(sig)):
        if sig.iloc[i]:
            if i - last < 3:
                sig.iloc[i] = False
            else:
                last = i
    return pd.DataFrame({"signal": sig, "sl": sl_price, "tp": tp_price})


# --------------------------------------------------------------------------
# 7. Elliott Wave (упрощённая версия из проекта)
# --------------------------------------------------------------------------
def strat_elliott(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from analysis.swing_detector import find_swings
        from analysis.wave_analyzer import find_impulse
        from analysis.indicators import add_indicators
    except ImportError:
        return pd.DataFrame({"signal": [False]*len(df), "sl": df["low"], "tp": df["high"]})

    df_ind = add_indicators(df.copy())
    at = atr(df, 14)
    signals = [False] * len(df)
    sl_arr = df["low"].values.copy()
    tp_arr = df["high"].values.copy()

    WINDOW = 150
    BARS_LIMIT = 20

    for i in range(WINDOW, len(df)):
        window = df_ind.iloc[max(0, i - WINDOW): i + 1].copy().reset_index(drop=True)
        try:
            swings = find_swings(window, lookback=7, atr_multiplier=0.5)
            if len(swings) < 6:
                continue
            impulses = find_impulse(swings, fib_tolerance=0.25)
            if not impulses:
                continue
            imp = impulses[0]
            if imp.confidence_score < 0.25 or imp.direction != "UP":
                continue
            p = imp.points
            n_rows = len(window)
            if p[5].idx >= n_rows - 1:
                continue
            bars_since = n_rows - 1 - p[5].idx
            if bars_since > BARS_LIMIT:
                continue
            cur_close = float(window.iloc[-1]["close"])
            high3 = p[3].price
            low4 = p[4].price
            high5 = p[5].price
            low0 = p[0].price
            if cur_close <= high3:
                continue
            sl = low4 - max((high3 - low4) * 0.05, 0.001 * high3)
            tp = high3 + (high5 - low0) * 1.618
            if sl < cur_close < tp:
                signals[i] = True
                sl_arr[i] = sl
                tp_arr[i] = tp
        except Exception:
            continue

    return pd.DataFrame({"signal": signals, "sl": sl_arr, "tp": tp_arr}, index=df.index)


# ==========================================================================
# ЗАПУСК
# ==========================================================================

STRATEGIES = [
    ("EMA Cross",    strat_ema_cross),
    ("RSI Reversal", strat_rsi_reversal),
    ("Breakout-20",  strat_breakout),
    ("MACD",         strat_macd),
    ("Supertrend",   strat_supertrend),
    ("Donchian-20",  strat_donchian),
    ("Elliott Wave", strat_elliott),
]

FILES = [
    ("ROSN", "ROSN_220122_260320.csv"),
    ("YDEX", "YDEX_220122_260320.csv"),
    ("LKOH", "LKOH_220122_260320.csv"),
    ("MGNT", "MGNT_220122_260320.csv"),
    ("SBER", "SBER_251222_260320.csv"),
]

def run_all():
    all_rows = []
    summary_by_strategy: dict[str, list[Result]] = {s[0]: [] for s in STRATEGIES}

    for ticker, fname in FILES:
        path = Path(fname)
        if not path.exists():
            print(f"  [!] Файл не найден: {fname}")
            continue
        df = load_csv(path)
        print(f"\n{'='*70}")
        print(f"  {ticker}   |   {len(df)} свечей   |   {df.index[0].date()} .. {df.index[-1].date()}")
        print(f"{'='*70}")

        rows = []
        for strat_name, strat_fn in STRATEGIES:
            try:
                result = run_engine(df, strat_fn, strat_name)
                rows.append(result.row())
                summary_by_strategy[strat_name].append(result)
            except Exception as e:
                rows.append({"Стратегия": strat_name, "Сделок": "ERR", "WR%": str(e)[:40]})

        # Таблица по тикеру
        col_w = [14, 7, 5, 9, 8, 7, 7, 5, 8, 8]
        cols = ["Стратегия","Сделок","WR%","Net P&L","Return%","MaxDD%","Sharpe","PF","Avg Win","Avg Loss"]
        header = "  ".join(c.ljust(w) for c, w in zip(cols, col_w))
        print(header)
        print("-" * len(header))
        for row in rows:
            line = "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(cols, col_w))
            print(line)

        all_rows.extend(rows)

    # Итоговая таблица: среднее по всем тикерам
    print(f"\n\n{'='*70}")
    print("  СВОДНАЯ ТАБЛИЦА — среднее по всем тикерам")
    print(f"{'='*70}")
    cols2 = ["Стратегия", "Всего сделок", "Avg WR%", "Avg Return%", "Avg Sharpe", "Avg PF", "Avg MaxDD%"]
    print("  ".join(c.ljust(14) for c in cols2))
    print("-" * 90)

    summary_rows = []
    for strat_name, res_list in summary_by_strategy.items():
        valid = [r for r in res_list if r.n > 0]
        if not valid:
            summary_rows.append({"name": strat_name, "trades": 0, "wr": 0,
                                  "ret": 0, "sharpe": -99, "pf": 0, "dd": 0})
            continue
        total_trades = sum(r.n for r in valid)
        avg_wr = sum(r.win_rate for r in valid) / len(valid)
        avg_ret = sum(r.total_return for r in valid) / len(valid)
        avg_sharpe = sum(r.sharpe for r in valid) / len(valid)
        pfs = [r.profit_factor for r in valid if r.profit_factor != float("inf")]
        avg_pf = sum(pfs) / len(pfs) if pfs else 999.0
        avg_dd = sum(r.max_dd for r in valid) / len(valid)
        summary_rows.append({"name": strat_name, "trades": total_trades, "wr": avg_wr,
                              "ret": avg_ret, "sharpe": avg_sharpe, "pf": avg_pf, "dd": avg_dd})

    summary_rows.sort(key=lambda x: x["sharpe"], reverse=True)

    for sr in summary_rows:
        row = [sr["name"], str(sr["trades"]),
               f"{sr['wr']*100:.0f}%" if sr['trades'] else "—",
               f"{sr['ret']*100:.2f}%" if sr['trades'] else "—",
               f"{sr['sharpe']:.2f}" if sr['trades'] else "—",
               f"{sr['pf']:.2f}" if sr['trades'] else "—",
               f"{sr['dd']*100:.1f}%" if sr['trades'] else "—"]
        print("  ".join(str(v).ljust(14) for v in row))

    best = summary_rows[0]
    print()
    print(f"  >>> ПОБЕДИТЕЛЬ по Sharpe: {best['name']}")
    print(f"      Avg Sharpe={best['sharpe']:.2f}  WR={best['wr']*100:.0f}%  "
          f"Return={best['ret']*100:.2f}%  PF={best['pf']:.2f}  MaxDD={best['dd']*100:.1f}%")
    print()
    print("Метрики:")
    print("  WR%     = win rate (доля прибыльных сделок)")
    print("  Return% = итоговая доходность на начальный капитал")
    print("  Sharpe  = Sharpe Ratio (норм. на sqrt(252))")
    print("  PF      = Profit Factor (gross прибыль / gross убыток)")
    print("  MaxDD%  = максимальная просадка капитала")


if __name__ == "__main__":
    run_all()
