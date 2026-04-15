"""
Стратегия 2: MTF Trend Confirmation v1 (MTF_V1)
================================================
Лучший результат: +11.8% годовых, MaxDD -13.8%, WR 47.4%
Период теста: 2022–2026, 11 тикеров с 1H данными

Логика:
  - Таймфреймы: 1H, 4H, 8H, 12H, D
  - Сигнал: score ≥ 3 (из 5 TF подтверждают бычий тренд)
  - SL: последний фрактальный минимум Б.Вильямса на дневном TF
  - Выход: Trailing Stop 2×ATR(D) + TIME через 30 дней
  - Фильтр ZigZag волн Эллиотта (не входим на 5-й волне)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Optional
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.indicators import ema, atr, rsi, adx, volume_ratio, bw_fractals, zigzag

STRATEGY_ID   = "mtf_v1"
STRATEGY_NAME = "MTF Trend Confirmation v1"
STRATEGY_DESC = (
    "Многотаймфреймная стратегия. Вход только когда 3+ таймфрейма "
    "(1H, 4H, 8H, 12H, D) одновременно подтверждают восходящий тренд. "
    "SL устанавливается по фракталу Билла Вильямса на дневном графике. "
    "Работает на тикерах с часовыми данными."
)
VERSION = "1.0.0"


@dataclass
class MTFv1Config:
    min_score: int = 3              # минимум TF с бычьим сигналом
    trail_mult: float = 2.0         # ATR-множитель trailing stop
    max_hold_days: int = 30         # максимальное удержание
    fractal_n: int = 2              # параметр фракталов Б.Вильямса
    zigzag_dev: float = 5.0         # % отклонение ZigZag
    max_sl_pct: float = 0.20        # максимальный SL от цены (ограничитель)
    # 4 позиции × 25% = 100% капитала — полное использование без простоя кэша
    max_positions: int = 4
    risk_pct: float = 0.25
    # БКС «Инвестор»: покупка 0%, продажа 0.30% (по тарифному плану клиента)
    # Проскальзывание MOEX часовой ТФ: ~0.10%
    commission_buy:  float = 0.0000  # покупка: 0% (БКС Инвестор)
    commission_sell: float = 0.0030  # продажа: 0.30% (БКС Инвестор)
    slippage: float        = 0.0010  # 0.10%
    commission: float      = 0.0015  # среднее — только для отображения в UI
    reinvest: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MTFv1Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


TF_LIST = ["1H", "4H", "8H", "12H", "D"]

# Пороги сигнала на каждом TF
TF_PARAMS = {
    "1H":  dict(atr_bo=1.2, rsi_lo=50, rsi_hi=80, adx=18, vol=1.2),
    "4H":  dict(atr_bo=0.8, rsi_lo=48, rsi_hi=82, adx=16, vol=1.1),
    "8H":  dict(atr_bo=0.5, rsi_lo=47, rsi_hi=83, adx=15, vol=1.0),
    "12H": dict(atr_bo=0.3, rsi_lo=46, rsi_hi=84, adx=14, vol=1.0),
    "D":   dict(atr_bo=0.2, rsi_lo=45, rsi_hi=85, adx=13, vol=1.0),
}


def _tf_signal(df: pd.DataFrame, tf: str) -> pd.Series:
    """Бычий сигнал на одном таймфрейме."""
    if df is None or len(df) < 210:
        return pd.Series(dtype=float)
    p = TF_PARAMS[tf]
    c = df["close"]
    at14 = atr(df, 14)
    e200 = ema(c, 200)
    r14  = rsi(c, 14)
    a14  = adx(df, 14)
    vr   = volume_ratio(df, 20)
    bm   = (c - c.shift(1)).clip(lower=0)
    sig = (
        (c > e200) & (bm >= p["atr_bo"] * at14) &
        (r14 >= p["rsi_lo"]) & (r14 <= p["rsi_hi"]) &
        (a14 >= p["adx"]) & (vr >= p["vol"])
    )
    return sig.astype(float)


def prepare_ticker(
    ticker: str,
    tf_data: dict[str, pd.DataFrame],
) -> Optional[dict]:
    """
    Подготавливает все необходимые серии для тикера.
    Возвращает словарь с выровненными на 1H серии или None.
    """
    if "1H" not in tf_data or "D" not in tf_data:
        return None

    df_1h = tf_data["1H"]
    df_d  = tf_data["D"]

    # MTF score (сумма сигналов по всем TF)
    total_score = pd.Series(0.0, index=df_1h.index)
    for tf in TF_LIST:
        df_tf = tf_data.get(tf)
        if df_tf is None:
            continue
        sig = _tf_signal(df_tf, tf)
        if sig.empty:
            continue
        aligned = sig.reindex(df_1h.index, method="ffill").ffill().fillna(0)
        total_score += aligned

    # Фракталы на D → выровнены на 1H
    fh_d, fl_d = bw_fractals(df_d, 2)
    frac_h = df_d["high"].where(fh_d).ffill()
    frac_l = df_d["low"].where(fl_d).ffill()
    frac_h_1h = frac_h.reindex(df_1h.index, method="ffill").ffill()
    frac_l_1h = frac_l.reindex(df_1h.index, method="ffill").ffill()

    # ATR(D) → выровнен на 1H
    atr_d = atr(df_d, 14)
    atr_d_1h = atr_d.reindex(df_1h.index, method="ffill").ffill()

    # ZigZag направление на D
    zz_d = zigzag(df_d, 5.0)
    zz_diff = (zz_d - zz_d.shift(1)).replace(0, np.nan).ffill().fillna(0)
    zz_dir = ((zz_diff > 0).astype(float) - (zz_diff < 0).astype(float))
    zz_dir_1h = zz_dir.reindex(df_1h.index, method="ffill").ffill().fillna(0)

    return {
        "df_1h":      df_1h,
        "score":      total_score,
        "frac_h":     frac_h_1h,
        "frac_l":     frac_l_1h,
        "atr_d":      atr_d_1h,
        "zz_dir":     zz_dir_1h,
    }


@dataclass
class BtPosition:
    ticker: str; entry_dt: pd.Timestamp; entry_px: float
    shares: float; sl_px: float; trail_sl: float


@dataclass
class BtTrade:
    ticker: str; entry_dt: pd.Timestamp; exit_dt: pd.Timestamp
    entry_px: float; exit_px: float; shares: float
    pnl_rub: float; pnl_pct: float; reason: str; hold_days: float

    def to_dict(self) -> dict:
        return {
            "ticker":     self.ticker,
            "entry_date": str(self.entry_dt)[:16],
            "exit_date":  str(self.exit_dt)[:16],
            "entry_px":   round(self.entry_px, 2),
            "exit_px":    round(self.exit_px, 2),
            "pnl_rub":    round(self.pnl_rub, 2),
            "pnl_pct":    round(self.pnl_pct, 2),
            "reason":     self.reason,
            "hold_days":  round(self.hold_days, 1),
        }


CORR_GROUPS = [
    {"SBER", "T"}, {"LKOH", "ROSN", "NVTK", "GAZP"}, {"NLMK", "MTLR"},
]

def _corr_blocked(ticker, open_set):
    return any(ticker in g and g & open_set for g in CORR_GROUPS)


def run_backtest(
    tf_data_all: dict[str, dict[str, pd.DataFrame]],
    cfg: MTFv1Config,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """
    Портфельный бэктест MTF v1.

    Args:
        tf_data_all: {ticker: {tf: DataFrame}} — данные по каждому тикеру и TF
        cfg: конфигурация
        initial_capital: начальный капитал
    """
    COMM_BUY  = getattr(cfg, 'commission_buy',  cfg.commission)
    COMM_SELL = getattr(cfg, 'commission_sell', cfg.commission)
    SLIP      = cfg.slippage

    # Подготовка данных
    prepared = {}
    for ticker, tfd in tf_data_all.items():
        p = prepare_ticker(ticker, tfd)
        if p:
            prepared[ticker] = p

    if not prepared:
        return {"error": "Нет данных с 1H и D таймфреймами"}

    # Общий timeline
    master = set()
    for p in prepared.values():
        master.update(p["df_1h"].index)
    timeline = sorted(master)

    IDX = {t: {d: i for i, d in enumerate(p["df_1h"].index)}
           for t, p in prepared.items()}

    # Генерируем точки входа
    ENTRY_SIGS: dict[str, pd.Series] = {}
    for ticker, p in prepared.items():
        df_1h = p["df_1h"]
        sig = (p["score"] >= cfg.min_score)
        # ZigZag фильтр: только восходящая фаза
        sig = sig & (p["zz_dir"] > 0)
        # Убираем дубликаты
        sig = sig & ~sig.shift(1).fillna(False)
        ENTRY_SIGS[ticker] = sig

    free_cash = initial_capital
    positions: dict[str, BtPosition] = {}
    trades:    list[BtTrade] = []
    equity = [initial_capital]
    equity_dates = [timeline[0]]
    peak_eq = initial_capital; max_dd = 0.0

    for dt in timeline:
        # ── Закрытие ──────────────────────────────────────────────────────────
        to_close = []
        for ticker, pos in positions.items():
            p   = prepared[ticker]
            idx = IDX[ticker].get(dt)
            if idx is None or idx <= 0 or dt <= pos.entry_dt:
                continue
            df1h = p["df_1h"]
            hi   = float(df1h["high"].iloc[idx])
            lo   = float(df1h["low"].iloc[idx])
            op   = float(df1h["open"].iloc[idx])
            cls_ = float(df1h["close"].iloc[idx])
            at_d = float(p["atr_d"].iloc[idx])
            fl   = float(p["frac_l"].iloc[idx])
            hd   = (dt - pos.entry_dt).total_seconds() / 86400

            # Trailing update
            trail = cls_ - cfg.trail_mult * at_d
            if trail > pos.trail_sl:
                pos.trail_sl = trail
            # SL следует фрактальным минимумам
            if fl > pos.sl_px and fl < cls_:
                pos.sl_px = fl
            curr_sl = max(pos.sl_px, pos.trail_sl)

            reason = exit_px = None
            if lo <= curr_sl:
                reason = "SL_FRACTAL"; exit_px = max(curr_sl * (1-SLIP), lo)
            elif hd >= cfg.max_hold_days:
                reason = "TIME"; exit_px = op * (1-SLIP)

            if reason:
                cash = exit_px * pos.shares * (1 - COMM_SELL)  # продажа 0.30%
                cost = pos.entry_px * pos.shares * (1 + COMM_BUY)   # покупка 0%
                pnl  = cash - cost; pnl_pct = pnl/cost*100
                free_cash += cash
                trades.append(BtTrade(
                    ticker, pos.entry_dt, dt, pos.entry_px, exit_px,
                    pos.shares, pnl, pnl_pct, reason, hd,
                ))
                to_close.append(ticker)
        for t in to_close:
            positions.pop(t, None)

        # ── Входы ─────────────────────────────────────────────────────────────
        for ticker, p in prepared.items():
            if len(positions) >= cfg.max_positions:
                break
            if ticker in positions:
                continue
            idx = IDX[ticker].get(dt)
            if idx is None or idx < 1:
                continue
            if not bool(ENTRY_SIGS[ticker].iloc[idx]):
                continue
            if _corr_blocked(ticker, set(positions.keys())):
                continue

            df1h  = p["df_1h"]
            entry = float(df1h["open"].iloc[idx]) * (1+SLIP)
            fl_v  = float(p["frac_l"].iloc[idx])
            at_d  = float(p["atr_d"].iloc[idx])

            if fl_v <= 0 or (entry - fl_v) / entry > cfg.max_sl_pct:
                fl_v = entry * (1 - cfg.max_sl_pct * 0.75)

            trail_i = entry - cfg.trail_mult * at_d
            init_sl = max(fl_v, trail_i)

            pos_val = sum(
                float(prepared[t]["df_1h"]["close"].iloc[IDX[t].get(dt,-1)]) * pos.shares
                for t, pos in positions.items()
                if IDX[t].get(dt) is not None
            )
            total_cap = (free_cash + pos_val) if cfg.reinvest else initial_capital
            alloc = min(total_cap * cfg.risk_pct, free_cash * 0.98)
            if alloc <= 0:
                continue
            shares = alloc / entry
            cost   = shares * entry * (1 + COMM_BUY)   # покупка 0%
            if cost > free_cash:
                continue

            free_cash -= cost
            positions[ticker] = BtPosition(ticker, dt, entry, shares, init_sl, init_sl)

        # Equity
        pos_val = sum(
            float(prepared[t]["df_1h"]["close"].iloc[IDX[t].get(dt,-1)]) * pos.shares
            for t, pos in positions.items()
            if IDX[t].get(dt) is not None
        )
        eq = free_cash + pos_val
        equity.append(eq); equity_dates.append(dt)
        if eq > peak_eq: peak_eq = eq
        dd = (peak_eq - eq)/peak_eq*100
        if dd > max_dd: max_dd = dd

    # Принудительное закрытие
    last_dt = timeline[-1]
    for ticker, pos in list(positions.items()):
        p   = prepared[ticker]
        cls_ = float(p["df_1h"]["close"].iloc[-1])
        ep   = cls_ * (1 - SLIP)
        cash = ep * pos.shares * (1 - COMM_SELL)   # продажа 0.30%
        cost = pos.entry_px * pos.shares * (1 + COMM_BUY)   # покупка 0%
        free_cash += cash
        trades.append(BtTrade(ticker, pos.entry_dt, last_dt, pos.entry_px, ep,
                              pos.shares, cash-cost, (cash-cost)/cost*100, "FORCED",
                              (last_dt-pos.entry_dt).total_seconds()/86400))

    # Статистика
    final = free_cash
    total_pnl = (final - initial_capital) / initial_capital * 100
    n_days = (timeline[-1] - timeline[0]).days
    ann = ((final/initial_capital)**(365/max(n_days,1))-1)*100
    pnls = np.array([t.pnl_pct for t in trades])
    n_tr = len(trades); n_win = (pnls>0).sum()
    wr = n_win/n_tr*100 if n_tr else 0
    wins = pnls[pnls>0]; losses = pnls[pnls<=0]
    pf = wins.sum()/(-losses.sum()+1e-9) if len(losses) else 99.0
    eq_arr = np.array(equity)
    dr = np.diff(eq_arr)/(eq_arr[:-1]+1e-9)
    sharpe = (dr.mean()/(dr.std()+1e-9))*np.sqrt(252*6.5)

    by_year: dict[str,dict] = {}
    running = initial_capital
    for t in sorted(trades, key=lambda x: x.exit_dt):
        yr = str(t.exit_dt)[:4]; running += t.pnl_rub
        by_year.setdefault(yr, {"start": running - t.pnl_rub, "end": running})
        by_year[yr]["end"] = running

    by_ticker: dict[str,dict] = {}
    for t in trades:
        d = by_ticker.setdefault(t.ticker, {"n":0,"wins":0,"pnl":0.0})
        d["n"]+=1; d["pnl"]+=t.pnl_pct
        if t.pnl_pct>0: d["wins"]+=1

    by_reason: dict[str,dict] = {}
    for t in trades:
        d = by_reason.setdefault(t.reason, {"n":0,"wins":0,"pnl":0.0})
        d["n"]+=1; d["pnl"]+=t.pnl_pct
        if t.pnl_pct>0: d["wins"]+=1

    return {
        "strategy_id": STRATEGY_ID, "strategy_name": STRATEGY_NAME,
        "initial_capital": initial_capital, "final_capital": round(final, 2),
        "total_pnl_pct": round(total_pnl, 2), "annual_return": round(ann, 2),
        "max_drawdown": round(-max_dd, 2), "sharpe": round(sharpe, 3),
        "n_trades": n_tr, "win_rate": round(wr, 1),
        "profit_factor": round(min(pf, 99), 2),
        "avg_win_pct": round(float(wins.mean()), 2) if len(wins) else 0,
        "avg_loss_pct": round(float(losses.mean()), 2) if len(losses) else 0,
        "n_days": n_days, "config": cfg.to_dict(),
        "equity_curve": [round(e, 2) for e in equity],
        "equity_dates": [str(d)[:16] for d in equity_dates],
        "trades": [t.to_dict() for t in trades],
        "by_year": {
            yr: {"start": round(v["start"],2), "end": round(v["end"],2),
                 "pnl": round(v["end"]-v["start"],2),
                 "pnl_pct": round((v["end"]-v["start"])/v["start"]*100,1)}
            for yr, v in by_year.items()
        },
        "by_ticker": {
            t: {"n":v["n"],"wr":round(v["wins"]/v["n"]*100,1) if v["n"] else 0,
                "total_pnl":round(v["pnl"],2),"avg_pnl":round(v["pnl"]/v["n"],2) if v["n"] else 0}
            for t,v in sorted(by_ticker.items(), key=lambda x:-x[1]["pnl"])
        },
        "by_reason": {
            r: {"n":v["n"],"wr":round(v["wins"]/v["n"]*100,1) if v["n"] else 0,
                "avg_pnl":round(v["pnl"]/v["n"],2) if v["n"] else 0}
            for r,v in by_reason.items()
        },
    }
