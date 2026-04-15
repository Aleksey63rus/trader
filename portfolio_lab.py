"""
Портфельная симуляция топ стратегий из strategies_lab.py.
Использует правильный расчёт P&L через shares (как в pro_portfolio_v2).
Начальный капитал: 100,000 руб.
Риск: 5% на сделку, max 5 позиций.
Тест всех 9 стратегий в портфельном режиме.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Optional
import numpy as np
import pandas as pd

# Импортируем все стратегии
from strategies_lab import (
    load_daily, TICKERS, FRACS, LEVELS, COMMISSION, SLIPPAGE, MAX_HOLD
)
from core.risk import RiskManager, RiskParams

INITIAL_CAPITAL = 100_000.0
YEARS           = 4.1
DATA_DIR        = Path("c:/investor/data")

RISK_PARAMS = RiskParams(
    risk_pct             = 0.05,
    max_risk_pct         = 0.06,
    min_risk_pct         = 0.01,
    max_positions        = 5,
    daily_loss_limit_pct = 0.06,
    dd_reduce_threshold  = 0.20,
    dd_halt_threshold    = 0.35,
    dd_lot_multiplier    = 0.5,
    kelly_fraction       = 0.30,
    kelly_max            = 0.07,
)

# Корреляционные группы
CORR_GROUPS = [
    frozenset({"SBER","SBERP","T","VTBR"}),
    frozenset({"LKOH","ROSN","NVTK","GAZP","SNGS","SNGSP"}),
    frozenset({"NLMK","MTLR","CHMF","MAGN","RUAL","GMKN"}),
    frozenset({"MGNT","OZON"}),
    frozenset({"TATN","TATNP"}),
]


@dataclass
class Position:
    ticker:      str
    entry_dt:    object
    entry_i:     int
    entry:       float
    sl:          float
    risk:        float
    pos_rub:     float
    shares:      float
    remaining:   float = 1.0
    partial_pnl: float = 0.0
    tp_hit:      int   = 0


@dataclass
class ClosedTrade:
    ticker:    str
    strategy:  str
    entry_dt:  object
    exit_dt:   object
    entry:     float
    exit:      float
    reason:    str
    pos_rub:   float
    pnl_rub:   float
    pnl_pct:   float
    hold_d:    int
    win:       bool


def corr_group(ticker: str) -> Optional[frozenset]:
    for g in CORR_GROUPS:
        if ticker in g:
            return g
    return None


def portfolio_sim(strategy_name: str,
                  sigs_dict: dict[str, pd.DataFrame],
                  data: dict[str, pd.DataFrame]) -> dict:
    """
    Полная портфельная симуляция одной стратегии.
    sigs_dict: {ticker -> DataFrame с колонками signal, sl, risk}
    """
    rm = RiskManager(INITIAL_CAPITAL, RISK_PARAMS)

    all_dates = sorted(set().union(*[df.index.tolist() for df in data.values()]))
    open_pos: dict[str, Position] = {}
    trades: list[ClosedTrade] = []
    equity = [INITIAL_CAPITAL]
    blocked = defaultdict(int)
    n_tp = len(FRACS)

    for date in all_dates:
        rm.on_day_start()

        # ── Закрытие позиций ──────────────────────────────────────────────
        to_close = []
        for ticker, pos in open_pos.items():
            df = data.get(ticker)
            if df is None or date not in df.index:
                continue
            bar   = df.loc[date]
            hi    = float(bar["high"])
            lo    = float(bar["low"])
            op    = float(bar["open"])
            i_now = df.index.get_loc(date)
            hold  = i_now - pos.entry_i
            lv    = pos.tp_hit

            while lv < n_tp and pos.remaining > 1e-9:
                tp_px = pos.entry + LEVELS[lv] * pos.risk
                if hi >= tp_px:
                    ep   = tp_px * (1 - SLIPPAGE)
                    frac = min(FRACS[lv], pos.remaining)
                    pos.partial_pnl += ((ep - pos.entry)*frac*pos.shares
                                        - (pos.entry+ep)*COMMISSION*frac*pos.shares)
                    pos.remaining -= frac
                    pos.sl = max(pos.sl,
                                 pos.entry*1.001 if lv==0 else
                                 pos.entry + LEVELS[lv-1]*pos.risk*0.90)
                    pos.tp_hit = lv+1; lv = pos.tp_hit
                else:
                    break

            reason = ep_f = None
            if pos.remaining <= 1e-6:
                reason = f"TP{n_tp}"; ep_f = pos.entry + LEVELS[-1]*pos.risk
            elif hold >= MAX_HOLD:
                reason = "TIME"; ep_f = op*(1-SLIPPAGE)
            elif lo <= pos.sl:
                reason = "SL"; ep_f = max(pos.sl*(1-SLIPPAGE), lo)

            if reason:
                rem = pos.remaining
                pnl = (pos.partial_pnl + (ep_f-pos.entry)*rem*pos.shares
                       - (pos.entry+ep_f)*COMMISSION*rem*pos.shares)
                pnl_pct = pnl / pos.pos_rub * 100
                ct = ClosedTrade(
                    ticker=ticker, strategy=strategy_name,
                    entry_dt=pos.entry_dt, exit_dt=date,
                    entry=pos.entry, exit=ep_f, reason=reason,
                    pos_rub=pos.pos_rub, pnl_rub=round(pnl,2),
                    pnl_pct=round(pnl_pct,3), hold_d=hold,
                    win=bool(pnl > 0),
                )
                trades.append(ct)
                rm.on_trade_closed(ticker, pnl, pnl_pct)
                equity.append(rm.state.capital)
                to_close.append(ticker)

        for t in to_close:
            del open_pos[t]

        # ── Новые входы ───────────────────────────────────────────────────
        candidates = []
        for ticker in TICKERS:
            if ticker in open_pos:
                continue
            sig = sigs_dict.get(ticker)
            df  = data.get(ticker)
            if sig is None or df is None or date not in df.index:
                continue
            i_now = df.index.get_loc(date)
            if i_now < 1:
                continue
            prev = sig.iloc[i_now - 1]
            if not bool(prev.get("signal", 0)):
                continue
            score = float(prev.get("score", 1))
            candidates.append((ticker, score, prev, i_now))

        candidates.sort(key=lambda x: -x[1])

        seen_groups: set = set()
        for ticker, score, prev, i_now in candidates:
            # Корреляционный фильтр (выбираем лучший из группы)
            g = corr_group(ticker)
            if g is not None:
                if any(t in open_pos for t in g):
                    blocked[f"CORR({ticker})"] += 1
                    continue
                if g in seen_groups:
                    blocked[f"CORR_PICK({ticker})"] += 1
                    continue
                seen_groups.add(g)

            df = data[ticker]
            if i_now >= len(df):
                continue
            entry = float(df.iloc[i_now]["open"]) * (1+SLIPPAGE)
            sl_v  = float(prev.get("sl", 0))
            risk_v= float(prev.get("risk", 0))
            if sl_v <= 0 or risk_v <= 0 or sl_v >= entry:
                continue

            sl_pct = (entry - sl_v) / entry
            dec = rm.can_open(ticker, sl_pct)
            if not dec.allowed:
                blocked[dec.reason] += 1
                continue

            pos_rub = dec.position_size_rub
            shares  = pos_rub / entry
            open_pos[ticker] = Position(
                ticker=ticker, entry_dt=date, entry_i=i_now,
                entry=entry, sl=sl_v, risk=risk_v,
                pos_rub=pos_rub, shares=shares,
            )
            rm.on_position_opened(ticker)

    # Закрытие остатков
    for ticker, pos in open_pos.items():
        df = data.get(ticker)
        if df is None:
            continue
        ep  = float(df["close"].iloc[-1]) * (1-SLIPPAGE)
        rem = pos.remaining
        pnl = (pos.partial_pnl + (ep-pos.entry)*rem*pos.shares
               - (pos.entry+ep)*COMMISSION*rem*pos.shares)
        pnl_pct = pnl / pos.pos_rub * 100
        ct = ClosedTrade(
            ticker=ticker, strategy=strategy_name,
            entry_dt=pos.entry_dt, exit_dt=df.index[-1],
            entry=pos.entry, exit=ep, reason="END",
            pos_rub=pos.pos_rub, pnl_rub=round(pnl,2),
            pnl_pct=round(pnl_pct,3),
            hold_d=len(df)-1-pos.entry_i, win=bool(pnl>0),
        )
        trades.append(ct)
        rm.on_trade_closed(ticker, pnl, pnl_pct)
        equity.append(rm.state.capital)

    # Расчёт метрик
    cap_final = rm.state.capital
    total_rub = cap_final - INITIAL_CAPITAL
    total_pct = total_rub / INITIAL_CAPITAL * 100
    ann_ret   = ((cap_final/INITIAL_CAPITAL)**(1/YEARS)-1)*100

    eq = np.array(equity)
    pk = np.maximum.accumulate(np.maximum(eq, 1.0))
    max_dd = float(((eq-pk)/pk*100).min())

    wins   = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    wr     = len(wins)/len(trades)*100 if trades else 0
    aw     = float(np.mean([t.pnl_rub for t in wins]))   if wins   else 0
    al     = float(np.mean([t.pnl_rub for t in losses])) if losses else 0
    pf     = (abs(sum(t.pnl_rub for t in wins)) /
              (abs(sum(t.pnl_rub for t in losses))+1e-9))
    exit_cnt = Counter(t.reason for t in trades)

    return {
        "strategy":   strategy_name,
        "cap_final":  cap_final,
        "total_rub":  total_rub,
        "total_pct":  total_pct,
        "ann_ret":    ann_ret,
        "max_dd":     max_dd,
        "trades":     len(trades),
        "wr":         wr,
        "pf":         pf,
        "avg_win":    aw,
        "avg_loss":   al,
        "wl_ratio":   abs(aw/al) if al else 99,
        "exit_cnt":   dict(exit_cnt),
        "blocked":    dict(blocked),
        "trades_list": trades,
        "equity":     equity,
    }


def generate_signals(strategy_fn, df: pd.DataFrame) -> pd.DataFrame:
    """Получить сигналы из функции стратегии и вернуть DataFrame."""
    # Запускаем стратегию в специальном режиме для получения сигналов
    # Хак: нам нужны sl и risk на каждый бар
    from strategies_lab import (
        ema, atr, adx, rsi, macd_hist, obv, supertrend,
        chandelier_exit, heikin_ashi, volume_ratio, er
    )
    c    = df["close"]
    h    = df["high"]
    l    = df["low"]
    at14 = atr(df, 14)
    e200 = ema(c, 200)
    vol_r= volume_ratio(df, 20)
    adx14= adx(df, 14)
    mh   = macd_hist(df)
    rsi14= rsi(c, 14)
    e9   = ema(c, 9)
    e21  = ema(c, 21)
    e55  = ema(c, 55)
    obv_s= obv(df)
    obv_e= ema(obv_s, 21)
    er20 = er(c, 20)
    st   = supertrend(df, 10, 3.0)
    ce   = chandelier_exit(df, 22, 3.0)
    at5  = atr(df, 5)
    sw5  = l.rolling(5).min()

    name = strategy_fn.__name__.replace("strategy_", "")

    if "turtle20" in name:
        h20 = h.rolling(20).max().shift(1)
        sig = (c > h20) & (c > e200) & (vol_r >= 1.3)
        sig = sig & ~sig.shift(1).fillna(False)
        sl  = (c - 2.0*at14).clip(lower=c*0.88)

    elif "turtle55" in name:
        h55 = h.rolling(55).max().shift(1)
        sig = (c > h55) & (c > e200) & (adx14 >= 20) & (vol_r >= 1.2)
        sig = sig & ~sig.shift(1).fillna(False)
        sl  = (c - 2.5*at14).clip(lower=c*0.85)

    elif "atr_breakout" in name:
        bar_move = (c - c.shift(1)).clip(lower=0)
        sig = ((c > e200) & (bar_move >= 1.5*at14) &
               (at5 > at14*0.95) & (rsi14 >= 52) & (rsi14 <= 82) &
               (adx14 >= 22) & (vol_r >= 1.5))
        sig = sig & ~sig.shift(1).fillna(False)
        sl  = (c - 1.8*at14).clip(lower=c*0.90)

    elif "52w" in name:
        h52  = h.rolling(252).max()
        l52  = l.rolling(252).min()
        pos52= (c-l52) / (h52-l52).replace(0, np.nan)
        sig = ((pos52 >= 0.92) & (c > e55) & (e55 > e200) &
               (adx14 >= 20) & (vol_r >= 1.5))
        sig = sig & ~sig.shift(1).fillna(False)
        sl  = (c - 2.0*at14).clip(lower=c*0.88)

    elif "triple_ema" in name:
        cross_up = (e9 > e21) & (e9.shift(1) <= e21.shift(1))
        sig = cross_up & (c > e55) & (e55 > e200) & (mh > 0) & (adx14 >= 20) & (vol_r >= 1.1)
        sl  = (c - 1.8*at14).clip(lower=c*0.90)

    elif "supertrend" in name:
        trend_flip = (st == 1) & (st.shift(1) == -1)
        sig = trend_flip & (c > e200) & (adx14 >= 18) & (mh > 0) & (vol_r >= 1.0)
        sl  = (c - 2.5*at14).clip(lower=c*0.87)

    elif "chandelier" in name:
        obv_bull = obv_s > obv_e
        above_ce = c > ce
        cross_up = (e9 > e21) & (e9.shift(1) <= e21.shift(1))
        sig = (cross_up & above_ce & obv_bull & (c > e200) &
               (rsi14 >= 50) & (rsi14 <= 80) & (vol_r >= 1.1))
        sl  = ce.clip(lower=c*0.88).clip(upper=c*0.95)

    elif "heikin" in name:
        ha      = heikin_ashi(df)
        ha_bull = (ha["ha_close"] > ha["ha_open"])
        streak  = ha_bull & ha_bull.shift(1).fillna(False) & ha_bull.shift(2).fillna(False)
        sig_raw = streak & ~streak.shift(1).fillna(False)
        sig = sig_raw & (c > e55) & (e55 > e200) & (mh > 0) & (adx14 >= 20) & (vol_r >= 1.1)
        sl  = (c - 2.0*at14).clip(lower=c*0.89)

    elif "apex" in name:
        mh_rising = mh > mh.shift(1)
        h20       = h.rolling(20).max().shift(1)
        quality   = ((c > e200) & (e9 > e21) & (e21 > e55) &
                     (adx14 >= 22) & (rsi14 >= 52) & (rsi14 <= 82) &
                     (mh > 0) & mh_rising & (er20 >= 0.30) &
                     (obv_s > obv_e) & (vol_r >= 1.2) & (st == 1))
        trig_a = quality & (c > h20) & (vol_r >= 1.5)
        near   = ((c-e21).abs()/e21.replace(0,np.nan)) <= 0.02
        trig_b = quality & near.shift(1).fillna(False) & (c > e21) & (c > e55)
        sig_raw= trig_a | trig_b
        sig    = sig_raw & ~sig_raw.shift(1).fillna(False)
        sl_atr = c - 1.8*at14
        sl_sw  = sw5*0.998
        sl     = pd.concat([sl_atr, sl_sw], axis=1).max(axis=1).clip(lower=c*0.88, upper=c*0.975)
    else:
        return pd.DataFrame()

    risk = (c - sl).clip(lower=0.001)
    result = pd.DataFrame({
        "signal": sig.astype(int),
        "sl":     sl,
        "risk":   risk,
        "score":  1,
    }, index=df.index)
    return result


def run_portfolio_comparison(data: dict):
    results = []

    strategy_fns = {
        "Turtle20":   "strategy_turtle20",
        "Turtle55":   "strategy_turtle55",
        "ATR_BO":     "strategy_atr_breakout",
        "52W_High":   "strategy_52w_high",
        "TripleEMA":  "strategy_triple_ema",
        "Supertrend": "strategy_supertrend_strat",
        "Chan+OBV":   "strategy_chandelier_obv",
        "HeikinAshi": "strategy_heikin_ashi",
        "APEX_v4":    "strategy_apex_v4",
    }

    import strategies_lab as sl_mod
    fn_map = {
        "Turtle20":   sl_mod.strategy_turtle20,
        "Turtle55":   sl_mod.strategy_turtle55,
        "ATR_BO":     sl_mod.strategy_atr_breakout,
        "52W_High":   sl_mod.strategy_52w_high,
        "TripleEMA":  sl_mod.strategy_triple_ema,
        "Supertrend": sl_mod.strategy_supertrend_strat,
        "Chan+OBV":   sl_mod.strategy_chandelier_obv,
        "HeikinAshi": sl_mod.strategy_heikin_ashi,
        "APEX_v4":    sl_mod.strategy_apex_v4,
    }

    for strat_name, fn in fn_map.items():
        print(f"  Симулирую: {strat_name}...", end=" ", flush=True)

        # Генерируем сигналы
        sigs = {}
        for ticker, df in data.items():
            try:
                s = generate_signals(fn, df)
                if not s.empty:
                    sigs[ticker] = s
            except Exception:
                pass

        if not sigs:
            print("нет сигналов")
            continue

        res = portfolio_sim(strat_name, sigs, data)
        results.append(res)
        t = res["trades"]
        print(f"сделок={t}  WR={res['wr']:.1f}%  P&L={res['total_rub']:+,.0f}₽  "
              f"Ann={res['ann_ret']:+.1f}%  DD={res['max_dd']:.1f}%")

    return results


def print_comparison_table(results: list):
    print("\n" + "═"*80)
    print("  ПОРТФЕЛЬНОЕ СРАВНЕНИЕ СТРАТЕГИЙ  (капитал 100,000 ₽)")
    print("═"*80)
    print(f"  {'Стратегия':12s} {'Сд':5s} {'WR%':6s} {'PF':5s} "
          f"{'P&L ₽':10s} {'P&L%':7s} {'Год%':6s} {'DD%':7s} {'W/L':5s}")
    print("  " + "─"*74)

    for r in sorted(results, key=lambda x: -x["ann_ret"]):
        pf_str = f"{min(r['pf'],99):.2f}"
        wl_str = f"{min(r['wl_ratio'],9.9):.2f}x"
        print(f"  {r['strategy']:12s} {r['trades']:5d} {r['wr']:5.1f}% "
              f"{pf_str:5s} {r['total_rub']:>+9,.0f} {r['total_pct']:>+6.1f}% "
              f"{r['ann_ret']:>+5.1f}% {r['max_dd']:>6.1f}% {wl_str:6s}")
    print("═"*80)


def print_detailed(res: dict):
    print(f"\n  {'─'*60}")
    print(f"  ДЕТАЛИ: {res['strategy']}")
    print(f"  {'─'*60}")
    print("  Начальный капитал: 100,000 ₽")
    print(f"  Конечный капитал:  {res['cap_final']:>10,.0f} ₽")
    print(f"  P&L:               {res['total_rub']:>+10,.0f} ₽  ({res['total_pct']:+.1f}%)")
    print(f"  Годовая доходность:{res['ann_ret']:>+10.1f}%")
    print(f"  Макс. просадка:    {res['max_dd']:>10.1f}%")
    print(f"  Win Rate:          {res['wr']:>10.1f}%")
    print(f"  Profit Factor:     {min(res['pf'],99):>10.2f}")
    print(f"  Avg Win:           {res['avg_win']:>+10,.0f} ₽")
    print(f"  Avg Loss:          {res['avg_loss']:>+10,.0f} ₽")
    print(f"  W/L Ratio:         {min(res['wl_ratio'],99):>10.2f}x")
    print("  Выходы:")
    for reason, cnt in sorted(res["exit_cnt"].items()):
        r_t = [t for t in res["trades_list"] if t.reason == reason]
        r_w = sum(1 for t in r_t if t.win)
        r_pnl = sum(t.pnl_rub for t in r_t)
        print(f"    {reason:8s}: {cnt:3d}  WR={r_w/cnt*100:4.0f}%  "
              f"P&L={r_pnl:>+9,.0f} ₽")
    # Топ тикеры
    tkr_pnl = defaultdict(float)
    for t in res["trades_list"]:
        tkr_pnl[t.ticker] += t.pnl_rub
    print("  Топ тикеры:")
    for tk, p in sorted(tkr_pnl.items(), key=lambda x: -x[1])[:8]:
        n = sum(1 for t in res["trades_list"] if t.ticker == tk)
        print(f"    {tk:6s}: {p:>+9,.0f} ₽  ({n} сд)")


if __name__ == "__main__":
    print("=" * 65)
    print("  ПОРТФЕЛЬНЫЙ ЛАБ — тест 9 стратегий на 100,000 ₽")
    print("=" * 65)

    print("\nЗагрузка данных...")
    data = {}
    for t in TICKERS:
        df = load_daily(t)
        if df is not None:
            data[t] = df
    print(f"  Загружено: {len(data)} тикеров\n")

    results = run_portfolio_comparison(data)
    print_comparison_table(results)

    # Детали топ-3
    top3 = sorted(results, key=lambda x: -x["ann_ret"])[:3]
    for r in top3:
        print_detailed(r)

    # Сохранение лога
    all_trades = []
    for r in results:
        for t in r["trades_list"]:
            all_trades.append({
                "strategy": t.strategy, "ticker": t.ticker,
                "entry_dt": t.entry_dt, "exit_dt": t.exit_dt,
                "entry": t.entry, "exit": t.exit, "reason": t.reason,
                "pos_rub": t.pos_rub, "pnl_rub": t.pnl_rub,
                "pnl_pct": t.pnl_pct, "hold_d": t.hold_d, "win": t.win,
            })
    pd.DataFrame(all_trades).to_csv("c:/investor/portfolio_lab_trades.csv", index=False)
    print("\n  Все сделки → portfolio_lab_trades.csv")
