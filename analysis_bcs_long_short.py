# -*- coding: utf-8 -*-
"""
Сравнение ATR_BO: LONG и SHORT по двум тарифам БКС (Инвестор, Трейдер).

Запуск: python analysis_bcs_long_short.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "trader"))

from core.data_loader import load_csv
from core.bcs_tariffs import BCSTariff, INVESTOR, TRADER_MID
from strategies.atr_bo_daily import ATRBOConfig, run_backtest
from strategies.atr_bo_short_daily import run_short_backtest

DATA_DIR = Path("c:/investor/data")
TICKERS = [
    "SBER", "LKOH", "GAZP", "YDEX", "T", "GMKN", "PLZL", "NVTK", "TATN", "VTBR",
    "ROSN", "SNGS", "X5", "MOEX", "OZON", "CHMF", "NLMK", "ALRS", "MGNT", "RTKM",
    "MTLR", "OZPH",
]


def load_all() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for t in TICKERS:
        cands = list(DATA_DIR.glob(f"{t}_*_D.csv"))
        if not cands:
            continue
        try:
            d = load_csv(cands[0])
            if d is not None and len(d) > 200:
                out[t] = d
        except Exception:
            pass
    return out


def cfg_from_tariff(t: BCSTariff) -> ATRBOConfig:
    return ATRBOConfig(
        commission_buy=t.commission_buy,
        commission_sell=t.commission_sell,
        slippage=t.slippage,
        commission=(t.commission_buy + t.commission_sell) / 2,
    )


def trader_fee_drag_rub(n_days: int, monthly_rub: float) -> float:
    months = max(1, n_days // 30)
    return monthly_rub * months


def main() -> None:
    print("Загрузка данных...")
    data = load_all()
    if not data:
        print("Нет CSV в data/.")
        return
    print(f"  Тикеров: {len(data)}")
    n_days = (
        (max(df.index.max() for df in data.values()) - min(df.index.min() for df in data.values())).days
    )

    tariffs: tuple[BCSTariff, ...] = (INVESTOR, TRADER_MID)

    print()
    print("=" * 96)
    print("  ATR Breakout Daily — LONG vs SHORT × тарифы БКС")
    print("=" * 96)
    print()
    print("  «Инвестор»: покупка 0%, продажа 0.30%, слиппаж 0.10%.")
    print("  «Трейдер» (модель): 0.02% + 0.02%, слипп 0.08%; колонка «абон» — оценка 299 ₽/мес за период (для 50/50 вычтена 1x).")
    print(f"  Период: ~{n_days} дн.; капитал 100 000 ₽ на каждый режим (отдельные прогоны).")
    print()

    hdr = (
        f"  {'Тариф':<32} {'Сторона':<8} {'Год.%':>8} {'P&L%':>8} {'MaxDD%':>9} "
        f"{'Сделок':>7} {'WR%':>7} {'PF':>6} {'Капитал':>12} {'абон ₽':>10}"
    )
    print(hdr)
    print("  " + "-" * 92)

    combined_rows: list[tuple[str, float, float, int, float]] = []

    for tr in tariffs:
        cfg = cfg_from_tariff(tr)
        rl = run_backtest(data, cfg, 100_000.0)
        rs = run_short_backtest(data, cfg, 100_000.0)
        rl50 = run_backtest(data, cfg, 50_000.0)
        rs50 = run_short_backtest(data, cfg, 50_000.0)

        nd = max(int(rl.get("n_days", n_days)), int(rs.get("n_days", n_days)))
        fee_est = trader_fee_drag_rub(nd, tr.monthly_fee_rub) if tr.monthly_fee_rub > 0 else 0.0

        for side, r in (("LONG", rl), ("SHORT", rs)):
            cap = float(r["final_capital"])
            pnl_pct = (cap - 100_000.0) / 100_000.0 * 100
            ann = ((cap / 100_000.0) ** (365 / max(int(r.get("n_days", n_days)), 1)) - 1) * 100
            print(
                f"  {tr.name:<32} {side:<8} {ann:>7.1f}% {pnl_pct:>7.1f}% "
                f"{r['max_drawdown']:>8.1f}% {r['n_trades']:>7} {r['win_rate']:>6.1f}% "
                f"{r['profit_factor']:>5.2f} {cap:>12,.0f} {fee_est:>10,.0f}"
            )

        fee_combo = trader_fee_drag_rub(nd, tr.monthly_fee_rub) if tr.monthly_fee_rub > 0 else 0.0
        comb = float(rl50["final_capital"]) + float(rs50["final_capital"]) - fee_combo
        pnl_c = (comb - 100_000.0) / 100_000.0 * 100
        ann_c = ((comb / 100_000.0) ** (365 / max(n_days, 1)) - 1) * 100
        combined_rows.append(
            (
                tr.name,
                ann_c,
                pnl_c,
                int(rl50["n_trades"]) + int(rs50["n_trades"]),
                fee_combo,
                comb,
            )
        )

    print()
    print("=" * 96)
    print("  50k LONG + 50k SHORT (один общий капитал 100k, два независимых бэктеста)")
    print("  " + "-" * 92)
    for name, ann_c, pnl_c, ntr, fee_c, comb_cap in combined_rows:
        print(
            f"  {name:<32} {'50/50':<8} {ann_c:>7.1f}% {pnl_c:>7.1f}% "
            f"  —        {ntr:>7}  —      —    {comb_cap:>12,.0f} {fee_c:>10,.0f}"
        )

    print()
    print("=" * 96)
    print("  ВЫВОДЫ")
    print("=" * 96)
    print("""
  • LONG и SHORT здесь — независимые бэктесты по 100k; «50/50» — два счёта по 50k без корреляции сделок.

  • «Трейдер» дешевле по обороту (симметричные ~0.02%), «Инвестор» выгоден при редких продажах (0% на покупку).

  • Шорт на «зеркальных» сигналах часто слабее лонга на длинном бычьем участке выборки; реальный шорт — маржа, GO, брокер.

  • Абонплата 299 ₽/мес для «Трейдера» вычтена грубо (× месяцев); уточните в договоре условия списания.
""")
    print("  Готово.")


if __name__ == "__main__":
    main()
