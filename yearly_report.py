"""Годовой отчёт по всем конфигурациям портфельной симуляции."""
import warnings; warnings.filterwarnings("ignore")
from final_portfolio_v4c import run_sim, INITIAL_CAP, all_dates

CONFIGS = [
    ("A-Trail1.5-noSL-h40",    0.00, 1.5, (), (),                40, 0),
    ("B-Trail2-noSL-h45",      0.00, 2.0, (), (),                45, 0),
    ("C-Trail2-SL8%-h35",      0.08, 2.0, (), (),                35, 0),
    ("D-Trail1.5-SL8%-h30",    0.08, 1.5, (), (),                30, 0),
    ("E-noTrail-SL20%-h40",    0.20, 0.0, (), (),                40, 0),
    ("F-noTrail-SL15%-h30",    0.15, 0.0, (), (),                30, 0),
    ("G-TP10R-noTrail-SL5%",   0.05, 0.0, (1.0,),(10.0,),        25, 0),
    ("H-TP10R-noTrail-SL8%",   0.08, 0.0, (1.0,),(10.0,),        25, 0),
    ("I-50%@3R+trail2-SL5%",   0.05, 2.0, (0.5,0.5),(3.0,99.0),  35, 1),
    ("J-40%@5R+trail2-SL8%",   0.08, 2.0, (0.4,0.6),(5.0,99.0),  40, 1),
    ("K-Trail2-SL5%-h30",      0.05, 2.0, (), (),                30, 0),
    ("L-Trail2-SL10%-h40",     0.10, 2.0, (), (),                40, 0),
]

print("Запуск симуляций...")
all_results = []
for cfg in CONFIGS:
    label, sl_pct, trail, tp_fracs, tp_R, hold, taft = cfg
    r = run_sim(sl_pct, trail, tp_fracs, tp_R, hold, taft, label)
    all_results.append(r)
    print(f"  {label}: {r['total_pnl']:+.1f}%")

print()

# ── Для каждой конфигурации — капитал по годам ─────────────────────────────
for r in all_results:
    trades = r["trades_list"]

    # Строим equity по годам через кумулятивный P&L в рублях
    capital_by_year = {}
    running = INITIAL_CAP
    for t in sorted(trades, key=lambda x: x.exit_date):
        yr = str(t.exit_date)[:4]
        running += t.pnl_rub
        capital_by_year[yr] = running

    # Годы из данных
    years = sorted(set(str(d)[:4] for d in all_dates))

    print("=" * 72)
    print(f"  Стратегия: {r['label']}")
    print(f"  Начальный капитал: {INITIAL_CAP:>10,.0f} ₽")
    print("=" * 72)
    print(f"  {'Год':6s} {'Капитал на конец года':>22s} {'Прибыль за год':>16s} {'Доходность':>12s}")
    print("  " + "─" * 60)

    prev = INITIAL_CAP
    for yr in years:
        cap = capital_by_year.get(yr, prev)  # если нет сделок — капитал не менялся
        profit_yr = cap - prev
        pct_yr = profit_yr / prev * 100 if prev > 0 else 0
        print(f"  {yr}   {cap:>22,.0f} ₽   {profit_yr:>+14,.0f} ₽   {pct_yr:>+10.1f}%")
        prev = cap

    total_profit = r["final"] - INITIAL_CAP
    print("  " + "─" * 60)
    print(f"  {'ИТОГО':6s} {r['final']:>22,.0f} ₽   {total_profit:>+14,.0f} ₽   {r['total_pnl']:>+10.1f}%")
    print(f"  Годовая доходность (CAGR): {r['ann']:>+.1f}%   |   MaxDD: {r['max_dd']:>+.1f}%   |   Sharpe: {r['sharpe']:.2f}")
    print()
