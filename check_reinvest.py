"""Проверка реинвестирования и запуск с явным compound-ростом позиций."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from final_portfolio_v4c import run_sim, INITIAL_CAP

# Лучшая конфигурация
r = run_sim(0.00, 2.0, (), (), 45, 0, "B-Trail2-noSL-h45")

trades = r["trades_list"]

# Размер позиций по годам — докажем compound
by_year = {}
for t in trades:
    yr = str(t.entry_date)[:4]
    by_year.setdefault(yr, []).append(t.shares * t.entry_px)

print("=" * 60)
print("ПРОВЕРКА РЕИНВЕСТИРОВАНИЯ")
print("Средний размер позиции по годам (должен расти с прибылью):")
print(f"  {'Год':6s} {'Avg, руб':>12s} {'Max, руб':>12s} {'Сделок':>8s}")
print("  " + "-" * 44)
for yr, sizes in sorted(by_year.items()):
    print(f"  {yr}   {np.mean(sizes):>12,.0f}   {np.max(sizes):>12,.0f}   {len(sizes):>6d}")

print(f"\nНачальный капитал: {INITIAL_CAP:>12,.0f} руб")
print(f"Итоговый капитал:  {r['final']:>12,.0f} руб")
print(f"Рост:              {r['total_pnl']:>+11.1f}%")
print(f"Годовых:           {r['ann']:>+11.1f}%")
print("=" * 60)

# Дополнительно: покажем капитал на начало каждого года
# (приблизительно, по дате первой сделки года)
print("\nДинамика капитала (по сделкам):")
running = INITIAL_CAP
print(f"  {'Дата':12s} {'Тикер':6s} {'P&L%':>7s} {'Капитал':>12s}")
print("  " + "-" * 44)
# Сортируем по дате закрытия
for t in sorted(trades, key=lambda x: x.exit_date):
    running += t.pnl_rub
    print(f"  {str(t.exit_date)[:10]:12s} {t.ticker:6s} {t.pnl_pct:>+6.1f}%  {running:>12,.0f}")
