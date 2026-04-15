"""Диагностика результатов портфельного бэктеста."""
import pandas as pd

df = pd.read_csv('portfolio_trades.csv')
df_no = df[df['ticker'] != 'TGKA'].copy()

print("=" * 60)
print("  ДИАГНОСТИКА БЕЗ TGKA")
print("=" * 60)
wins   = df_no[df_no['pnl'] > 0]
losses = df_no[df_no['pnl'] <= 0]
total_pnl = df_no['pnl'].sum()
wr = len(wins) / len(df_no) * 100
pf = abs(wins['pnl'].sum()) / abs(losses['pnl'].sum())
print(f"  Сделок:          {len(df_no)}")
print(f"  Win Rate:        {wr:.1f}%")
print(f"  Средний выигрыш: {wins['pnl'].mean():+.1f} руб.")
print(f"  Средний проигрыш:{losses['pnl'].mean():+.1f} руб.")
print(f"  Profit Factor:   {pf:.3f}")
print(f"  Суммарный P&L:   {total_pnl:+,.0f} руб.")
print(f"  Ratio W/L:       {abs(wins['pnl'].mean()) / abs(losses['pnl'].mean()):.2f}x  (нужно >1.5x)")
print()

print("  РАСПРЕДЕЛЕНИЕ P&L ПО РАЗМЕРУ:")
bins   = [-99999, -500, -200, -100, -50, 0, 50, 100, 200, 500, 99999]
labels = ['<-500','-500..-200','-200..-100','-100..-50','-50..0','0..50','50..100','100..200','200..500','>500']
df_no['bucket'] = pd.cut(df_no['pnl'], bins=bins, labels=labels)
dist = df_no.groupby('bucket', observed=True).agg(n=('pnl','count'), total=('pnl','sum'))
for b, row in dist.iterrows():
    bar = '#' * min(int(abs(row['n'])/1), 40)
    print(f"  {str(b):>14}: {int(row['n']):>4} сд.  сумма={row['total']:>+8,.0f}  {bar}")

print()
print("  КОМИССИОННАЯ НАГРУЗКА:")
# 0.05% per side = 0.1% round trip
est_comm = (df_no['entry_price'] * df_no['shares'] * 0.001).sum()
print(f"  Оценка суммарных комиссий: {est_comm:+,.0f} руб.")
print(f"  Это {est_comm/100000*100:.1f}% от стартового капитала за весь период")
print(f"  Количество сделок: {len(df_no)} — очень много, комиссии поедают прибыль")

print()
print("  SL-СДЕЛКИ БЕЗ ЧАСТИЧНОГО TP (чистые убытки):")
sl0 = df_no[(df_no['reason'] == 'SL') & (df_no['tp_hit'] == 0)]
sl1 = df_no[(df_no['reason'] == 'SL') & (df_no['tp_hit'] > 0)]
print(f"  SL без частичного TP: {len(sl0)} сделок → {sl0['pnl'].sum():+,.0f} руб.")
print(f"  SL с частичным TP:    {len(sl1)} сделок → {sl1['pnl'].sum():+,.0f} руб.")
print(f"  TP3 (полный выход):   {len(df_no[df_no['reason']=='TP3'])} сделок → {df_no[df_no['reason']=='TP3']['pnl'].sum():+,.0f} руб.")
print(f"  TIME (4д выход):      {len(df_no[df_no['reason']=='TIME'])} сделок → {df_no[df_no['reason']=='TIME']['pnl'].sum():+,.0f} руб.")

print()
print("  РЕЗУЛЬТАТЫ ПО ГОДАМ:")
df_no['year'] = pd.to_datetime(df_no['entry_dt']).dt.year
yearly = df_no.groupby('year').agg(
    n=('pnl','count'),
    wr=('pnl', lambda x: (x > 0).mean() * 100),
    total=('pnl','sum')
)
for yr, row in yearly.iterrows():
    sign = '+' if row['total'] >= 0 else ''
    print(f"  {yr}: {int(row['n']):>3} сделок  WR={row['wr']:.0f}%  P&L={sign}{row['total']:,.0f} руб.")

print()
print("  ТОП-5 ПРИБЫЛЬНЫХ ТИКЕРОВ:")
t_stat = df_no.groupby('ticker').agg(
    n=('pnl','count'),
    wr=('pnl', lambda x: (x > 0).mean() * 100),
    total=('pnl','sum'),
    avg_win=('pnl', lambda x: x[x>0].mean() if (x>0).any() else 0),
    avg_loss=('pnl', lambda x: x[x<=0].mean() if (x<=0).any() else 0),
).sort_values('total', ascending=False)
for tk, row in t_stat.iterrows():
    pf_t = 0 if row['avg_loss'] == 0 else abs(row['avg_win'] * row['wr']/100) / abs(row['avg_loss'] * (1-row['wr']/100))
    print(f"  {tk:<6}: {int(row['n']):>3} сд  WR={row['wr']:.0f}%  avg_W={row['avg_win']:>+.0f}  avg_L={row['avg_loss']:>+.0f}  P&L={row['total']:>+,.0f}")

print()
print("=" * 60)
print("  КОРЕНЬ ПРОБЛЕМЫ:")
print("=" * 60)
ratio = abs(wins['pnl'].mean()) / abs(losses['pnl'].mean())
print(f"""
  1. СООТНОШЕНИЕ W/L = {ratio:.2f}x
     Выигрыш ({wins['pnl'].mean():+.0f}) < Проигрыш ({losses['pnl'].mean():+.0f})
     При WR=58% нужен W/L > 0.74 для безубытка → у нас {ratio:.2f}
     Фактически граничный PF = {wr/100 * abs(wins['pnl'].mean()) / ((1-wr/100) * abs(losses['pnl'].mean())):.2f} (нужен > 1.0)

  2. ПЕРЕТРЕЙДИНГ: {len(df_no)} сделок за 4 года = {len(df_no)/4:.0f} сделок/год
     = {len(df_no)/(4*12):.0f} сделок/месяц по 12 тикерам
     Каждая сделка платит двойную комиссию (вход+выход)

  3. SL СЛИШКОМ БЛИЗКО: мелкие шумовые движения выбивают стоп
     {len(sl0)} сделок закрыты по SL без единого частичного TP

  4. МАЛЫЙ SIZE ПОЗИЦИИ: при капитале 100K и распределении на 3 позиции
     каждая позиция ~33K. При цене SBER 250 руб. × 90 акций = 22К.
     Выигрыш +0.5% = +110 руб., проигрыш -1.3% = -286 руб.
     Асимметрия в пользу убытков.
""")
