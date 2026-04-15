"""Тюнинг v3c: тестируем разные комбинации тикеров и схем TP."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from pro_portfolio_v3c import ProSimulatorV3c, RISK_PARAMS, SIGNAL_CFG
from core.strategy_v2 import BacktestEngineV2, TP_SCHEMES
from core.risk import RiskManager
from pro_portfolio_v3c import load_daily, INITIAL_CAPITAL, MAX_HOLD

BASE_TICKERS = [
    "LKOH","NVTK","ROSN","SBERP","T",
    "GMKN","NLMK","MTLR","CHMF","MAGN","RUAL","ALRS","PLZL",
    "OZON","MGNT","TATN","TATNP","IRAO","PHOR","OZPH",
]

TESTS = [
    # name, убранные тикеры, scheme
    ("v3c-base",           [],                          "AGR"),
    ("v3c-noIRAO",         ["IRAO"],                    "AGR"),
    ("v3c-noTATN_IRAO",    ["IRAO","TATN"],              "AGR"),
    ("v3c-no3bad",         ["IRAO","TATN","PHOR"],        "AGR"),
    ("v3c-no3bad+BAL",     ["IRAO","TATN","PHOR"],        "BAL"),
    ("v3c-no3bad+CONS",    ["IRAO","TATN","PHOR"],        "CONS"),
    ("v3c-no3bad+FAST",    ["IRAO","TATN","PHOR"],        "FAST"),
]

print(f"{'Config':22s}  {'Tr':4s}  {'WR':6s}  {'PF':5s}  {'Total%':8s}  {'DD':7s}  {'Ann%':6s}")
print("-" * 75)
best_ann = -99; best = None

for name, remove, scheme in TESTS:
    tickers = [t for t in BASE_TICKERS if t not in remove]

    sim = ProSimulatorV3c.__new__(ProSimulatorV3c)
    sim.rm      = RiskManager(INITIAL_CAPITAL, RISK_PARAMS)
    sim.data    = {}
    sim.signals = {}
    sim.trades  = []
    sim.equity  = []
    sim.blocked = []

    gen = BacktestEngineV2(scheme, MAX_HOLD, SIGNAL_CFG, "D")
    for t in tickers:
        df = load_daily(t)
        if df is not None and len(df) >= 250:
            sim.data[t]    = df
            sim.signals[t] = gen._gen.generate(df)

    # Patch TICKERS inside run()
    import pro_portfolio_v3c as m
    orig = m.TICKERS; m.TICKERS = tickers
    fracs_orig, levels_orig = m.FRACS, m.LEVELS
    m.FRACS, m.LEVELS = TP_SCHEMES[scheme]

    sim.run()

    m.TICKERS = orig
    m.FRACS, m.LEVELS = fracs_orig, levels_orig

    trades = sim.trades
    if not trades:
        print(f"{name:22s}: нет сделок")
        continue

    cap = sim.rm.state.capital
    total_pct = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    years = 4.1
    ann   = ((cap / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    wins  = [t for t in trades if t.win]
    losses= [t for t in trades if not t.win]
    wr    = len(wins) / len(trades) * 100
    pf    = abs(sum(t.pnl_rub for t in wins)) / (abs(sum(t.pnl_rub for t in losses)) + 1e-9)
    eq    = np.array([e[1] for e in sim.equity])
    if len(eq) > 0:
        pk = np.maximum.accumulate(np.maximum(eq, 1.0))
        dd = float(((eq - pk) / pk * 100).min())
    else:
        dd = 0.0

    marker = " <--" if ann > best_ann else ""
    if ann > best_ann:
        best_ann = ann; best = name
    print(f"{name:22s}  {len(trades):4d}  {wr:5.1f}%  {pf:4.2f}  {total_pct:+8.1f}%  {dd:6.1f}%  {ann:+5.1f}%{marker}")

print(f"\nЛучший: {best}")
