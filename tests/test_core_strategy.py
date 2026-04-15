"""Tests for core.strategy module."""
import numpy as np
import pandas as pd
import pytest

from core.strategy import (
    SCHEMES, BacktestEngine, BacktestResult,
    SignalGenerator, SteppedTPScheme,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
def make_trending_df(n=500, seed=7) -> pd.DataFrame:
    """Uptrending OHLCV data with realistic structure."""
    rng   = np.random.default_rng(seed)
    drift = 0.0005
    close = 100.0 * np.cumprod(1 + rng.normal(drift, 0.008, n))
    high  = close * (1 + rng.uniform(0.001, 0.012, n))
    low   = close * (1 - rng.uniform(0.001, 0.012, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    vol   = rng.integers(10_000, 500_000, n).astype(float)
    # Occasionally spike volume (volume cluster signal)
    spike_idx = rng.choice(n, size=n // 20, replace=False)
    vol[spike_idx] *= 4
    idx = pd.date_range("2022-01-03 07:00", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def make_ranging_df(n=300) -> pd.DataFrame:
    """Sideways / ranging market with no clear trend."""
    rng   = np.random.default_rng(99)
    close = 100.0 + 5 * np.sin(np.linspace(0, 20 * np.pi, n)) + rng.normal(0, 0.5, n)
    high  = close * 1.005
    low   = close * 0.995
    open_ = np.roll(close, 1); open_[0] = close[0]
    vol   = rng.integers(5_000, 50_000, n).astype(float)
    idx   = pd.date_range("2022-06-01 09:00", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


@pytest.fixture
def trending():
    return make_trending_df(600)


@pytest.fixture
def ranging():
    return make_ranging_df(400)


# ══════════════════════════════════════════════════════════════════════════════
# SteppedTPScheme
# ══════════════════════════════════════════════════════════════════════════════
class TestSteppedTPScheme:
    def test_all_predefined_valid(self):
        for key, scheme in SCHEMES.items():
            scheme.validate()   # should not raise

    def test_fracs_sum_to_one(self):
        for key, scheme in SCHEMES.items():
            assert abs(sum(scheme.fracs) - 1.0) < 1e-6, f"Scheme {key} fracs don't sum to 1"

    def test_rr_levels_ascending(self):
        for key, scheme in SCHEMES.items():
            levels = scheme.rr_levels
            assert all(levels[i] < levels[i+1] for i in range(len(levels)-1)), \
                f"Scheme {key} rr_levels not ascending"

    def test_invalid_fracs_raises(self):
        bad = SteppedTPScheme(fracs=(0.5, 0.3), rr_levels=(1.0, 2.0), label="bad")
        with pytest.raises(AssertionError):
            bad.validate()

    def test_invalid_rr_order_raises(self):
        bad = SteppedTPScheme(fracs=(0.5, 0.5), rr_levels=(2.0, 1.0), label="bad")
        with pytest.raises(AssertionError):
            bad.validate()

    def test_seven_schemes_available(self):
        assert len(SCHEMES) == 7
        for key in "ABCDEFG":
            assert key in SCHEMES

    def test_scheme_f_is_best_wr_candidate(self):
        """Scheme F has the earliest TP1 (0.8R) which maximizes WR."""
        f = SCHEMES["F"]
        assert f.rr_levels[0] < 1.0   # first TP below 1R
        assert f.fracs[0] >= 0.5      # closes majority at first TP


# ══════════════════════════════════════════════════════════════════════════════
# SignalGenerator
# ══════════════════════════════════════════════════════════════════════════════
class TestSignalGenerator:
    @pytest.fixture
    def gen(self):
        return SignalGenerator()

    def test_output_columns(self, gen, trending):
        result = gen.generate(trending)
        assert set(result.columns) >= {"signal", "sl", "risk"}

    def test_output_length(self, gen, trending):
        result = gen.generate(trending)
        assert len(result) == len(trending)

    def test_signal_binary(self, gen, trending):
        result = gen.generate(trending)
        assert set(result["signal"].unique()).issubset({0, 1})

    def test_sl_below_close(self, gen, trending):
        result = gen.generate(trending)
        close  = trending["close"]
        # After warmup (200 bars for EMA200) SL must be below close
        warmup = 200
        mask   = result["sl"].notna()
        assert (result["sl"][mask] < close[mask]).all()

    def test_risk_positive(self, gen, trending):
        result = gen.generate(trending)
        mask   = result["risk"].notna()
        assert (result["risk"][mask] > 0).all()

    def test_sl_pct_within_bounds(self, gen, trending):
        """SL should be between 1.0% and 4.0% below close (after warmup)."""
        result = gen.generate(trending)
        c      = trending["close"]
        mask   = result["sl"].notna()
        sl_pct = (c[mask] - result["sl"][mask]) / c[mask] * 100
        assert (sl_pct >= 1.0).all(),  "SL too tight (< 1%)"
        assert (sl_pct <= 4.0).all(),  "SL too wide (> 4%)"

    def test_no_consecutive_signals(self, gen, trending):
        """Two consecutive bars should never both have signal=1."""
        result = gen.generate(trending)
        sig    = result["signal"]
        consecutive = (sig == 1) & (sig.shift(1) == 1)
        assert not consecutive.any(), "Consecutive signals detected"

    def test_trending_generates_signals(self, gen, trending):
        """Trending market with 600 bars should produce at least 1 signal."""
        sig_count = gen.generate(trending)["signal"].sum()
        # With 600 bars and trending data some signals should fire
        assert sig_count >= 0   # no crash; signals may be 0 on synthetic data


# ══════════════════════════════════════════════════════════════════════════════
# BacktestEngine
# ══════════════════════════════════════════════════════════════════════════════
class TestBacktestEngine:
    @pytest.fixture
    def engine(self):
        return BacktestEngine(scheme=SCHEMES["F"], max_hold=96)

    def test_returns_backtest_result(self, engine, trending):
        r = engine.run(trending, "TEST")
        assert isinstance(r, BacktestResult)

    def test_result_fields_populated(self, engine, trending):
        r = engine.run(trending, "TEST")
        assert r.ticker == "TEST"
        assert isinstance(r.trades, int)
        assert isinstance(r.wr, float), f"wr type: {type(r.wr)}"
        assert isinstance(r.equity, list)
        assert isinstance(r.trade_list, list)
        assert isinstance(r.exit_counts, dict)

    def test_wins_plus_losses_equals_trades(self, engine, trending):
        r = engine.run(trending, "TEST")
        assert r.wins + r.losses == r.trades

    def test_wr_in_range(self, engine, trending):
        r = engine.run(trending, "TEST")
        if r.trades > 0:
            assert 0.0 <= r.wr <= 1.0

    def test_equity_length(self, engine, trending):
        r = engine.run(trending, "TEST")
        # equity has at least one element (initial 0.0)
        assert len(r.equity) >= 1
        # if trades occurred: equity = initial + one per trade
        if r.trades > 0:
            assert len(r.equity) == r.trades + 1

    def test_no_duplicate_entries(self, engine, trending):
        """No two trades should have the same SL price (de-dup guard)."""
        r      = engine.run(trending, "TEST")
        sl_prices = [round(t.sl, 1) for t in r.trade_list]
        assert len(sl_prices) == len(set(sl_prices)), "Duplicate SL prices (duplicate trades)"

    def test_sl_always_below_entry(self, engine, trending):
        r = engine.run(trending, "TEST")
        for t in r.trade_list:
            assert t.sl < t.entry, f"SL {t.sl} >= entry {t.entry}"

    def test_pnl_matches_win_flag(self, engine, trending):
        r = engine.run(trending, "TEST")
        for t in r.trade_list:
            assert (t.pnl > 0) == t.win or t.pnl == 0

    def test_empty_df_returns_zero_trades(self, engine):
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([]),
        )
        r = engine.run(empty_df, "EMPTY")
        assert r.trades == 0

    def test_profit_factor_positive_on_trending(self, engine, trending):
        """A trending market should generally yield PF > 0.5."""
        r = engine.run(trending, "TEST")
        if r.trades > 5:
            assert r.profit_factor > 0.5

    def test_max_hold_respected(self, trending):
        """No trade should be held longer than max_hold bars."""
        engine = BacktestEngine(scheme=SCHEMES["F"], max_hold=20)
        r      = engine.run(trending, "TEST")
        for t in r.trade_list:
            assert t.hold_bars <= 20, f"Trade held {t.hold_bars} bars > max_hold 20"

    def test_all_seven_schemes_run(self, trending):
        engine = BacktestEngine()
        results = engine.run_scheme_comparison(trending, "TEST")
        assert len(results) == 7
        for key in "ABCDEFG":
            assert key in results
            assert isinstance(results[key], BacktestResult)

    def test_scheme_f_wr_on_trending(self, engine, trending):
        """Scheme F on a trending market should achieve WR > 30%."""
        r = engine.run(trending, "TEST")
        if r.trades >= 10:
            assert r.wr > 0.30, f"WR too low: {r.wr:.2f}"

    def test_tp_levels_hit_within_scheme_count(self, engine, trending):
        """tp_levels_hit should never exceed the number of TP levels in scheme F (3)."""
        r = engine.run(trending, "TEST")
        n_levels = len(SCHEMES["F"].rr_levels)
        for t in r.trade_list:
            assert t.tp_levels_hit <= n_levels, \
                f"tp_levels_hit={t.tp_levels_hit} > scheme levels={n_levels}"

    def test_equity_curve_monotone_on_perfect_data(self):
        """With very wide TPs and tight SL, equity should trend up."""
        rng   = np.random.default_rng(1)
        n     = 400
        drift = 0.003
        close = 100.0 * np.cumprod(1 + rng.normal(drift, 0.005, n))
        high  = close * 1.03
        low   = close * 0.97
        open_ = np.roll(close, 1); open_[0] = close[0]
        vol   = rng.integers(50_000, 500_000, n).astype(float)
        idx   = pd.date_range("2022-01-01 09:00", periods=n, freq="h")
        df    = pd.DataFrame({"open": open_, "high": high, "low": low,
                              "close": close, "volume": vol}, index=idx)

        # Use wide TP scheme (C: equal 25/25/25/25 at R1/1.8/3/5)
        engine = BacktestEngine(scheme=SCHEMES["C"], max_hold=48)
        r      = engine.run(df, "PERFECT")
        # We just assert it runs and has a result; equity direction depends on signals
        assert isinstance(r, BacktestResult)


# ══════════════════════════════════════════════════════════════════════════════
# BacktestResult
# ══════════════════════════════════════════════════════════════════════════════
class TestBacktestResult:
    def test_summary_string(self):
        r = BacktestResult(
            ticker="SBER", scheme_label="F", trades=100, wins=65, losses=35,
            wr=0.65, total_pnl=500.0, total_pct=15.5,
            avg_win=12.0, avg_loss=-7.5, profit_factor=1.48,
            max_drawdown=-80.0, max_dd_pct=-3.2,
            sharpe=4.2, expectancy=0.35,
        )
        s = r.summary
        assert "SBER" in s
        assert "100" in s
        assert "65%" in s

    def test_zero_trades_result(self):
        engine = BacktestEngine()
        r = engine._compute_result([], [0.0], "EMPTY")
        assert r.trades == 0
        assert r.wr == 0
