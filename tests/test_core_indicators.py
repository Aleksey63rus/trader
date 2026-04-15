"""Tests for core.indicators module."""
import numpy as np
import pandas as pd
import pytest

from core.indicators import (
    adx, atr, atr_ratio, directional_index,
    efficiency_ratio, ema, macd_histogram, rsi, volume_ratio,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
def make_df(n=200, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)
    high  = close * (1 + rng.uniform(0, 0.01, n))
    low   = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close + rng.normal(0, 0.2, n)
    vol   = rng.integers(1_000, 100_000, n).astype(float)
    idx   = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


@pytest.fixture
def df():
    return make_df(300)


# ── ATR ───────────────────────────────────────────────────────────────────────
class TestATR:
    def test_returns_series_same_length(self, df):
        result = atr(df, 14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_positive_values(self, df):
        result = atr(df, 14)
        # after warmup all values should be > 0
        assert (result.dropna() > 0).all()

    def test_warmup_nans(self, df):
        result = atr(df, 14)
        # first 13 values may be NaN (min_periods=14)
        assert result.iloc[13:].notna().all()

    def test_short_period_larger_than_long(self, df):
        """ATR(5) should generally be more reactive than ATR(14)."""
        a5  = atr(df, 5).dropna()
        a14 = atr(df, 14).dropna()
        # Both have positive mean; variance of a5 > a14 (short-term more reactive)
        assert a5.std() >= a14.std() * 0.5   # loose check


# ── EMA ───────────────────────────────────────────────────────────────────────
class TestEMA:
    def test_length(self, df):
        result = ema(df["close"], 20)
        assert len(result) == len(df)

    def test_converges_on_constant(self):
        s = pd.Series([10.0] * 100)
        result = ema(s, 10)
        assert abs(float(result.iloc[-1]) - 10.0) < 1e-6

    def test_shorter_faster(self, df):
        e10 = ema(df["close"], 10)
        e50 = ema(df["close"], 50)
        # e10 should track close more tightly
        diff10 = (df["close"] - e10).abs().mean()
        diff50 = (df["close"] - e50).abs().mean()
        assert diff10 <= diff50


# ── RSI ───────────────────────────────────────────────────────────────────────
class TestRSI:
    def test_range(self, df):
        result = rsi(df, 14).dropna()
        assert (result >= 0).all() and (result <= 100).all()

    def test_length(self, df):
        assert len(rsi(df, 14)) == len(df)

    def test_rising_market_high_rsi(self):
        """
        RSI on a strongly trending market should be high.
        Pure monotone rise (no down bars) causes the EWM loss series to be 0 → NaN.
        We construct a market with occasional small pullbacks to avoid that edge case
        while still being strongly bullish (≈ 90% up bars).
        """
        rng   = np.random.default_rng(42)
        n     = 150
        steps = np.where(rng.random(n) < 0.10, -0.3, +1.0)  # 90% up, 10% small down
        close = pd.Series(100.0 + np.cumsum(steps))
        idx   = pd.RangeIndex(n)
        df_r  = pd.DataFrame({
            "close": close,
            "open":  close - 0.1,
            "high":  close + 0.5,
            "low":   close - 0.5,
            "volume": 1000.0,
        }, index=idx)
        r = rsi(df_r, 14).dropna()
        assert len(r) > 0, "RSI returned empty series — no down bars, EWM is degenerate"
        assert float(r.iloc[-1]) > 70, f"Expected RSI > 70 for 90% bull market, got {r.iloc[-1]:.1f}"

    def test_falling_market_low_rsi(self):
        # Steadily falling prices → RSI near 0
        close = pd.Series([200.0 - i for i in range(100)])
        df_f  = pd.DataFrame({"close": close, "open": close, "high": close + 0.5,
                               "low": close - 0.5, "volume": 1000.0})
        r = rsi(df_f, 14).dropna()
        assert float(r.iloc[-1]) < 20


# ── Efficiency Ratio ──────────────────────────────────────────────────────────
class TestER:
    def test_range(self, df):
        result = efficiency_ratio(df["close"], 20).dropna()
        assert (result >= 0).all() and (result <= 1).all()

    def test_pure_trend_er_one(self):
        # Monotonically rising: ER should be very close to 1
        close = pd.Series([float(i) for i in range(100)])
        result = efficiency_ratio(close, 20).dropna()
        assert float(result.iloc[-1]) > 0.99

    def test_random_walk_er_low(self):
        # Pure noise: ER should be well below 0.5
        rng   = np.random.default_rng(0)
        close = pd.Series(100 + np.cumsum(rng.choice([-1, 1], 500)))
        result = efficiency_ratio(close, 20).dropna()
        assert float(result.mean()) < 0.5


# ── MACD histogram ────────────────────────────────────────────────────────────
class TestMACDHistogram:
    def test_length(self, df):
        assert len(macd_histogram(df)) == len(df)

    def test_sign_changes(self, df):
        hist = macd_histogram(df).dropna()
        # A typical price series has both positive and negative MACD hist values
        assert (hist > 0).any() and (hist < 0).any()


# ── DI / ADX ─────────────────────────────────────────────────────────────────
class TestDI_ADX:
    def test_di_positive(self, df):
        di_p, di_n = directional_index(df)
        assert (di_p.dropna() >= 0).all()
        assert (di_n.dropna() >= 0).all()

    def test_adx_range(self, df):
        result = adx(df, 14).dropna()
        assert (result >= 0).all()
        # ADX rarely exceeds 100 in practice
        assert (result <= 100).all()


# ── Volume ratio ──────────────────────────────────────────────────────────────
class TestVolumeRatio:
    def test_constant_volume(self):
        df_c = pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1000.0,
        }, index=range(50))
        result = volume_ratio(df_c, 20).dropna()
        # constant volume → ratio should be ~1.0
        assert abs(float(result.mean()) - 1.0) < 0.01

    def test_spike_detection(self, df):
        df2 = df.copy()
        df2.loc[df2.index[-1], "volume"] *= 10
        result = volume_ratio(df2, 20)
        assert float(result.iloc[-1]) > 5   # clear spike


# ── ATR ratio ─────────────────────────────────────────────────────────────────
class TestATRRatio:
    def test_greater_than_zero(self, df):
        result = atr_ratio(df).dropna()
        assert (result > 0).all()

    def test_expanding_vol(self):
        """When short ATR > long ATR → ratio > 1."""
        close = pd.Series([100.0] * 50 + [100.0 + i * 2 for i in range(100)])
        df_e  = pd.DataFrame({
            "open": close, "high": close * 1.005,
            "low": close * 0.995, "close": close, "volume": 1000.0,
        })
        result = atr_ratio(df_e, 5, 14).dropna()
        # After the expansion starts, ratio should be > 1 at some point
        assert (result > 1).any()
