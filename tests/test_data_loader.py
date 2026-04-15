"""Tests for core.data_loader module."""
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from core.data_loader import load_csv, save_csv


# ── Helpers ───────────────────────────────────────────────────────────────────
def write_tmp(tmp_path: Path, content: str, name: str = "test.csv") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# load_csv
# ══════════════════════════════════════════════════════════════════════════════
class TestLoadCSV:

    FINAM_CONTENT = """\
        <TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>
        SBER;60;03/01/22;100000;280.5;282.0;279.0;281.0;1500000
        SBER;60;03/01/22;110000;281.0;283.5;280.0;283.0;1200000
        SBER;60;04/01/22;100000;283.0;285.0;282.0;284.5;900000
        SBER;60;04/01/22;110000;284.5;286.0;283.5;285.0;800000
    """

    GENERIC_CONTENT = """\
        datetime,open,high,low,close,volume
        2022-01-03 10:00:00,280.5,282.0,279.0,281.0,1500000
        2022-01-03 11:00:00,281.0,283.5,280.0,283.0,1200000
        2022-01-04 10:00:00,283.0,285.0,282.0,284.5,900000
    """

    def test_load_finam_format(self, tmp_path):
        p  = write_tmp(tmp_path, self.FINAM_CONTENT)
        df = load_csv(p)
        assert len(df) == 4
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_load_generic_format(self, tmp_path):
        p  = write_tmp(tmp_path, self.GENERIC_CONTENT)
        df = load_csv(p)
        assert len(df) == 3
        assert "close" in df.columns

    def test_index_is_datetime(self, tmp_path):
        p  = write_tmp(tmp_path, self.FINAM_CONTENT)
        df = load_csv(p)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_sorted(self, tmp_path):
        p  = write_tmp(tmp_path, self.FINAM_CONTENT)
        df = load_csv(p)
        assert df.index.is_monotonic_increasing

    def test_columns_float(self, tmp_path):
        p  = write_tmp(tmp_path, self.FINAM_CONTENT)
        df = load_csv(p)
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == float, f"{col} is not float"

    def test_no_nans(self, tmp_path):
        p  = write_tmp(tmp_path, self.FINAM_CONTENT)
        df = load_csv(p)
        assert not df.isnull().any().any()

    def test_vol_alias_renamed(self, tmp_path):
        content = """<TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>
SBER;60;03/01/22;100000;100;101;99;100.5;5000
"""
        p  = write_tmp(tmp_path, content)
        df = load_csv(p)
        assert "volume" in df.columns
        assert "vol" not in df.columns

    def test_missing_datetime_raises(self, tmp_path):
        bad = "open,high,low,close,volume\n100,101,99,100,5000\n"
        p   = write_tmp(tmp_path, bad)
        with pytest.raises((ValueError, KeyError)):
            load_csv(p)

    def test_real_file_if_exists(self):
        """Integration test: load one of the real CSV files if present."""
        p = Path("SBER_220103_260320_1H.csv")
        if not p.exists():
            pytest.skip("Real data file not present")
        df = load_csv(p)
        assert len(df) > 1000
        assert "close" in df.columns
        assert df.index.is_monotonic_increasing


# ══════════════════════════════════════════════════════════════════════════════
# save_csv
# ══════════════════════════════════════════════════════════════════════════════
class TestSaveCSV:

    def _sample_df(self) -> pd.DataFrame:
        idx = pd.date_range("2023-01-02 10:00", periods=5, freq="h")
        return pd.DataFrame({
            "open":   [100.0, 101.0, 102.0, 101.5, 103.0],
            "high":   [101.0, 102.0, 103.0, 102.5, 104.0],
            "low":    [99.0,  100.0, 101.0, 100.5, 102.0],
            "close":  [100.5, 101.5, 102.5, 101.0, 103.5],
            "volume": [10000.0, 12000.0, 8000.0, 9500.0, 11000.0],
        }, index=idx)

    def test_creates_file(self, tmp_path):
        df = self._sample_df()
        p  = save_csv(df, tmp_path / "SBER_test.csv")
        assert p.exists()

    def test_roundtrip(self, tmp_path):
        df_orig = self._sample_df()
        p       = save_csv(df_orig, tmp_path / "SBER_rt.csv")
        df_back = load_csv(p)
        # Check same length and close values within float tolerance
        assert len(df_back) == len(df_orig)
        pd.testing.assert_series_equal(
            df_back["close"].reset_index(drop=True),
            df_orig["close"].reset_index(drop=True),
            check_names=False,
        )

    def test_header_format(self, tmp_path):
        df = self._sample_df()
        p  = save_csv(df, tmp_path / "SBER_hdr.csv")
        header = p.read_text(encoding="utf-8").split("\n")[0]
        assert "<TICKER>" in header
        assert "<CLOSE>" in header

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "SBER_nested.csv"
        df = self._sample_df()
        save_csv(df, nested)
        assert nested.exists()
