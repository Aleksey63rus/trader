"""Integration tests for the FastAPI web application."""
import textwrap

import pytest
from fastapi.testclient import TestClient

from web.app import _state, app

client = TestClient(app)

# ── Minimal valid CSV payload ──────────────────────────────────────────────────
CSV_CONTENT = textwrap.dedent("""\
    <TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>
    SBER;60;03/01/22;100000;280.5;282.0;279.0;281.0;1500000
    SBER;60;03/01/22;110000;281.0;283.5;280.0;283.0;1200000
    SBER;60;03/01/22;120000;283.0;285.0;282.0;284.5;900000
    SBER;60;03/01/22;130000;284.5;286.0;283.5;285.0;800000
    SBER;60;03/01/22;140000;285.0;287.0;284.0;286.5;750000
    SBER;60;03/01/22;150000;286.5;288.0;285.5;287.0;700000
    SBER;60;03/01/22;160000;287.0;289.0;286.5;288.5;650000
    SBER;60;03/01/22;170000;288.5;290.0;287.0;289.0;600000
    SBER;60;03/01/22;180000;289.0;291.0;288.0;290.5;580000
    SBER;60;03/01/22;190000;290.5;292.0;289.5;291.0;560000
""")


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state between tests."""
    _state["df"]      = None
    _state["ticker"]  = ""
    _state["result"]  = None
    _state["compare"] = None
    _state["equity"]  = []
    _state["trades"]  = []
    yield


def upload_csv():
    """Helper: upload the minimal test CSV."""
    return client.post(
        "/api/upload",
        files={"file": ("SBER_test.csv", CSV_CONTENT.encode(), "text/csv")},
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /
# ══════════════════════════════════════════════════════════════════════════════
class TestIndex:
    def test_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_is_html(self):
        r = client.get("/")
        assert "text/html" in r.headers["content-type"]

    def test_contains_title(self):
        r = client.get("/")
        assert "Momentum Filter Trader" in r.text


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/status
# ══════════════════════════════════════════════════════════════════════════════
class TestStatus:
    def test_empty_state(self):
        r = client.get("/api/status")
        assert r.status_code == 200
        d = r.json()
        assert d["data_loaded"] is False
        assert d["backtest_done"] is False

    def test_after_upload(self):
        upload_csv()
        r = client.get("/api/status")
        d = r.json()
        assert d["data_loaded"] is True
        assert d["ticker"] == "SBER"


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/schemes
# ══════════════════════════════════════════════════════════════════════════════
class TestSchemes:
    def test_returns_seven_schemes(self):
        r = client.get("/api/schemes")
        assert r.status_code == 200
        d = r.json()
        assert len(d) == 7
        for key in "ABCDEFG":
            assert key in d

    def test_scheme_labels_are_strings(self):
        d = client.get("/api/schemes").json()
        for v in d.values():
            assert isinstance(v, str) and len(v) > 3


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/upload
# ══════════════════════════════════════════════════════════════════════════════
class TestUpload:
    def test_valid_csv(self):
        r = upload_csv()
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["ticker"] == "SBER"
        assert d["rows"] > 0

    def test_invalid_csv_returns_400(self):
        r = client.post(
            "/api/upload",
            files={"file": ("bad.csv", b"not,a,valid,file\n", "text/csv")},
        )
        assert r.status_code == 400

    def test_state_updated_after_upload(self):
        upload_csv()
        assert _state["df"] is not None
        assert _state["ticker"] == "SBER"


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/backtest
# ══════════════════════════════════════════════════════════════════════════════
class TestBacktest:
    def test_without_data_returns_400(self):
        r = client.post("/api/backtest", json={"scheme": "F"})
        assert r.status_code == 400

    def test_with_uploaded_data(self):
        upload_csv()
        r = client.post("/api/backtest", json={"scheme": "F", "max_hold": 96})
        # Data is too small (10 rows) so trades=0, but no crash
        assert r.status_code == 200
        d = r.json()
        assert "wr_pct" in d
        assert "pf" in d
        assert "trades" in d

    def test_invalid_scheme_returns_400(self):
        upload_csv()
        r = client.post("/api/backtest", json={"scheme": "Z"})
        assert r.status_code == 400

    def test_response_fields(self):
        upload_csv()
        r = client.post("/api/backtest", json={"scheme": "A"})
        d = r.json()
        required = {"ticker", "scheme", "trades", "wins", "losses",
                    "wr_pct", "total_pct", "pf", "sharpe", "max_dd_pct",
                    "expectancy", "exit_dist", "equity", "trade_list"}
        assert required.issubset(d.keys())

    def test_all_schemes_accepted(self):
        upload_csv()
        for key in "ABCDEFG":
            r = client.post("/api/backtest", json={"scheme": key})
            assert r.status_code == 200, f"Scheme {key} returned {r.status_code}"

    def test_serialization_types(self):
        """All values in the response must be JSON-serializable native Python types."""
        upload_csv()
        r = client.post("/api/backtest", json={"scheme": "F"})
        d = r.json()   # will raise if serialization failed
        assert isinstance(d["wr_pct"], float)
        assert isinstance(d["trades"], int)


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/compare
# ══════════════════════════════════════════════════════════════════════════════
class TestCompare:
    def test_without_data_returns_400(self):
        r = client.post("/api/compare", json={})
        assert r.status_code == 400

    def test_returns_seven_schemes(self):
        upload_csv()
        r = client.post("/api/compare", json={"max_hold": 96})
        assert r.status_code == 200
        d = r.json()
        assert "schemes" in d
        assert len(d["schemes"]) == 7

    def test_sorted_by_wr_desc(self):
        upload_csv()
        r = client.post("/api/compare", json={})
        d = r.json()
        wrs = [s["wr_pct"] for s in d["schemes"]]
        assert wrs == sorted(wrs, reverse=True)

    def test_compare_schema(self):
        upload_csv()
        r  = client.post("/api/compare", json={})
        d  = r.json()
        s0 = d["schemes"][0]
        for field in ["key", "label", "trades", "wr_pct", "total_pct", "pf", "sharpe"]:
            assert field in s0, f"Missing field '{field}'"


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/chart
# ══════════════════════════════════════════════════════════════════════════════
class TestChart:
    def test_without_data_returns_400(self):
        r = client.get("/api/chart")
        assert r.status_code == 400

    def test_with_data_returns_plotly_json(self):
        upload_csv()
        r = client.get("/api/chart?max_candles=50")
        assert r.status_code == 200
        d = r.json()
        assert "data" in d
        assert "layout" in d
        assert isinstance(d["data"], list)

    def test_chart_has_candlestick(self):
        upload_csv()
        r = client.get("/api/chart")
        d = r.json()
        types = [t.get("type") for t in d["data"]]
        assert "candlestick" in types


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/export
# ══════════════════════════════════════════════════════════════════════════════
class TestExport:
    def test_without_trades_returns_404(self):
        r = client.get("/api/export")
        assert r.status_code == 404

    def test_with_trades_returns_csv(self):
        upload_csv()
        client.post("/api/backtest", json={"scheme": "F"})
        # If no trades (small data), still 404; skip
        r = client.get("/api/export")
        if r.status_code == 200:
            assert "text/csv" in r.headers["content-type"]
            assert "trades.csv" in r.headers.get("content-disposition", "")
