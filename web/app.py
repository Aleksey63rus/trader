"""
Momentum Filter Trader — FastAPI Web Application v2.0
=====================================================
Routes:
  GET  /                       main dashboard
  POST /api/upload             upload CSV
  GET  /api/download/{ticker}  download fresh data from MOEX ISS
  POST /api/backtest           run backtest (single ticker)
  POST /api/compare            run all 7 schemes on uploaded data
  GET  /api/chart              Plotly JSON chart
  GET  /api/schemes            list of available TP schemes
  GET  /api/status             current app state
  GET  /api/export             download backtest results as CSV
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plotly.subplots import make_subplots
from pydantic import BaseModel

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

import sys
sys.path.insert(0, str(ROOT_DIR))

from core.data_loader import fetch_moex, load_csv, save_csv
from core.strategy   import SCHEMES, BacktestEngine, BacktestResult

app = FastAPI(title="Momentum Filter Trader", version="2.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ── Global state ──────────────────────────────────────────────────────────────
_state: dict = {
    "df":        None,   # current DataFrame
    "ticker":    "",
    "result":    None,   # last BacktestResult
    "compare":   None,   # scheme comparison dict
    "equity":    [],
    "trades":    [],
}


# ══════════════════════════════════════════════════════════════════════════════
# Request schemas
# ══════════════════════════════════════════════════════════════════════════════
class BacktestReq(BaseModel):
    csv_path:  Optional[str] = None
    scheme:    str           = "F"
    max_hold:  int           = 96
    time_from: Optional[str] = None   # "07:00"
    time_to:   Optional[str] = None   # "23:00"


class DownloadReq(BaseModel):
    ticker:    str
    days:      int  = 365 * 3
    interval:  str  = "1H"


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ── Data management ───────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_csv(file: UploadFile):
    content = await file.read()
    tmp = BASE_DIR / "static" / "uploaded.csv"
    tmp.write_bytes(content)
    try:
        df = load_csv(tmp).between_time("07:00", "23:00")
        _state["df"]    = df
        _state["ticker"] = file.filename.split("_")[0].upper()
        _state["result"] = None
        _state["compare"] = None
        return {
            "ok":     True,
            "rows":   len(df),
            "ticker": _state["ticker"],
            "from":   str(df.index[0]),
            "to":     str(df.index[-1]),
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/download")
async def download_moex(req: DownloadReq):
    """Download fresh OHLCV from MOEX ISS and cache it."""
    try:
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=req.days)
        df = fetch_moex(req.ticker, from_dt, to_dt, req.interval)
        if df.empty:
            raise ValueError(f"No data returned for {req.ticker}")
        df = df.between_time("07:00", "23:00")

        out = ROOT_DIR / f"{req.ticker}_moex_{req.interval}.csv"
        save_csv(df, out)

        _state["df"]     = df
        _state["ticker"] = req.ticker
        _state["result"] = None
        _state["compare"] = None
        return {
            "ok":     True,
            "rows":   len(df),
            "ticker": req.ticker,
            "from":   str(df.index[0]),
            "to":     str(df.index[-1]),
            "saved":  str(out),
        }
    except Exception as e:
        raise HTTPException(400, str(e))


# ── Backtest ──────────────────────────────────────────────────────────────────
@app.post("/api/backtest")
async def run_backtest(req: BacktestReq):
    df = _get_df(req)
    scheme_key = req.scheme.upper()
    if scheme_key not in SCHEMES:
        raise HTTPException(400, f"Unknown scheme '{scheme_key}'. Valid: {list(SCHEMES)}")

    scheme = SCHEMES[scheme_key]
    engine = BacktestEngine(scheme=scheme, max_hold=req.max_hold)
    r      = engine.run(df, _state["ticker"])

    _state["result"] = r
    _state["equity"] = r.equity
    _state["trades"] = _serialise_trades(r.trade_list)

    avg_e = float(np.mean([t.entry for t in r.trade_list])) if r.trade_list else 1.0
    return _result_to_json(r, avg_e, scheme.label)


@app.post("/api/compare")
async def compare_schemes(req: BacktestReq):
    """Run all 7 schemes on the current data and return comparison table."""
    df = _get_df(req)
    engine   = BacktestEngine(max_hold=req.max_hold)
    results  = engine.run_scheme_comparison(df, _state["ticker"])
    _state["compare"] = results

    rows = []
    for key, r in results.items():
        avg_e = float(np.mean([t.entry for t in r.trade_list])) if r.trade_list else 1.0
        rows.append({
            "key":      key,
            "label":    r.scheme_label,
            "trades":   r.trades,
            "wr_pct":   round(r.wr * 100, 1),
            "total_pct":round(r.total_pct, 2),
            "pf":       round(r.profit_factor, 2),
            "sharpe":   round(r.sharpe, 2),
            "max_dd_pct": round(r.max_dd_pct, 2),
            "expectancy": round(r.expectancy, 3),
        })
    rows.sort(key=lambda x: -x["wr_pct"])
    return {"schemes": rows}


# ── Chart ──────────────────────────────────────────────────────────────────────
@app.get("/api/chart")
async def get_chart(max_candles: int = 700, scheme: str = "F"):
    if _state["df"] is None:
        raise HTTPException(400, "No data loaded.")

    df     = _state["df"].tail(max_candles).copy()
    trades = _state["trades"]
    equity = _state["equity"]

    # If result uses different scheme, recalculate signals for overlay only
    fig = _build_chart(df, trades, equity)
    return JSONResponse(content=json.loads(fig.to_json()))


# ── Utility ───────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    df = _state["df"]
    r  = _state["result"]
    return {
        "data_loaded":   df is not None,
        "rows":          len(df) if df is not None else 0,
        "ticker":        _state["ticker"],
        "backtest_done": r is not None,
        "trades":        len(_state["trades"]),
        "wr_pct":        round(r.wr * 100, 1) if r else None,
    }


@app.get("/api/schemes")
async def list_schemes():
    return {k: v.label for k, v in SCHEMES.items()}


@app.get("/api/export")
async def export_trades():
    """Download trade list as CSV."""
    trades = _state["trades"]
    if not trades:
        raise HTTPException(404, "No trades to export.")
    df_out = pd.DataFrame(trades)
    buf    = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trades.csv"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _get_df(req: BacktestReq) -> pd.DataFrame:
    df = _state["df"]
    if df is None:
        if req.csv_path:
            try:
                df = load_csv(req.csv_path).between_time("07:00", "23:00")
                _state["df"] = df
            except Exception as e:
                raise HTTPException(400, str(e))
        else:
            raise HTTPException(400, "Upload a CSV file first (/api/upload).")
    return df


def _serialise_trades(trade_list) -> list[dict]:
    return [
        {
            "idx":        i + 1,
            "entry_dt":   str(t.entry_dt)[:16],
            "exit_dt":    str(t.exit_dt)[:16],
            "entry":      round(float(t.entry), 2),
            "exit":       round(float(t.exit_price), 2),
            "pnl":        round(float(t.pnl), 2),
            "pnl_pct":    round(float(t.pnl_pct), 2),
            "hold":       int(t.hold_bars),
            "reason":     str(t.reason),
            "tps_hit":    int(t.tp_levels_hit),
            "win":        bool(t.win),
        }
        for i, t in enumerate(trade_list)
    ]


def _result_to_json(r: BacktestResult, avg_e: float, scheme_label: str) -> dict:
    return {
        "ticker":      _state["ticker"],
        "scheme":      scheme_label,
        "trades":      int(r.trades),
        "wins":        int(r.wins),
        "losses":      int(r.losses),
        "wr_pct":      round(float(r.wr) * 100, 1),
        "total_pct":   round(float(r.total_pct), 2),
        "avg_win":     round(float(r.avg_win), 2),
        "avg_loss":    round(float(r.avg_loss), 2),
        "pf":          round(float(r.profit_factor), 2),
        "max_dd_pct":  round(float(r.max_drawdown) / avg_e * 100, 2),
        "sharpe":      round(float(r.sharpe), 2),
        "expectancy":  round(float(r.expectancy), 3),
        "exit_dist":   {k: int(v) for k, v in r.exit_counts.items()},
        "equity":      r.equity,
        "trade_list":  _state["trades"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Chart builder
# ══════════════════════════════════════════════════════════════════════════════
def _build_chart(df: pd.DataFrame, trades: list, equity: list) -> go.Figure:
    from core.indicators import ema as _ema, atr as _atr

    has_equity = len(equity) > 1
    n_rows     = 3  # candles | volume | equity
    row_h      = [0.55, 0.15, 0.30] if has_equity else [0.70, 0.30, 0.0]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=row_h,
        subplot_titles=("Price  ·  EMA20 / EMA200  ·  Trades",
                        "Volume",
                        "Equity Curve (cumulative P&L)"),
        vertical_spacing=0.04,
    )

    dates = df.index.tolist()

    # ── 1. Candlestick ────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    for span, color, name in [(20, "#FFD54F", "EMA 20"), (200, "#42A5F5", "EMA 200")]:
        fig.add_trace(go.Scatter(
            x=dates, y=_ema(df["close"], span).values,
            name=name, mode="lines",
            line=dict(color=color, width=1.2), opacity=0.75,
        ), row=1, col=1)

    # ATR bands (SL reference)
    at14 = _atr(df, 14)
    sl_band = df["close"] - 1.8 * at14
    fig.add_trace(go.Scatter(
        x=dates, y=sl_band.values,
        name="SL band (1.8×ATR)", mode="lines",
        line=dict(color="#ef5350", width=0.8, dash="dot"), opacity=0.5,
    ), row=1, col=1)

    # ── Trade markers ─────────────────────────────────────────────────────────
    visible_set = set(str(d)[:16] for d in dates)
    for color, label, sym, subset in [
        ("#26a69a", "Entry WIN",  "triangle-up",   [t for t in trades if t["win"]]),
        ("#ef5350", "Entry loss", "triangle-down",  [t for t in trades if not t["win"]]),
    ]:
        xs, ys, txts = [], [], []
        for t in subset:
            if t["entry_dt"] not in visible_set:
                continue
            match = next((d for d in dates if str(d)[:16] == t["entry_dt"]), None)
            if match is None:
                continue
            xs.append(match)
            ys.append(t["entry"])
            txts.append(
                f"#{t['idx']} {t['reason']} TP:{t['tps_hit']}<br>"
                f"{t['pnl_pct']:+.2f}% · {t['hold']}h"
            )
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", name=label,
                marker=dict(symbol=sym, size=9, color=color,
                            line=dict(width=1, color="#0d1117")),
                text=txts, hoverinfo="text+x",
            ), row=1, col=1)

    # ── 2. Volume bars ────────────────────────────────────────────────────────
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=dates, y=df["volume"].values,
        name="Volume", marker_color=vol_colors, opacity=0.6,
        showlegend=False,
    ), row=2, col=1)

    # Volume MA
    vol_ma = df["volume"].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=dates, y=vol_ma.values,
        name="Vol MA20", mode="lines",
        line=dict(color="#FFD54F", width=1), opacity=0.8,
    ), row=2, col=1)

    # ── 3. Equity curve ───────────────────────────────────────────────────────
    if has_equity:
        eq_x = list(range(len(equity)))
        peak = np.maximum.accumulate(equity)
        dd   = [e - p for e, p in zip(equity, peak)]

        fig.add_trace(go.Scatter(
            x=eq_x, y=equity, name="Equity",
            mode="lines", line=dict(color="#7C4DFF", width=2),
            fill="tozeroy", fillcolor="rgba(124,77,255,0.10)",
        ), row=3, col=1)

        # Drawdown shading
        fig.add_trace(go.Scatter(
            x=eq_x, y=dd, name="Drawdown",
            mode="lines", line=dict(color="#ef5350", width=1),
            fill="tozeroy", fillcolor="rgba(239,83,80,0.12)",
        ), row=3, col=1)

        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                      line_width=1, row=3, col=1)

        # Annotate peak
        pk_val = float(max(equity))
        pk_idx = equity.index(pk_val) if pk_val in equity else 0
        if pk_val != 0:
            fig.add_annotation(
                x=pk_idx, y=pk_val, row=3, col=1,
                text=f"Peak {pk_val:+.1f}",
                showarrow=True, arrowhead=2, arrowcolor="#7C4DFF",
                font=dict(size=10, color="#a78bfa"), ax=40, ay=-20,
            )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(family="Inter,'Segoe UI',sans-serif", size=12, color="#c9d1d9"),
        height=820,
        margin=dict(l=60, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        barmode="overlay",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.03)", zeroline=False, showgrid=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.03)", zeroline=False, showgrid=True)

    return fig
