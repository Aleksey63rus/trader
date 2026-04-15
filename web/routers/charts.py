"""График свечей + свинги (Plotly JSON)."""
from __future__ import annotations

import json
import logging

import plotly.graph_objects as go
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from analysis.indicators import enrich_ohlc
from analysis.swing_detector import find_swings
from analysis.wave_analyzer import find_impulses
from backtesting.engine import demo_ohlc_frame

log = logging.getLogger("investor.web")

router = APIRouter(prefix="/chart", tags=["chart"])


@router.get("/{symbol}/data")
async def chart_data(symbol: str, bars: int = 220) -> JSONResponse:
    """JSON Plotly Figure для отображения на фронте."""
    df = demo_ohlc_frame(n=max(80, min(bars, 2000)), seed=hash(symbol) % 2**32)
    data = enrich_ohlc(df)
    swings = find_swings(data)
    impulses = find_impulses(swings)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["datetime"],
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name=symbol,
            )
        ]
    )

    for s in swings:
        color = "#2563eb" if s.kind == "high" else "#dc2626"
        fig.add_trace(
            go.Scatter(
                x=[data["datetime"].iloc[s.index]],
                y=[s.price],
                mode="markers",
                marker={"size": 9, "color": color, "symbol": "circle"},
                name=f"swing {s.kind}",
                showlegend=False,
            )
        )

    for imp in impulses[-3:]:
        xs = [data["datetime"].iloc[p.index] for p in imp.points]
        ys = [p.price for p in imp.points]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line={"width": 2, "dash": "solid"},
                name=f"impulse {imp.direction} ({imp.confidence:.0%})",
            )
        )

    fig.update_layout(
        title=f"{symbol} — демо-свечи (синтетика)",
        xaxis_title="Время",
        yaxis_title="Цена",
        template="plotly_white",
        height=640,
        xaxis_rangeslider_visible=False,
    )
    # to_json() даёт сериализуемый JSON (в т.ч. numpy → list)
    return JSONResponse(content=json.loads(fig.to_json()))


@router.get("/{symbol}/view", response_class=HTMLResponse)
async def chart_view(symbol: str) -> str:
    """Страница с Plotly.js (CDN)."""
    return f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>График {symbol}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head><body style="font-family:system-ui;margin:1rem">
<p><a href="/">← Главная</a> · <a href="/trades">Сделки</a></p>
<div id="g" style="width:100%;height:680px"></div>
<script>
fetch('/chart/{symbol}/data').then(r=>r.json()).then(fig => {{
  Plotly.newPlot('g', fig.data, fig.layout, {{responsive:true}});
}});
</script>
</body></html>"""
