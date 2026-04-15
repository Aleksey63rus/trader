"""Генерация торговых сигналов из волнового анализа."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

import constants as C
from analysis.wave_analyzer import ImpulseWave


@dataclass
class Signal:
    symbol: str
    direction: str  # 'BUY' | 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    confidence: float
    wave: ImpulseWave | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reason": self.reason,
            "confidence": self.confidence,
            "wave": self.wave.to_dict() if self.wave else None,
        }


def _rsi_divergence_bearish(rsi: pd.Series, p3_idx: int, p5_idx: int) -> bool:
    """Цена: новый максимум на p5 vs p3; RSI на p5 ниже, чем на p3."""
    if p3_idx < 0 or p5_idx >= len(rsi):
        return False
    r3 = rsi.iloc[p3_idx]
    r5 = rsi.iloc[p5_idx]
    if pd.isna(r3) or pd.isna(r5):
        return False
    return r5 < r3


def generate_signal(
    df: pd.DataFrame,
    wave: ImpulseWave | None,
    symbol: str = "DEMO",
    rsi_series: pd.Series | None = None,
) -> Signal | None:
    """
    BUY: завершённый UP-импульс, подтверждение close > p3.
    SELL: DOWN-импульс, close < p3.
    Опционально: медвежья дивергенция RSI на UP для усиления SELL-фильтра (здесь только для BUY ослабляет агрессию — не используем как блокер).
    """
    if wave is None or len(df) == 0:
        return None

    last_close = float(df["close"].iloc[-1])
    last_idx = len(df) - 1
    p0, p1, p2, p3, p4, p5 = wave.points

    if last_idx < p5.index:
        return None

    atr = float(df["atr14"].iloc[-1]) if "atr14" in df.columns and pd.notna(df["atr14"].iloc[-1]) else 0.0
    buf = C.STOP_BUFFER_ATR * atr if atr > 0 else 0.01 * last_close

    if wave.direction == "UP":
        if last_close <= p3.price:
            return None
        if last_close <= p4.price:
            return None
        stop = p4.price - buf
        impulse_len = p5.price - p0.price
        tp = last_close + C.FIB_LEVELS["tp_extension"] * max(impulse_len, 1e-9)
        reason = f"UP импульс, подтверждение close>{p3.price:.4f}, стоп за W4"
        if C.RSI_DIVERGENCE_ENABLED and rsi_series is not None and _rsi_divergence_bearish(rsi_series, p3.index, p5.index):
            reason += "; RSI медвежья дивергенция на W5"
        return Signal(
            symbol=symbol,
            direction="BUY",
            entry_price=last_close,
            stop_loss=stop,
            take_profit=tp,
            reason=reason,
            confidence=wave.confidence,
            wave=wave,
        )

    if wave.direction == "DOWN":
        if last_close >= p3.price:
            return None
        if last_close >= p4.price:
            return None
        stop = p4.price + buf
        impulse_len = p0.price - p5.price
        tp = last_close - C.FIB_LEVELS["tp_extension"] * max(impulse_len, 1e-9)
        reason = f"DOWN импульс, подтверждение close<{p3.price:.4f}, стоп за W4"
        return Signal(
            symbol=symbol,
            direction="SELL",
            entry_price=last_close,
            stop_loss=stop,
            take_profit=tp,
            reason=reason,
            confidence=wave.confidence,
            wave=wave,
        )

    return None
