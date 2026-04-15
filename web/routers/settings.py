"""Публичные настройки стратегии (без секретов)."""
from __future__ import annotations

import config
import constants as C
from fastapi import APIRouter

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
async def public_settings() -> dict:
    return {
        "symbols": config.SYMBOLS,
        "interval": config.INTERVAL,
        "risk_percent": config.RISK_PERCENT,
        "max_positions": config.MAX_POSITIONS,
        "atr_period": C.ATR_PERIOD,
        "lookback": C.LOOKBACK_WINDOW,
        "fib_tolerance": C.FIB_TOLERANCE,
        "min_wave_confidence": C.MIN_CONFIDENCE_THRESHOLD,
    }
