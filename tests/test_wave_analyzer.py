"""Тесты волнового анализа (6 свингов, find_impulse)."""
from __future__ import annotations

import pandas as pd

from analysis.swing_detector import SwingPoint
from analysis.wave_analyzer import find_impulse


def _ts(i: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)


def test_find_impulse_up_ideal() -> None:
    swings = [
        SwingPoint(0, _ts(0), 100.0, "LOW"),
        SwingPoint(10, _ts(10), 110.0, "HIGH"),
        SwingPoint(20, _ts(20), 104.5, "LOW"),
        SwingPoint(35, _ts(35), 128.0, "HIGH"),
        SwingPoint(45, _ts(45), 118.0, "LOW"),
        SwingPoint(55, _ts(55), 133.0, "HIGH"),
    ]
    imps = find_impulse(swings)
    assert len(imps) >= 1
    assert imps[0].direction == "UP"
    assert imps[0].confidence_score >= 0.5


def test_find_impulse_rejects_overlap() -> None:
    swings = [
        SwingPoint(0, _ts(0), 100.0, "LOW"),
        SwingPoint(10, _ts(10), 110.0, "HIGH"),
        SwingPoint(20, _ts(20), 104.5, "LOW"),
        SwingPoint(35, _ts(35), 128.0, "HIGH"),
        SwingPoint(45, _ts(45), 108.0, "LOW"),
        SwingPoint(55, _ts(55), 133.0, "HIGH"),
    ]
    imps = find_impulse(swings)
    assert len(imps) == 0
