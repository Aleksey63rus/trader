"""
Unit-тесты для wave_analyzer на синтетических данных.
"""


from analysis.swing_detector import SwingPoint
from analysis.wave_analyzer import FibRatios, find_impulse, _check_window


def _sp(idx: int, price: float, kind: str) -> SwingPoint:
    return SwingPoint(idx=idx, timestamp=idx, price=price, kind=kind)


def _make_perfect_up_impulse() -> list[SwingPoint]:
    """
    Идеальный восходящий импульс Эллиотта с классическими пропорциями Фибоначчи.

    W1 = 100, W2 = 61.8% (откат), W3 = 161.8% W1, W4 = 38.2% W3, W5 = W1
    """
    p0 = 1000.0  # low₀
    p1 = p0 + 100          # high₁  W1=100
    p2 = p1 - 61.8         # low₂   W2=61.8% откат
    p3 = p2 + 161.8        # high₃  W3=161.8% W1
    p4 = p3 - 61.8         # low₄   W4=38.2% W3 (≈38.2%)
    p5 = p4 + 100          # high₅  W5=W1

    return [
        _sp(0, p0, "LOW"),
        _sp(10, p1, "HIGH"),
        _sp(20, p2, "LOW"),
        _sp(30, p3, "HIGH"),
        _sp(40, p4, "LOW"),
        _sp(50, p5, "HIGH"),
    ]


def _make_perfect_down_impulse() -> list[SwingPoint]:
    """Нисходящий импульс."""
    p0 = 2000.0  # high₀
    p1 = p0 - 100
    p2 = p1 + 61.8
    p3 = p2 - 161.8
    p4 = p3 + 61.8
    p5 = p4 - 100

    return [
        _sp(0, p0, "HIGH"),
        _sp(10, p1, "LOW"),
        _sp(20, p2, "HIGH"),
        _sp(30, p3, "LOW"),
        _sp(40, p4, "HIGH"),
        _sp(50, p5, "LOW"),
    ]


class TestCheckWindow:
    def test_perfect_up_impulse_passes(self):
        pts = _make_perfect_up_impulse()
        wave = _check_window(pts, tol=0.15)
        assert wave is not None
        assert wave.direction == "UP"

    def test_perfect_down_impulse_passes(self):
        pts = _make_perfect_down_impulse()
        wave = _check_window(pts, tol=0.15)
        assert wave is not None
        assert wave.direction == "DOWN"

    def test_wave2_violation_rejected(self):
        """Волна 2 откатывает более 100% волны 1 → отклонение."""
        pts = _make_perfect_up_impulse()
        # low₂ ниже low₀
        pts[2] = _sp(20, pts[0].price - 10, "LOW")
        wave = _check_window(pts, tol=0.10)
        assert wave is None

    def test_wave4_overlap_rejected(self):
        """Волна 4 перекрывает зону волны 1 → отклонение."""
        pts = _make_perfect_up_impulse()
        # low₄ ниже high₁
        pts[4] = _sp(40, pts[1].price - 5, "LOW")
        wave = _check_window(pts, tol=0.10)
        assert wave is None

    def test_wave3_shortest_rejected(self):
        """Волна 3 — самая короткая среди 1,3,5 → отклонение."""
        pts = _make_perfect_up_impulse()
        # Делаем W3 крошечной
        p3_tiny = pts[2].price + 5  # W3 = 5
        pts[3] = _sp(30, p3_tiny, "HIGH")
        pts[4] = _sp(40, p3_tiny - 10, "LOW")  # поправляем W4 чтобы не нарушать R1/R3
        pts[5] = _sp(50, p3_tiny - 10 + 100, "HIGH")  # W5 = 100 > W3
        wave = _check_window(pts, tol=0.10)
        assert wave is None

    def test_confidence_score_range(self):
        pts = _make_perfect_up_impulse()
        wave = _check_window(pts, tol=0.15)
        assert wave is not None
        assert 0.0 <= wave.confidence_score <= 1.0

    def test_fib_ratios_computed(self):
        pts = _make_perfect_up_impulse()
        wave = _check_window(pts, tol=0.15)
        assert wave is not None
        assert isinstance(wave.fib_ratios, FibRatios)
        assert wave.fib_ratios.wave2_retracement > 0
        assert wave.fib_ratios.wave3_extension > 0

    def test_wrong_alternation_rejected(self):
        """Если точки не чередуются LOW/HIGH — отклонение."""
        pts = _make_perfect_up_impulse()
        pts[1] = _sp(10, pts[1].price, "LOW")  # нарушаем чередование
        wave = _check_window(pts, tol=0.10)
        assert wave is None


class TestFindImpulse:
    def test_finds_impulse_in_long_list(self):
        """find_impulse должна найти импульс в более длинном списке свингов."""
        prefix = [
            _sp(0, 500, "LOW"),
            _sp(5, 600, "HIGH"),
            _sp(8, 550, "LOW"),
        ]
        impulse_pts = _make_perfect_up_impulse()
        # Сдвигаем индексы
        for i, sp in enumerate(impulse_pts):
            impulse_pts[i] = _sp(sp.idx + 20, sp.price, sp.kind)

        swings = prefix + impulse_pts
        results = find_impulse(swings, fib_tolerance=0.15)
        assert len(results) > 0

    def test_sorted_by_confidence(self):
        """Результаты отсортированы по убыванию confidence_score."""
        impulse = _make_perfect_up_impulse()
        results = find_impulse(impulse, fib_tolerance=0.15)
        for i in range(1, len(results)):
            assert results[i - 1].confidence_score >= results[i].confidence_score

    def test_empty_swings(self):
        assert find_impulse([]) == []

    def test_too_few_swings(self):
        swings = _make_perfect_up_impulse()[:4]
        assert find_impulse(swings) == []
