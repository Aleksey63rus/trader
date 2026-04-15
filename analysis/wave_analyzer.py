"""
Идентификатор 5-волновых импульсов Эллиотта.

5-волновой импульс = 6 свинг-точек:
  Восходящий: low₀ → high₁ → low₂ → high₃ → low₄ → high₅
  Нисходящий: high₀ → low₁ → high₂ → low₃ → high₄ → low₅

Правила Эллиотта (обязательные):
  1. Волна 2 не откатывает более 100% волны 1 (low₂ > low₀ для восходящего)
  2. Волна 3 не самая короткая среди волн 1, 3, 5
  3. Волна 4 не перекрывает territory волны 1 (low₄ > high₁ для восходящего)

Правила Фибоначчи (с допуском FIB_TOLERANCE):
  4. Волна 2 = 50–78.6% волны 1
  5. Волна 3 = 161.8% волны 1 (типичное отношение)
  6. Волна 4 = 38.2% волны 3
  7. Волна 5 ≈ волна 1 или 61.8% × (high₃ - low₀)
"""

from dataclasses import dataclass, field
from typing import Optional

from analysis.swing_detector import SwingPoint
from config import FIB_TOLERANCE

# Ключевые уровни Фибоначчи
FIB_236 = 0.236
FIB_382 = 0.382
FIB_500 = 0.500
FIB_618 = 0.618
FIB_786 = 0.786
FIB_1000 = 1.000
FIB_1618 = 1.618


@dataclass
class FibRatios:
    wave2_retracement: float   # откат волны 2 от волны 1
    wave3_extension: float     # расширение волны 3 относительно волны 1
    wave4_retracement: float   # откат волны 4 от волны 3
    wave5_vs_wave1: float      # соотношение длины волны 5 к волне 1


@dataclass
class ImpulseWave:
    direction: str             # "UP" или "DOWN"
    points: list[SwingPoint]   # 6 точек: p0…p5
    fib_ratios: FibRatios
    confidence_score: float    # 0.0–1.0: доля выполненных правил Фибоначчи
    violations: list[str] = field(default_factory=list)  # нарушенные правила (для отладки)

    @property
    def wave_lengths(self) -> dict[str, float]:
        p = self.points
        return {
            "W1": abs(p[1].price - p[0].price),
            "W2": abs(p[2].price - p[1].price),
            "W3": abs(p[3].price - p[2].price),
            "W4": abs(p[4].price - p[3].price),
            "W5": abs(p[5].price - p[4].price),
        }

    def __repr__(self) -> str:
        return (
            f"ImpulseWave({self.direction}, score={self.confidence_score:.2f}, "
            f"p0={self.points[0].price:.4f}…p5={self.points[5].price:.4f})"
        )


def _fib_check(ratio: float, target: float, tol: float) -> bool:
    return abs(ratio - target) <= tol


def find_impulse(
    swings: list[SwingPoint],
    fib_tolerance: float = FIB_TOLERANCE,
) -> list[ImpulseWave]:
    """
    Ищет все завершённые 5-волновые импульсы в списке свинг-точек.

    Проверяет каждое скользящее окно из 6 точек.
    Возвращает список ImpulseWave, прошедших обязательные правила Эллиотта.
    Сортирует по убыванию confidence_score.
    """
    results: list[ImpulseWave] = []

    for i in range(len(swings) - 5):
        window = swings[i : i + 6]
        wave = _check_window(window, fib_tolerance)
        if wave is not None:
            results.append(wave)

    results.sort(key=lambda w: w.confidence_score, reverse=True)
    return results


def _check_window(
    pts: list[SwingPoint],
    tol: float,
) -> Optional[ImpulseWave]:
    """Проверяет 6 свинг-точек на соответствие 5-волновому импульсу."""
    p0, p1, p2, p3, p4, p5 = pts

    # Определяем направление: первый свинг должен быть LOW (восходящий) или HIGH (нисходящий)
    if p0.kind == "LOW" and p1.kind == "HIGH":
        direction = "UP"
    elif p0.kind == "HIGH" and p1.kind == "LOW":
        direction = "DOWN"
    else:
        return None

    # Проверяем строгое чередование
    expected = (
        ["LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH"]
        if direction == "UP"
        else ["HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW"]
    )
    if [p.kind for p in pts] != expected:
        return None

    # Длины волн
    w1 = abs(p1.price - p0.price)
    w2 = abs(p2.price - p1.price)
    w3 = abs(p3.price - p2.price)
    w4 = abs(p4.price - p3.price)
    w5 = abs(p5.price - p4.price)

    if w1 == 0 or w3 == 0:
        return None

    violations: list[str] = []

    # --- Обязательные правила Эллиотта ---
    if direction == "UP":
        # Правило 1: волна 2 не откатывает более 100% волны 1
        if p2.price <= p0.price:
            return None
        # Правило 3: волна 4 не входит в зону волны 1
        if p4.price <= p1.price:
            return None
    else:
        if p2.price >= p0.price:
            return None
        if p4.price >= p1.price:
            return None

    # Правило 2: волна 3 не самая короткая
    if w3 < w1 and w3 < w5:
        return None

    # --- Правила Фибоначчи (влияют на confidence_score) ---
    fib_checks_passed = 0
    total_fib_checks = 4

    # Откат волны 2 (50–78.6% волны 1)
    w2_ret = w2 / w1
    if FIB_500 - tol <= w2_ret <= FIB_786 + tol:
        fib_checks_passed += 1
    else:
        violations.append(f"W2 retracement={w2_ret:.3f} (ожидалось 0.50–0.786)")

    # Расширение волны 3 (138.2–261.8% волны 1)
    w3_ext = w3 / w1
    if FIB_1618 * (1 - tol * 2) <= w3_ext <= FIB_1618 * (1 + tol * 2) or (
        1.382 - tol <= w3_ext <= 2.618 + tol
    ):
        fib_checks_passed += 1
    else:
        violations.append(f"W3 extension={w3_ext:.3f} (ожидалось ~1.618)")

    # Откат волны 4 (23.6–50% волны 3)
    w4_ret = w4 / w3
    if FIB_236 - tol <= w4_ret <= FIB_500 + tol:
        fib_checks_passed += 1
    else:
        violations.append(f"W4 retracement={w4_ret:.3f} (ожидалось 0.236–0.50)")

    # Волна 5 ≈ волна 1 (±tol) или 61.8% расстояния p0→p3
    w5_vs_w1 = w5 / w1
    p0_to_p3 = abs(p3.price - p0.price)
    w5_vs_proj = w5 / p0_to_p3 if p0_to_p3 > 0 else 0
    if (1 - tol <= w5_vs_w1 <= 1 + tol) or (FIB_618 - tol <= w5_vs_proj <= FIB_618 + tol):
        fib_checks_passed += 1
    else:
        violations.append(f"W5={w5:.4f} не равна W1={w1:.4f} и не 61.8% от p0→p3")

    confidence = fib_checks_passed / total_fib_checks

    fib_ratios = FibRatios(
        wave2_retracement=round(w2 / w1, 4),
        wave3_extension=round(w3 / w1, 4),
        wave4_retracement=round(w4 / w3, 4),
        wave5_vs_wave1=round(w5 / w1, 4),
    )

    return ImpulseWave(
        direction=direction,
        points=list(pts),
        fib_ratios=fib_ratios,
        confidence_score=round(confidence, 4),
        violations=violations,
    )
