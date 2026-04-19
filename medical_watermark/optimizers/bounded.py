"""Bounded scalar maximization for one-dimensional alpha search."""

from __future__ import annotations

from collections.abc import Callable

from scipy.optimize import minimize_scalar


def bounded_maximize(
    fitness: Callable[[float], float],
    bounds: tuple[float, float],
    max_iter: int = 24,
    xatol: float = 1e-3,
) -> tuple[float, float, list[float]]:
    """
    Maximize ``fitness(x)`` for x in [low, high] with bounded scalar search.

    This uses the same objective as PSO, but is more suitable for this project
    because alpha is a single scalar parameter.
    """
    history: list[float] = []

    def objective(x: float) -> float:
        value = float(fitness(float(x)))
        best = max(history[-1], value) if history else value
        history.append(best)
        return -value

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"maxiter": int(max_iter), "xatol": float(xatol)},
    )
    best_alpha = float(result.x)
    best_fitness = float(-result.fun)
    if not history or history[-1] < best_fitness:
        history.append(best_fitness)
    return best_alpha, best_fitness, history
