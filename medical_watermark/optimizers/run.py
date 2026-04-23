"""Grey Wolf Optimizer (scalar maximization, 1-D search)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def gwo_maximize(
    fitness: Callable[[float], float],
    bounds: tuple[float, float],
    n_wolves: int = 20,
    n_iter: int = 25,
    seed: int | None = 42,
) -> tuple[float, float, list[float]]:
    """
    Maximize ``fitness(x)`` for x in [low, high].

    Returns (best_x, best_fitness, history_best_fitness_each_iteration).
    """
    low, high = bounds
    rng = np.random.default_rng(seed)
    pos = rng.uniform(low, high, size=n_wolves).astype(np.float64)
    hist: list[float] = []

    def eval_all() -> np.ndarray:
        return np.array([fitness(float(x)) for x in pos])

    fit = eval_all()
    order = np.argsort(-fit)
    alpha_p = float(pos[order[0]])
    beta_p = float(pos[order[1]])
    delta_p = float(pos[order[2]])
    hist.append(float(fit[order[0]]))

    for t in range(1, n_iter):
        a = 2.0 * (1.0 - t / max(n_iter - 1, 1))
        for i in range(n_wolves):
            X = pos[i]
            r1, r2 = rng.random(), rng.random()
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * rng.random()
            X1 = alpha_p - A1 * abs(C1 * alpha_p - X)
            r1, r2 = rng.random(), rng.random()
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * rng.random()
            X2 = beta_p - A2 * abs(C2 * beta_p - X)
            r1, r2 = rng.random(), rng.random()
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * rng.random()
            X3 = delta_p - A3 * abs(C3 * delta_p - X)
            pos[i] = float(np.clip((X1 + X2 + X3) / 3.0, low, high))

        fit = eval_all()
        order = np.argsort(-fit)
        alpha_p = float(pos[order[0]])
        beta_p = float(pos[order[1]])
        delta_p = float(pos[order[2]])
        hist.append(float(fit[order[0]]))

    return alpha_p, float(fit[order[0]]), hist
