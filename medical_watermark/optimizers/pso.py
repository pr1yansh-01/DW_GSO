"""Particle Swarm Optimization (scalar maximization)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def pso_maximize(
    fitness: Callable[[float], float],
    bounds: tuple[float, float],
    n_particles: int = 20,
    n_iter: int = 25,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int | None = 42,
) -> tuple[float, float, list[float]]:
    """
    Maximize ``fitness(x)`` for x in [low, high].

    Returns (best_x, best_fitness, history_best_fitness_each_iteration).
    """
    low, high = bounds
    rng = np.random.default_rng(seed)
    pos = rng.uniform(low, high, size=n_particles)
    vel = rng.uniform(-(high - low) * 0.2, (high - low) * 0.2, size=n_particles)
    pbest_pos = pos.copy()
    pbest_val = np.array([fitness(float(x)) for x in pos])
    g_idx = int(np.argmax(pbest_val))
    gbest_pos = float(pbest_pos[g_idx])
    gbest_val = float(pbest_val[g_idx])
    hist: list[float] = []

    for _ in range(n_iter):
        r1 = rng.random(n_particles)
        r2 = rng.random(n_particles)
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = np.clip(pos + vel, low, high)
        for i in range(n_particles):
            fv = fitness(float(pos[i]))
            if fv > pbest_val[i]:
                pbest_val[i] = fv
                pbest_pos[i] = pos[i]
        g_idx = int(np.argmax(pbest_val))
        gbest_val = float(pbest_val[g_idx])
        gbest_pos = float(pbest_pos[g_idx])
        hist.append(gbest_val)

    return gbest_pos, gbest_val, hist
