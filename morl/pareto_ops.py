"""Pareto utilities for vector-reward RL/MORL.

Core operator implemented here:

V*(s) = ND( U_a [ R(s,a) + gamma * V*(s') ] )

where ND keeps only non-dominated vectors.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple


Vector = Tuple[float, ...]


def dominates(a: Vector, b: Vector, epsilon: float = 0.0) -> bool:
    """True if a Pareto-dominates b under epsilon tolerance (maximize objectives)."""
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    weakly_better = all(x >= y - epsilon for x, y in zip(a, b))
    strictly_better = any(x > y + epsilon for x, y in zip(a, b))
    return weakly_better and strictly_better


def non_dominated_indices(vectors: Sequence[Vector], epsilon: float = 0.0) -> List[int]:
    nd: List[int] = []
    for i, v in enumerate(vectors):
        dominated = False
        for j, u in enumerate(vectors):
            if i == j:
                continue
            if dominates(u, v, epsilon=epsilon):
                dominated = True
                break
        if not dominated:
            nd.append(i)
    return nd


def dominance_depths(vectors: Sequence[Vector], epsilon: float = 0.0) -> List[int]:
    """Depth proxy: number of vectors that dominate each vector."""
    depths = [0] * len(vectors)
    for i, v in enumerate(vectors):
        for j, u in enumerate(vectors):
            if i == j:
                continue
            if dominates(u, v, epsilon=epsilon):
                depths[i] += 1
    return depths


def frontier_advantages(vectors: Sequence[Vector], epsilon: float = 0.0) -> List[float]:
    """Simple scalar advantage from Pareto status.

    - non-dominated -> +1
    - dominated -> negative, scaled by dominance depth
    """
    depths = dominance_depths(vectors, epsilon=epsilon)
    max_depth = max(depths, default=0)
    if max_depth == 0:
        return [1.0] * len(vectors)

    out: List[float] = []
    for d in depths:
        if d == 0:
            out.append(1.0)
        else:
            out.append(-float(d) / float(max_depth))
    return out


def _crowding_distance(points: Sequence[Vector]) -> List[float]:
    """NSGA-style crowding distance for frontier truncation."""
    n = len(points)
    if n == 0:
        return []
    if n <= 2:
        return [float("inf")] * n

    m = len(points[0])
    dist = [0.0] * n

    for obj in range(m):
        order = sorted(range(n), key=lambda i: points[i][obj])
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")

        lo = points[order[0]][obj]
        hi = points[order[-1]][obj]
        width = hi - lo
        if width <= 0:
            continue

        for k in range(1, n - 1):
            prev_val = points[order[k - 1]][obj]
            next_val = points[order[k + 1]][obj]
            dist[order[k]] += (next_val - prev_val) / width

    return dist


def pareto_prune(
    vectors: Sequence[Vector],
    epsilon: float = 0.0,
    max_points: int | None = None,
) -> List[Vector]:
    """Keep only non-dominated vectors; optionally cap frontier size."""
    nd = [vectors[i] for i in non_dominated_indices(vectors, epsilon=epsilon)]
    if max_points is None or len(nd) <= max_points:
        return nd

    distances = _crowding_distance(nd)
    keep = sorted(range(len(nd)), key=lambda i: distances[i], reverse=True)[:max_points]
    keep_set = set(keep)
    return [v for i, v in enumerate(nd) if i in keep_set]


def pareto_bellman_backup(
    reward: Vector,
    next_frontier: Sequence[Vector],
    gamma: float,
    epsilon: float = 0.0,
    max_points: int | None = None,
) -> List[Vector]:
    """Set-valued Bellman backup for one action.

    Returns ND({reward + gamma * v for v in next_frontier}).
    """
    if not next_frontier:
        return [reward]

    out: List[Vector] = []
    for v in next_frontier:
        if len(v) != len(reward):
            raise ValueError("dimension mismatch in Bellman backup")
        out.append(tuple(r + gamma * x for r, x in zip(reward, v)))
    return pareto_prune(out, epsilon=epsilon, max_points=max_points)


def bellman_union_over_actions(
    rewards_by_action: Sequence[Vector],
    next_frontiers_by_action: Sequence[Sequence[Vector]],
    gamma: float,
    epsilon: float = 0.0,
    max_points: int | None = None,
) -> List[Vector]:
    """V*(s) = ND( union_a [ R(s,a) + gamma * V(s') ] )."""
    if len(rewards_by_action) != len(next_frontiers_by_action):
        raise ValueError("action list mismatch")

    merged: List[Vector] = []
    for reward, nxt in zip(rewards_by_action, next_frontiers_by_action):
        merged.extend(
            pareto_bellman_backup(
                reward,
                nxt,
                gamma=gamma,
                epsilon=epsilon,
                max_points=max_points,
            )
        )
    return pareto_prune(merged, epsilon=epsilon, max_points=max_points)


def _estimate_hypervolume(
    points: Sequence[Vector],
    reference: Vector,
    samples: int,
    seed: int,
) -> float:
    if not points:
        return 0.0

    dim = len(points[0])
    upper = [max(p[d] for p in points) for d in range(dim)]
    for d in range(dim):
        if upper[d] <= reference[d]:
            return 0.0

    volume = 1.0
    for d in range(dim):
        volume *= upper[d] - reference[d]

    rng = random.Random(seed)
    dominated = 0

    for _ in range(samples):
        x = tuple(reference[d] + rng.random() * (upper[d] - reference[d]) for d in range(dim))
        is_dominated = any(all(p[d] >= x[d] for d in range(dim)) for p in points)
        if is_dominated:
            dominated += 1

    return volume * (dominated / float(samples))


def approx_hypervolume_contributions(
    vectors: Sequence[Vector],
    reference: Vector,
    samples: int = 5000,
    seed: int = 0,
) -> List[float]:
    """Monte-Carlo hypervolume contribution for each point.

    Returns contribution_i = HV(all) - HV(all except i).
    """
    if not vectors:
        return []

    hv_all = _estimate_hypervolume(vectors, reference, samples=samples, seed=seed)
    contrib: List[float] = []
    for i in range(len(vectors)):
        subset = [v for j, v in enumerate(vectors) if j != i]
        hv_wo_i = _estimate_hypervolume(subset, reference, samples=samples, seed=seed + 13 + i)
        contrib.append(hv_all - hv_wo_i)
    return contrib
