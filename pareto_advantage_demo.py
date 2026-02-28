#!/usr/bin/env python3
"""
Pareto reward vector demo for LLM-style preference learning.

Why this exists:
- Scalar rewards collapse multi-objective signal into one number.
- Vector rewards preserve objective-level structure.
- Under Pareto logic, two candidates can both be acceptable when incomparable.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


Vector = Tuple[float, ...]


@dataclass(frozen=True)
class Candidate:
    name: str
    reward: Vector


def dominates(a: Vector, b: Vector) -> bool:
    """Return True if a Pareto-dominates b (>= all dims and > at least one)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def compare_pareto(a: Vector, b: Vector) -> str:
    """Pairwise comparison under Pareto dominance."""
    if dominates(a, b):
        return "a_dominates"
    if dominates(b, a):
        return "b_dominates"
    return "incomparable"


def pareto_front(candidates: Sequence[Candidate]) -> List[Candidate]:
    front: List[Candidate] = []
    for i, c in enumerate(candidates):
        is_dominated = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if dominates(other.reward, c.reward):
                is_dominated = True
                break
        if not is_dominated:
            front.append(c)
    return front


def scalarize(v: Vector, weights: Sequence[float] | None = None) -> float:
    if weights is None:
        return sum(v) / len(v)
    return sum(w * x for w, x in zip(weights, v))


def scalar_advantages(candidates: Sequence[Candidate], weights: Sequence[float] | None = None) -> List[float]:
    scores = [scalarize(c.reward, weights) for c in candidates]
    baseline = sum(scores) / len(scores)
    return [s - baseline for s in scores]


def pareto_advantages(candidates: Sequence[Candidate]) -> List[float]:
    """
    A simple vector-aware advantage surrogate.

    Pairwise rule:
    - +1 if candidate dominates another
    -  0 if incomparable (both can be right)
    - -1 if dominated

    Final advantage = mean pairwise signal against all other candidates.
    """
    n = len(candidates)
    out: List[float] = []
    for i, c in enumerate(candidates):
        total = 0.0
        for j, other in enumerate(candidates):
            if i == j:
                continue
            rel = compare_pareto(c.reward, other.reward)
            if rel == "a_dominates":
                total += 1.0
            elif rel == "b_dominates":
                total -= 1.0
            else:
                total += 0.0
        out.append(total / (n - 1))
    return out


def random_candidates(n: int, dims: int, lo: int, hi: int, seed: int) -> List[Candidate]:
    rng = random.Random(seed)
    items: List[Candidate] = []
    for i in range(n):
        reward = tuple(float(rng.randint(lo, hi)) for _ in range(dims))
        items.append(Candidate(name=f"r{i+1}", reward=reward))
    return items


def print_candidates(candidates: Sequence[Candidate]) -> None:
    print("Candidates:")
    for c in candidates:
        print(f"  {c.name:>4} -> {c.reward}")


def print_pairwise(candidates: Sequence[Candidate]) -> None:
    print("\nPairwise Pareto relation:")
    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates):
            if i >= j:
                continue
            rel = compare_pareto(a.reward, b.reward)
            if rel == "a_dominates":
                msg = f"{a.name} dominates {b.name}"
            elif rel == "b_dominates":
                msg = f"{b.name} dominates {a.name}"
            else:
                msg = f"{a.name} and {b.name} are incomparable"
            print(f"  - {msg}")


def print_summary(candidates: Sequence[Candidate], weights: Sequence[float] | None = None) -> None:
    scalar_adv = scalar_advantages(candidates, weights)
    pareto_adv = pareto_advantages(candidates)
    front = {c.name for c in pareto_front(candidates)}

    print("\nAdvantages (scalar vs Pareto):")
    print("  name   scalar_adv   pareto_adv   on_pareto_front")
    for c, sa, pa in zip(candidates, scalar_adv, pareto_adv):
        mark = "yes" if c.name in front else "no"
        print(f"  {c.name:>4}   {sa:>10.3f}   {pa:>10.3f}   {mark}")


def parse_weights(text: str | None, dims: int) -> List[float] | None:
    if text is None:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    vals = [float(x) for x in parts]
    if len(vals) != dims:
        raise ValueError(f"weights must have exactly {dims} entries")
    s = sum(vals)
    if s == 0:
        raise ValueError("weights must not sum to 0")
    return [v / s for v in vals]


def curated_example() -> List[Candidate]:
    # Includes the exact pattern from your description.
    return [
        Candidate("r1", (1.0, 1.0, 1.0)),
        Candidate("r2", (1.0, 1.0, 2.0)),
        Candidate("r3", (1.0, 2.0, 1.0)),
        Candidate("r4", (2.0, 1.0, 1.0)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto reward-vector demo")
    parser.add_argument("--random", action="store_true", help="use random candidates instead of curated example")
    parser.add_argument("--n", type=int, default=8, help="number of random candidates")
    parser.add_argument("--dims", type=int, default=3, help="reward vector dimensions for random mode")
    parser.add_argument("--lo", type=int, default=0, help="minimum reward value in random mode")
    parser.add_argument("--hi", type=int, default=4, help="maximum reward value in random mode")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="comma-separated scalarization weights (e.g. 0.5,0.3,0.2)",
    )
    args = parser.parse_args()

    if args.random:
        if args.dims < 2:
            raise ValueError("dims must be >= 2")
        candidates = random_candidates(args.n, args.dims, args.lo, args.hi, args.seed)
        dims = args.dims
    else:
        candidates = curated_example()
        dims = len(candidates[0].reward)

    weights = parse_weights(args.weights, dims)

    print_candidates(candidates)
    print_pairwise(candidates)
    print_summary(candidates, weights)


if __name__ == "__main__":
    main()
