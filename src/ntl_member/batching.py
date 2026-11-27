"""Deterministic logical prompt batches and physical padded-token limits."""

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence


def padded_batches(rows: Iterable[int], lengths: Sequence[int], budget: int) -> list[list[int]]:
    batches: list[list[int]] = []
    batch: list[int] = []
    width = 0
    for row in rows:
        n = lengths[row]
        if not 1 < n <= budget:
            raise ValueError("sequence is too short for masked pooling or exceeds token budget")
        candidate_width = max(n, width)
        if batch and (len(batch) + 1) * candidate_width > budget:
            batches.append(batch)
            batch, width = [], 0
        batch.append(row)
        width = max(n, width)
    if batch:
        batches.append(batch)
    return batches


def plan_steps(
    lengths: Sequence[int],
    groups: Sequence[int],
    budget: int,
    *,
    seed: int,
    pool: bool,
    target: int,
) -> list[list[list[int]]]:
    """An optimizer step contains logical batches. Official batches retain entire prompts."""
    if len(lengths) != len(groups) or not lengths or target < 1:
        raise ValueError("invalid lengths, group assignment or accumulation target")
    by_group: dict[int, list[int]] = defaultdict(list)
    for row, group in enumerate(groups):
        by_group[group].append(row)
    units = list(by_group.values())
    rng = random.Random(seed)
    rng.shuffle(units)
    if pool:
        # Keep an ordinary prompt together. Split only oversized external groups.
        micro: list[list[int]] = []
        batch: list[int] = []
        width = 0
        for unit in units:
            for part in padded_batches(unit, lengths, budget):
                part_width = max(lengths[row] for row in part)
                if batch and (len(batch) + len(part)) * max(width, part_width) > budget:
                    micro.append(batch)
                    batch, width = [], 0
                batch.extend(part)
                width = max(width, part_width)
        if batch:
            micro.append(batch)
    else:
        micro = units
    steps: list[list[list[int]]] = []
    current: list[list[int]] = []
    size = 0
    for batch in micro:
        current.append(batch)
        size += len(batch) if pool else 1
        if size >= target:
            steps.append(current)
            current, size = [], 0
    if current:
        steps.append(current)
    return steps
