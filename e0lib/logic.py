"""Predicates, logical sufficiency typing and the ideal-integrator null model.

Everything here is pure Python so that the type tables in the design document
can be verified without any model or dataset.
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, Sequence, Tuple

PRESENCE_DOMAIN: Tuple[int, ...] = (0, 1)        # 0 = absent, 1 = present
BIN_DOMAIN: Tuple[int, ...] = (0, 1, 2)          # 0 = zero, 1 = one, 2 = two or more

PREDICATES: Dict[str, Callable[[int, int], bool]] = {
    "OR": lambda a, b: bool(a or b),
    "AND": lambda a, b: bool(a and b),
    "XOR": lambda a, b: a != b,
    "GE": lambda a, b: a >= b,
    "GT": lambda a, b: a > b,
    "EQ": lambda a, b: a == b,
}

# Predicate families used by each block.
BLOCK_PREDICATES: Dict[str, Tuple[str, ...]] = {
    "presence": ("OR", "AND", "XOR"),
    "count": ("GE", "GT", "EQ"),
}
BLOCK_DOMAIN: Dict[str, Tuple[int, ...]] = {
    "presence": PRESENCE_DOMAIN,
    "count": BIN_DOMAIN,
}
SYMMETRIC = {"OR", "AND", "XOR", "EQ"}


def answer(pred: str, v1: int, v2: int) -> bool:
    return PREDICATES[pred](v1, v2)


def sufficiency(pred: str, v1: int, v2: int, domain: Sequence[int]) -> Tuple[bool, bool]:
    """(s1, s2): whether slot 1 / slot 2 alone fixes the answer over the logical domain."""
    f = PREDICATES[pred]
    s1 = len({f(v1, x) for x in domain}) == 1
    s2 = len({f(x, v2) for x in domain}) == 1
    return s1, s2


def suff_type(s1: bool, s2: bool) -> str:
    if s1 and s2:
        return "redundant"
    if s1:
        return "unique1"
    if s2:
        return "unique2"
    return "synergistic"


def coarse_type(t: str) -> str:
    return "unique" if t.startswith("unique") else t


def type_table(pred: str, domain: Sequence[int]) -> Dict[Tuple[int, int], Tuple[bool, str]]:
    out = {}
    for v1, v2 in itertools.product(domain, domain):
        s1, s2 = sufficiency(pred, v1, v2, domain)
        out[(v1, v2)] = (answer(pred, v1, v2), suff_type(s1, s2))
    return out


def partial_target(pred: str, v1: int, v2: int, domain: Sequence[int], visible: str):
    """Correct ternary target when only some slots are visible.

    visible: 'both' | 'only1' | 'only2' | 'none'.
    Returns 'yes' / 'no' / 'undetermined'.
    """
    if visible == "both":
        return "yes" if answer(pred, v1, v2) else "no"
    s1, s2 = sufficiency(pred, v1, v2, domain)
    determined = {"only1": s1, "only2": s2, "none": False}[visible]
    if not determined:
        return "undetermined"
    return "yes" if answer(pred, v1, v2) else "no"


def ideal_integrator_accuracy(pred: str, truth: bool, q1: Dict[int, float], q2: Dict[int, float]) -> float:
    """Expected accuracy of a logically perfect integrator whose per-slot readings
    are drawn independently from q1, q2 (distributions over atom values)."""
    f = PREDICATES[pred]
    total = 0.0
    for v1, p1 in q1.items():
        for v2, p2 in q2.items():
            if f(v1, v2) == truth:
                total += p1 * p2
    return total


def pipeline_answer(pred: str, r1: int, r2: int) -> bool:
    """Answer of the read-then-compute pipeline given argmax readings."""
    return answer(pred, r1, r2)


def presented(v1: int, v2: int, order: str) -> Tuple[int, int]:
    """Atom values as presented to the model ('normal' keeps slots, 'swap' exchanges them)."""
    return (v1, v2) if order == "normal" else (v2, v1)


def all_tables() -> Dict[str, Dict[Tuple[int, int], Tuple[bool, str]]]:
    tables = {}
    for block, preds in BLOCK_PREDICATES.items():
        for p in preds:
            tables[p] = type_table(p, BLOCK_DOMAIN[block])
    return tables


def iter_block_cells(block: str) -> Iterable[Tuple[int, int]]:
    d = BLOCK_DOMAIN[block]
    return itertools.product(d, d)
