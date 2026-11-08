"""E3: n-image presence operators, rotated patterns, the n-image ideal integrator (the preregistered analysis plan)."""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

from . import prompts, s1

NWORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
OPS = ("ALL", "ANY", "ONE", "NONE", "ATLEAST2")
TRAIN3_OPS = ("ALL", "ANY", "ONE", "NONE")
TEMPLATES: Dict[str, str] = {
    "ALL": "Do all of the {N} images contain {X}?",
    "ANY": "Does at least one of the {N} images contain {X}?",
    "ONE": "Does exactly one of the {N} images contain {X}?",
    "NONE": "Is it true that none of the {N} images contains {X}?",
    "ATLEAST2": "Do at least two of the {N} images contain {X}?",
}
# Base patterns over image indices 0..n-1 (1 = image contains the target); duplicates removed per n.
BASE = ("Z", "A1", "A2", "B2", "B1", "F")


def answer(op: str, vals: Sequence[int]) -> bool:
    k, n = sum(vals), len(vals)
    return {"ALL": k == n, "ANY": k >= 1, "ONE": k == 1, "NONE": k == 0, "ATLEAST2": k >= 2}[op]


def ops_for(n: int):
    return OPS if n > 2 else tuple(o for o in OPS if o != "ATLEAST2")


def question(op: str, n: int, phrase: Dict[str, str]) -> str:
    return TEMPLATES[op].format(N=NWORD[n], X=phrase["singular"])


def base_patterns(n: int) -> Dict[str, List[int]]:
    """name -> per-image-index values; for n = 2 only the four distinct patterns are kept."""
    pats = {"Z": [0] * n, "A1": [1] + [0] * (n - 1), "A2": [1, 1] + [0] * (n - 2),
            "B2": [0, 0] + [1] * (n - 2), "B1": [0] + [1] * (n - 1), "F": [1] * n}
    out, seen = {}, set()
    for name in BASE:
        t = tuple(pats[name])
        if t not in seen:
            seen.add(t)
            out[name] = pats[name]
    return out


def arrange(n: int, r: int) -> List[int]:
    """Image index shown in each slot under rotation r: image 0 sits in slot r (0-based), cyclic order kept."""
    return [(s - r) % n for s in range(n)]


def ideal(op: str, truth: bool, qs: Sequence[Dict[int, float]]) -> float:
    """Expected accuracy of a logically perfect integrator with independent per-image readings qs."""
    tot = 0.0
    for vals in itertools.product((0, 1), repeat=len(qs)):
        p = 1.0
        for q, v in zip(qs, vals):
            p *= q.get(v, 0.0)
        tot += p * (answer(op, vals) == truth)
    return tot


def teacher_segments(paths, q, block, phrase, read_vals):
    """As s1.teacher_segments, with the number of images spelled out (identical text for two images)."""
    facts = "\n".join(s1.fact_lines(block, phrase, read_vals))
    return s1.image_segments(paths) + [("text", f"Facts about the {NWORD[len(paths)]} images:\n{facts}\n{q}\n"
                                                f"{prompts.INSTR_BIN}")]
