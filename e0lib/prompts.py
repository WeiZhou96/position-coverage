"""Prompt rendering. Every model input is a list of segments
("text", str) / ("image", path); model adapters turn segments into chat messages.

Wording follows E0 design v2, section 5. Do not edit templates after the freeze.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

HIDDEN = "[This image exists but is hidden from you.]"
INSTR_BIN = "Answer with Yes or No only."
INSTR_TER = "Answer with the option letter only."
LETTERS = ("A", "B", "C")
BIN_WORDS = ("zero", "one", "two or more")

REL_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "OR": ("Does at least one of the two images contain {X}?",
           "Is it true that Image 1 or Image 2 (or both) contains {X}?"),
    "AND": ("Do both images contain {X}?",
            "Is it true that Image 1 and Image 2 each contain {X}?"),
    "XOR": ("Does exactly one of the two images contain {X}?",
            "Is it true that one image contains {X} and the other does not?"),
    "GE": ("Place the number of {Xs} in each image into one of three bins: zero, one, or two or more. "
           "Is the bin for Image 1 at least as high as the bin for Image 2?",
           "Using the bins zero, one, and two or more for the number of {Xs}, "
           "is Image 1's bin greater than or equal to Image 2's bin?"),
    "GT": ("Place the number of {Xs} in each image into one of three bins: zero, one, or two or more. "
           "Is the bin for Image 1 strictly higher than the bin for Image 2?",
           "Using the bins zero, one, and two or more for the number of {Xs}, "
           "is Image 1's bin greater than Image 2's bin?"),
    "EQ": ("Place the number of {Xs} in each image into one of three bins: zero, one, or two or more. "
           "Are the two images in the same bin?",
           "Using the bins zero, one, and two or more for the number of {Xs}, "
           "do the two images fall into the same bin?"),
}

PROBE_PRESENCE_SINGLE = ("Does this image contain {X}?", "Is there {X} in this image?")
PROBE_PRESENCE_CTX = ("Does Image {k} contain {X}?", "Is there {X} in Image {k}?")
PROBE_COUNT_SINGLE = ("How many {Xs} are there in this image?",
                      "Which bin does the number of {Xs} in this image fall into?")
PROBE_COUNT_CTX = ("How many {Xs} are there in Image {k}?",
                   "Which bin does the number of {Xs} in Image {k} fall into?")

TER_REL_OPTIONS = ("yes", "no", "undetermined")
TER_REL_TEXT = {"yes": "Yes", "no": "No", "undetermined": "Cannot be determined from the visible image(s)"}


def cyclic_perm(items: Sequence[str], p: int) -> List[str]:
    p %= len(items)
    return list(items[p:]) + list(items[:p])


def options_block(semantics: Sequence[str], text_of: Dict[str, str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {text_of[s]}" for i, s in enumerate(semantics))


def fill(t: str, phrase: Dict[str, str], k: int = 0) -> str:
    return t.format(X=phrase["singular"], Xs=phrase["plural"], k=k)


def slot_segments(paths: Sequence[str], visible: Sequence[bool]) -> List[Tuple[str, str]]:
    segs: List[Tuple[str, str]] = []
    for i, (path, vis) in enumerate(zip(paths, visible), start=1):
        segs.append(("text", ("" if i == 1 else "\n") + f"Image {i}: "))
        segs.append(("image", path) if vis else ("text", HIDDEN))
    segs.append(("text", "\n"))
    return segs


def relation_request(pred: str, phrase: Dict[str, str], template: int, paths: Sequence[str],
                     visible: Sequence[bool], fmt: str, perm: int = 0):
    """Relation question. fmt 'bin' -> candidates Yes/No; 'ter' -> letters with a cyclic option order."""
    q = fill(REL_TEMPLATES[pred][template], phrase)
    segs = slot_segments(paths, visible)
    if fmt == "bin":
        segs.append(("text", f"{q}\n{INSTR_BIN}"))
        return segs, ["Yes", "No"], ["yes", "no"]
    sem = cyclic_perm(TER_REL_OPTIONS, perm)
    segs.append(("text", f"{q}\n{options_block(sem, TER_REL_TEXT)}\n{INSTR_TER}"))
    return segs, list(LETTERS), sem


def text_control_request(pred: str, phrase: Dict[str, str], template: int, block: str, v1: int, v2: int):
    lines = []
    for k, v in ((1, v1), (2, v2)):
        if block == "presence":
            lines.append(f"Image {k} contains {phrase['singular']}." if v else
                         f"Image {k} does not contain {phrase['singular']}.")
        else:
            lines.append(f"The number of {phrase['plural']} in Image {k} is in the bin: {BIN_WORDS[v]}.")
    q = fill(REL_TEMPLATES[pred][template], phrase)
    segs = [("text", "The two images are described below instead of being shown.\n"
             + "\n".join(lines) + f"\n{q}\n{INSTR_BIN}")]
    return segs, ["Yes", "No"], ["yes", "no"]


def probe_single_request(block: str, phrase: Dict[str, str], template: int, path: str, perm: int = 0):
    if block == "presence":
        segs = [("image", path), ("text", "\n" + fill(PROBE_PRESENCE_SINGLE[template], phrase) + f"\n{INSTR_BIN}")]
        return segs, ["Yes", "No"], [1, 0]
    sem = cyclic_perm(("0", "1", "2"), perm)
    opts = "\n".join(f"{LETTERS[i]}. {BIN_WORDS[int(s)]}" for i, s in enumerate(sem))
    segs = [("image", path), ("text", "\n" + fill(PROBE_COUNT_SINGLE[template], phrase) + f"\n{opts}\n{INSTR_TER}")]
    return segs, list(LETTERS), [int(s) for s in sem]


def probe_ctx_request(block: str, phrase: Dict[str, str], template: int, paths: Sequence[str], k: int, perm: int = 0):
    segs = slot_segments(paths, (True, True))
    if block == "presence":
        segs.append(("text", fill(PROBE_PRESENCE_CTX[template], phrase, k) + f"\n{INSTR_BIN}"))
        return segs, ["Yes", "No"], [1, 0]
    sem = cyclic_perm(("0", "1", "2"), perm)
    opts = "\n".join(f"{LETTERS[i]}. {BIN_WORDS[int(s)]}" for i, s in enumerate(sem))
    segs.append(("text", fill(PROBE_COUNT_CTX[template], phrase, k) + f"\n{opts}\n{INSTR_TER}"))
    return segs, list(LETTERS), [int(s) for s in sem]
