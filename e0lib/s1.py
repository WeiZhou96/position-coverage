"""S1 screening: shared prompts, three-image logic and a minimal LoRA (preregistered).

Arms differ only in the training signal; images, questions, templates and optimiser steps are shared.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

from . import prompts

TRAIN_OPS = {"presence": ("AND", "OR"), "count": ("GE", "GT")}
HELDOUT_OPS = {"presence": ("XOR",), "count": ("EQ",)}

# Five paraphrases per operator that never appear in training (held-out wording).
_CNT = ("Count the {Xs} in each image and put the count into one of three bins: zero, one, or two or more. ",
        "With the bins zero, one, and two or more for the number of {Xs}, ",
        "Group the number of {Xs} per image as zero, one, or two or more. ",
        "Using three bins (zero, one, two or more) for how many {Xs} each image has, ",
        "Classify each image by its number of {Xs}: zero, one, or two or more. ")
HO_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "OR": ("Can {X} be found in Image 1, in Image 2, or in both?",
           "Is {X} present in either of the two images?",
           "Looking at both pictures, does {X} appear in at least one of them?",
           "Would it be correct to say that {X} is visible in one or both of the images?",
           "Is there {X} in Image 1 or in Image 2?"),
    "AND": ("Can {X} be found in Image 1 and also in Image 2?",
            "Is {X} present in each of the two images?",
            "Looking at both pictures, does {X} appear in every one of them?",
            "Would it be correct to say that {X} is visible in both images?",
            "Is there {X} in Image 1 as well as in Image 2?"),
    "XOR": ("Can {X} be found in only one of the two images?",
            "Is {X} present in one image but absent from the other?",
            "Looking at both pictures, does {X} appear in exactly one of them?",
            "Would it be correct to say that just one of the two images shows {X}?",
            "Is there {X} in one of the images and not in the other?"),
    "GE": (_CNT[0] + "Does Image 1 fall into the same bin as Image 2 or a higher one?",
           _CNT[1] + "is Image 2's bin no higher than Image 1's bin?",
           _CNT[2] + "Is it true that Image 1's group is not lower than Image 2's group?",
           _CNT[3] + "does Image 1 reach at least the bin of Image 2?",
           _CNT[4] + "Is Image 1's class greater than or equal to Image 2's class?"),
    "GT": (_CNT[0] + "Does Image 1 fall into a higher bin than Image 2?",
           _CNT[1] + "is Image 2's bin lower than Image 1's bin?",
           _CNT[2] + "Is it true that Image 1's group is above Image 2's group?",
           _CNT[3] + "does Image 1 land in a strictly higher bin than Image 2?",
           _CNT[4] + "Is Image 1's class strictly greater than Image 2's class?"),
    "EQ": (_CNT[0] + "Do both images fall into the same bin?",
           _CNT[1] + "is Image 1's bin identical to Image 2's bin?",
           _CNT[2] + "Is it true that the two images belong to the same group?",
           _CNT[3] + "do Image 1 and Image 2 land in the same bin?",
           _CNT[4] + "Do the two images have the same class?"),
}

# Three-image presence questions (never trained).
THREE_OPS = ("ALL", "ANY", "ONE", "NONE")
THREE_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "ALL": ("Do all three images contain {X}?",
            "Is it true that Image 1, Image 2 and Image 3 each contain {X}?"),
    "ANY": ("Does at least one of the three images contain {X}?",
            "Is it true that at least one of Image 1, Image 2 and Image 3 contains {X}?"),
    "ONE": ("Does exactly one of the three images contain {X}?",
            "Is it true that one of the three images contains {X} and the other two do not?"),
    "NONE": ("Do none of the three images contain {X}?",
             "Is it true that no image among Image 1, Image 2 and Image 3 contains {X}?"),
}


def three_answer(op: str, vals: Sequence[int]) -> bool:
    n = sum(vals)
    return {"ALL": n == len(vals), "ANY": n >= 1, "ONE": n == 1, "NONE": n == 0}[op]


# State-then-answer instruction (arm D and the zero-shot reference); written for any number of images.
SRT_INSTR = {
    "presence": "First state, for each image separately, whether it contains {X}. "
                "Then give the final answer on a new line as 'Answer: Yes' or 'Answer: No'.",
    "count": "First state, for each image separately, which bin (zero, one, or two or more) the number of {Xs} "
             "falls into. Then give the final answer on a new line as 'Answer: Yes' or 'Answer: No'.",
}


def fact_lines(block: str, phrase: Dict[str, str], vals: Sequence[int]) -> List[str]:
    """Same wording as the E0 text control and the facts probe."""
    out = []
    for k, v in enumerate(vals, start=1):
        if block == "presence":
            out.append(f"Image {k} contains {phrase['singular']}." if v else
                       f"Image {k} does not contain {phrase['singular']}.")
        else:
            out.append(f"The number of {phrase['plural']} in Image {k} is in the bin: {prompts.BIN_WORDS[v]}.")
    return out


def image_segments(paths: Sequence[str], visible: Sequence[bool] = None):
    visible = visible or [True] * len(paths)
    segs = []
    for i, (p, vis) in enumerate(zip(paths, visible), start=1):
        segs.append(("text", ("" if i == 1 else "\n") + f"Image {i}: "))
        segs.append(("image", p) if vis else ("text", prompts.HIDDEN))
    segs.append(("text", "\n"))
    return segs


def student_segments(paths, q, visible=None):
    return image_segments(paths, visible) + [("text", f"{q}\n{prompts.INSTR_BIN}")]


def teacher_segments(paths, q, block, phrase, read_vals):
    """C3 teacher: the same images plus the model's own per-image readings as full sentences BEFORE the question."""
    facts = "\n".join(fact_lines(block, phrase, read_vals))
    return image_segments(paths) + [("text", f"Facts about the two images:\n{facts}\n{q}\n{prompts.INSTR_BIN}")]


def srt_segments(paths, q, block, phrase):
    return image_segments(paths) + [("text", f"{q}\n" + prompts.fill(SRT_INSTR[block], phrase))]


def srt_target(block, phrase, vals, ans: bool) -> str:
    return "\n".join(fact_lines(block, phrase, vals)) + f"\nAnswer: {'Yes' if ans else 'No'}"


# --------------------------------------------------------------------------- LoRA
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(torch.nn.Module):
    """y = W x + (alpha / r) * B(A(dropout(x))); W frozen, A/B in fp32, B initialised to zero."""

    def __init__(self, base, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = torch.nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def forward(self, x):
        y = self.base(x)
        z = self.drop(x).to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()
        return y + (self.scale * z).to(y.dtype)


def language_layers_root(model):
    """Module that holds the decoder layers of the language model (vision tower excluded)."""
    for name in ("model.language_model", "language_model", "model.text_model"):
        m = model
        try:
            for part in name.split("."):
                m = getattr(m, part)
            if hasattr(m, "layers"):
                return m
        except AttributeError:
            continue
    raise RuntimeError("language model layers not found")


def inject_lora(model, r=16, alpha=32, dropout=0.05, targets=LORA_TARGETS):
    root = language_layers_root(model)
    n = 0
    for layer in root.layers:
        for parent in layer.modules():
            for cname, child in list(parent.named_children()):
                if cname in targets and isinstance(child, torch.nn.Linear):
                    setattr(parent, cname, LoRALinear(child, r, alpha, dropout))
                    n += 1
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
            params += [m.lora_A, m.lora_B]
    return params, n


def lora_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def merge_lora(model):
    """Fold every LoRA update into its frozen weight and restore plain nn.Linear modules (for fast evaluation)."""
    for parent in list(model.modules()):
        for cname, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    delta = (child.lora_B @ child.lora_A) * child.scale
                    child.base.weight += delta.to(child.base.weight.dtype)
                setattr(parent, cname, child.base)


# --------------------------------------------------------------------------- positions for representation alignment
def _find_last(ids: List[int], sub: List[int]) -> int:
    m = len(sub)
    for s in range(len(ids) - m, -1, -1):
        if ids[s:s + m] == sub:
            return s
    return -1


def align_positions(tok, ids_row: List[int], q: str) -> Tuple[int, int]:
    """(last token before the question, answer position) in one left-padded row.

    The question text is identical in the student and teacher prompts; only what precedes it differs
    (image tokens vs. image tokens plus the stated facts)."""
    q_ids = tok.encode(q, add_special_tokens=False)[:-1]   # the final '?' merges with the newline that follows
    s = _find_last(ids_row, q_ids)
    if s < 0:                                   # first question token merged with the preceding text
        s = _find_last(ids_row, q_ids[1:]) - 1
    if s < 1:
        raise ValueError("question tokens not found in prompt")
    return s - 1, len(ids_row) - 1


def rep_layers(n_layers: int) -> Tuple[int, int]:
    return n_layers // 2, (3 * n_layers) // 4
