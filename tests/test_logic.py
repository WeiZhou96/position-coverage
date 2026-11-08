"""Type tables must match E0 design v2 section 2 cell by cell. Run: python -m pytest tests -q  (or python tests/test_logic.py)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e0lib import logic, prompts  # noqa: E402

R, U1, U2, S = "redundant", "unique1", "unique2", "synergistic"

PRESENCE = {  # (v1, v2): {pred: (answer, type)}   design table 2.1
    (1, 1): {"OR": (True, R), "AND": (True, S), "XOR": (False, S)},
    (0, 0): {"OR": (False, S), "AND": (False, R), "XOR": (False, S)},
    (1, 0): {"OR": (True, U1), "AND": (False, U2), "XOR": (True, S)},
    (0, 1): {"OR": (True, U2), "AND": (False, U1), "XOR": (True, S)},
}
COUNT = {  # design table 2.2
    (0, 0): {"GE": (True, U2), "GT": (False, U1), "EQ": (True, S)},
    (0, 1): {"GE": (False, S), "GT": (False, U1), "EQ": (False, S)},
    (0, 2): {"GE": (False, S), "GT": (False, R), "EQ": (False, S)},
    (1, 0): {"GE": (True, U2), "GT": (True, S), "EQ": (False, S)},
    (1, 1): {"GE": (True, S), "GT": (False, S), "EQ": (True, S)},
    (1, 2): {"GE": (False, S), "GT": (False, U2), "EQ": (False, S)},
    (2, 0): {"GE": (True, R), "GT": (True, S), "EQ": (False, S)},
    (2, 1): {"GE": (True, U1), "GT": (True, S), "EQ": (False, S)},
    (2, 2): {"GE": (True, U1), "GT": (False, U2), "EQ": (True, S)},
}


def test_tables():
    for table, dom in ((PRESENCE, logic.PRESENCE_DOMAIN), (COUNT, logic.BIN_DOMAIN)):
        for cell, preds in table.items():
            for pred, (ans, typ) in preds.items():
                got = logic.type_table(pred, dom)[cell]
                assert got == (ans, typ), (pred, cell, got, (ans, typ))


def test_same_pair_contrasts_share_answer():
    """Every same-pair contrast used by T1-T4 compares items with the same answer and different types."""
    from e0lib.analysis import _test_pair
    for test, dom in (("T1", (0, 1)), ("T2", (0, 1)), ("T3", (0, 1, 2)), ("T4", (0, 1, 2))):
        for a1 in dom:
            for a2 in dom:
                pr = _test_pair(test, a1, a2)
                if pr is None:
                    continue
                hi, lo = pr
                assert logic.answer(hi, a1, a2) == logic.answer(lo, a1, a2), (test, a1, a2)
                t_hi = logic.suff_type(*logic.sufficiency(hi, a1, a2, dom))
                t_lo = logic.suff_type(*logic.sufficiency(lo, a1, a2, dom))
                assert t_lo == S and t_hi != S, (test, a1, a2, t_hi, t_lo)


def test_null_noise_matches_closed_form():
    e = 0.03
    q = lambda v: {v: 1 - e, 1 - v: e}
    acc = lambda pred, a, b: logic.ideal_integrator_accuracy(pred, logic.answer(pred, a, b), q(a), q(b))
    assert abs((acc("OR", 1, 1) - acc("AND", 1, 1)) - (2 * e - 2 * e * e)) < 1e-12   # ~5.8 points at e=3 %
    assert abs((acc("OR", 1, 0) - acc("XOR", 1, 0)) - (e - e * e)) < 1e-12


def test_partial_targets():
    assert logic.partial_target("OR", 1, 0, (0, 1), "only1") == "yes"
    assert logic.partial_target("OR", 1, 0, (0, 1), "only2") == "undetermined"
    assert logic.partial_target("AND", 1, 0, (0, 1), "only2") == "no"
    assert logic.partial_target("XOR", 1, 0, (0, 1), "only1") == "undetermined"
    assert logic.partial_target("GE", 2, 1, (0, 1, 2), "none") == "undetermined"


def test_prompts_hidden_and_options():
    segs, c, s = prompts.relation_request("AND", {"singular": "a dog", "plural": "dogs"}, 0,
                                          ("/x/1.png", "/x/2.png"), (True, False), "ter", perm=1)
    text = "".join(v for k, v in segs if k == "text")
    assert "Image 2: " + prompts.HIDDEN in text and sum(k == "image" for k, _ in segs) == 1
    assert s == ["no", "undetermined", "yes"] and c == ["A", "B", "C"]
    assert "B. Cannot be determined" in text and text.endswith(prompts.INSTR_TER)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
