"""Unseen-operator probe on the E3 images (direction re-diagnosis; post hoc screening, inference only).

Operators never trained by any checkpoint, asked about the E3 arrangements (n = 3 and 6, every pattern, every
rotation; the 12 groups with an even index per source, i.e. half of the E3 groups):
  count-only (depend only on k, the number of images with the target)
    EXACTLY2  "Do exactly two of the N images contain X?"
    ATMOST1   "Does at most one of the N images contain X?"
    MAJORITY  "Do more than half of the N images contain X?"
  identity-dependent (depend on which images contain the target; two named slots i, j)
    BOTH_IJ   "Do both Image i and Image j contain X?"
    ONLY_IJ   "Does Image i contain X while Image j does not?"
    slot pairs: n = 3 -> (1, 3), (2, 3); n = 6 -> (1, 6), (3, 4)
Both protocols: logit (direct) with the E3 student layout, gen (state-then-answer) with the E3 SRT layout.
Output: data/e3ops/eval_requests.jsonl (python scripts/s1_eval.py --run <run> --data e3ops --modes logit [gen]).
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib import e3, s1  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402

OUT = os.path.join(ROOT, "data", "e3ops")
NS = (3, 6)
PAIRS = {3: [(1, 3), (2, 3)], 6: [(1, 6), (3, 4)]}
COUNT_OPS = {
    "EXACTLY2": ("Do exactly two of the {N} images contain {X}?", lambda v: sum(v) == 2),
    "ATMOST1": ("Does at most one of the {N} images contain {X}?", lambda v: sum(v) <= 1),
    "MAJORITY": ("Do more than half of the {N} images contain {X}?", lambda v: 2 * sum(v) > len(v)),
}
ID_OPS = {
    "BOTH_IJ": ("Do both Image {i} and Image {j} contain {X}?", lambda v, i, j: v[i - 1] == 1 and v[j - 1] == 1),
    "ONLY_IJ": ("Does Image {i} contain {X} while Image {j} does not?", lambda v, i, j: v[i - 1] == 1 and v[j - 1] == 0),
}


def rid(*parts):
    return hashlib.md5(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:20]


def main():
    grid = [r for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl"))
            if r["set"] == "e3_grid" and r["mode"] == "logit" and r["predicate"] == "ALL" and r["n"] in NS
            and int(r["group"][-2:]) % 2 == 0]
    reqs = []
    for r in grid:
        paths = [v for k, v in r["segments"] if k == "image"]
        q0 = r["segments"][-1][1].split("\n")[0]
        X = re.search(r"contains? (.+)\?$", q0).group(1)
        phrase = {"singular": X, "plural": X}
        n, v = r["n"], r["vals"]
        meta = {k: r[k] for k in ("source", "block", "descriptor", "group", "n", "pattern", "rot", "vals",
                                  "image_ids")}
        qs = [(op, None, t.format(N=e3.NWORD[n], X=X), f(v)) for op, (t, f) in COUNT_OPS.items()]
        qs += [(op, (i, j), t.format(i=i, j=j, X=X), f(v, i, j)) for op, (t, f) in ID_OPS.items()
               for i, j in PAIRS[n]]
        for op, pair, q, ans in qs:
            for mode, segs in (("logit", s1.student_segments(paths, q)),
                               ("gen", s1.srt_segments(paths, q, "presence", phrase))):
                reqs.append(dict(rid=rid("e3ops", r["rid"], op, pair, mode), set="e3ops", mode=mode,
                                 segments=[list(s) for s in segs], cands=["Yes", "No"], sems=[1, 0],
                                 predicate=op, pair=pair, answer=bool(ans), **meta))
    assert len({x["rid"] for x in reqs}) == len(reqs)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "eval_requests.jsonl"), "w") as f:
        for x in reqs:
            f.write(json.dumps(x) + "\n")
    print(len(grid), "arrangements;", Counter((x["mode"], x["n"]) for x in reqs))
    print(Counter((x["predicate"], x["answer"]) for x in reqs if x["mode"] == "logit"))
    print([x for x in reqs if x["mode"] == "logit" and x["predicate"] == "ONLY_IJ"][0]["segments"][-1])


if __name__ == "__main__":
    main()
