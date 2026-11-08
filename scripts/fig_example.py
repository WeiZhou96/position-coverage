"""Qualitative-figure data (paper; evaluation grid, n = 6, ALL, direct answers). Writes outputs/paper/example2.json.

groups      for every image group (24): mean P(Yes) for the arrangement with every target present (F, mean over the
            six rotations) and for the arrangement missing the target at slot s = 1..6 (B1), for Base, 2-img and
            2+3-img (trained designs: mean of two seeds); plus the 2-img detection profile (P(Yes) < 0.5 by slot).
example     selection rule, fixed before drawing: the group whose 2-img detection profile over slots 1..6 has the
            smallest Euclidean distance to the mean profile over all 24 groups (ties: group id). For that group:
            image paths in slot order per rotation, which image lacks the target, P(Yes) per checkpoint and seed,
            in-context readings per slot (Base, 2-img s1, 2+3-img s1), and the question text.
Usage: python scripts/fig_example.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib.io import iter_jsonl  # noqa: E402

RUNS = os.path.join(ROOT, "outputs", "s1", "runs")
CK = {"base": ["base"], "A": ["A_s1", "A_s2"], "A23": ["A23_s1", "A23_s2"]}


def load(run, name):
    p = os.path.join(RUNS, run, name)
    return {o["rid"]: o for o in iter_jsonl(p)} if os.path.exists(p) else {}


def main():
    items = [r for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl"))
             if r["mode"] == "logit" and r.get("set") == "e3_grid" and r["predicate"] == "ALL" and r["n"] == 6
             and r["pattern"] in ("F", "B1")]
    outs = {run: load(run, "eval_e3_logit.jsonl") for runs in CK.values() for run in runs}
    py = defaultdict(list)          # (group, design, pattern, slot) -> P(Yes) values over seeds (and rotations for F)
    det = defaultdict(list)         # (group, slot) -> 2-img detections
    for r in items:
        slot = r["vals"].index(0) + 1 if r["pattern"] == "B1" else 0
        for d, runs in CK.items():
            for run in runs:
                p = outs[run][r["rid"]]["probs"][0]
                py[(r["group"], d, r["pattern"], slot)].append(p)
                if d == "A" and r["pattern"] == "B1":
                    det[(r["group"], slot)].append(float(p < 0.5))
    groups = sorted({r["group"] for r in items})
    prof = {g: np.array([np.mean(det[(g, s)]) for s in range(1, 7)]) for g in groups}
    mean_prof = np.mean(list(prof.values()), axis=0)
    pick = min(groups, key=lambda g: (float(np.linalg.norm(prof[g] - mean_prof)), g))
    print("mean profile", np.round(mean_prof, 3), "pick", pick, np.round(prof[pick], 3), flush=True)
    gsum = {}
    for g in groups:
        gsum[g] = {d: dict(F=float(np.mean(py[(g, d, "F", 0)])),
                           B1=[float(np.mean(py[(g, d, "B1", s)])) for s in range(1, 7)]) for d in CK}
        gsum[g]["det_A"] = prof[g].tolist()
    dxreq = [q for q in iter_jsonl(os.path.join(ROOT, "data", "e3dx", "eval_requests.jsonl"))
             if q["set"] == "dx_ctx" and q["group"] == pick and q["n"] == 6]
    dxo = {c: load(c, "eval_e3dx_logit.jsonl") for c in ("base", "A_s1", "A23_s1")}
    rows = []
    for r in sorted([r for r in items if r["group"] == pick], key=lambda r: (r["pattern"], r["rot"])):
        row = dict(pattern=r["pattern"], rot=r["rot"], vals=r["vals"], image_ids=r["image_ids"],
                   paths=[v for k, v in r["segments"] if k == "image"], descriptor=r["descriptor"],
                   question=[v for k, v in r["segments"] if k == "text"][-1],
                   p_yes={run: outs[run][r["rid"]]["probs"][0] for runs in CK.values() for run in runs})
        rd = {}
        for c, o in dxo.items():
            v = sorted((q["slot"], sum(p for s, p in zip(q["sems"], o[q["rid"]]["probs"]) if int(s) == 1))
                       for q in dxreq if q["pattern"] == r["pattern"] and q["rot"] == r["rot"] and q["rid"] in o)
            rd[c] = [x for _, x in v]
        row["ctx_read"] = rd
        rows.append(row)
    json.dump(dict(rule="2-img detection profile closest to the mean over 24 groups", mean_profile=mean_prof.tolist(),
                   pick=pick, groups=gsum, rows=rows),
              open(os.path.join(ROOT, "outputs", "paper", "example2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
