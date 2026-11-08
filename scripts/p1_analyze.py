"""P1 (position-coverage test) analysis on the E3 grid, direct answering (the preregistered analysis plan).

Metric (fixed before any P1 output existed):
  interior balanced accuracy IBA(n, op): one-odd-image arrangements whose odd image sits at an interior slot
    (2..n-1) against the uniform arrangements of the same group and n, balanced 50/50:
      ALL:            0.5 * acc(B1, odd slot interior) + 0.5 * acc(F)
      ANY, ONE, NONE: 0.5 * acc(A1, odd slot interior) + 0.5 * acc(Z)
    (an always-Yes or always-No answer scores 0.5). EBA is the same with the odd image at slot 1 or n.
  IBA_46 = mean over op in {ALL, ANY, ONE, NONE} and n in {4, 5, 6}; IBA_46_ALL = ALL only; IBA_3 = n = 3 (slot 2).
  sufficiency: k-balanced accuracy on the trained operators of the run's training counts (n = 2: ALL, ANY as the
    nearest E3 forms of the trained AND, OR; n = 3 or 4: ALL, ANY, ONE, NONE) against min(0.95, E - 0.02), E from the
    run's own isolated readings.
  CIs: 1,000 cluster-bootstrap resamples of the 24 E3 groups (within source), seed 20261106; arm = mean of seeds.
Output: outputs/p1/{p1.json, p1.md}. Usage: python scripts/p1_analyze.py [runs ...]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib import e3  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402

RUNS = os.path.join(ROOT, "outputs", "s1", "runs")
OUT = os.path.join(ROOT, "outputs", "p1")
ARMS = ["base", "A", "A23", "A3edge", "A24", "A23q"]
TRAIN_N = {"A": (2,), "A23": (2, 3), "A3edge": (2, 3), "A24": (2, 4), "A23q": (2, 3)}
OPS4 = ("ALL", "ANY", "ONE", "NONE")
B = 1000
SEED = 20261106


def runs_of(arm):
    return ["base"] if arm == "base" else [f"{arm}_s1", f"{arm}_s2"]


def main():
    arms = sys.argv[1:] or ARMS
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        if r["mode"] == "logit":
            r.pop("segments", None)
            req[r["rid"]] = r
    groups = sorted({r["group"] for r in req.values() if r["set"] == "e3_grid"})
    gi = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(SEED)
    W = np.ones((B + 1, len(groups)))
    for src in ("clevr", "coco"):
        idx = np.array([gi[g] for g in groups if g.startswith(src)])
        for b in range(1, B + 1):
            W[b, idx] = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))

    def cell_key(r):
        """Which balanced cell a grid item belongs to, or None."""
        n, op, pat = r["n"], r["predicate"], r["pattern"]
        if op not in OPS4:
            return None
        odd_pat, uni = ("B1", "F") if op == "ALL" else ("A1", "Z")
        if pat == uni:
            return (n, op, "uni")
        if pat == odd_pat:
            slot = r["vals"].index(0 if op == "ALL" else 1) + 1
            return (n, op, "int" if 1 < slot < n else "edge")
        return None

    res = {}
    for arm in arms:
        per_run = []
        for run in runs_of(arm):
            path = os.path.join(RUNS, run, "eval_e3_logit.jsonl")
            if not os.path.exists(path):
                print("missing", run)
                per_run = []
                break
            out = {o["rid"]: o for o in iter_jsonl(path)}
            iso = {}
            S = defaultdict(lambda: np.zeros((len(groups), 2)))      # key -> per group (correct, count)
            K = defaultdict(lambda: np.zeros((len(groups), 3)))      # (n, op, k) -> per group (correct, E, count)
            for rid, o in out.items():
                r = req[rid]
                if r["set"] == "e3_single":
                    p1 = sum(p for s, p in zip(r["sems"], o["probs"]) if int(s) == 1)
                    iso[(r["source"], r["descriptor"], r["image_id"])] = {1: p1, 0: 1 - p1}
            for rid, o in out.items():
                r = req[rid]
                if r["set"] != "e3_grid":
                    continue
                c = float((o["probs"][0] > 0.5) == r["answer"])
                key = cell_key(r)
                if key:
                    S[key][gi[r["group"]]] += (c, 1)
                if r["predicate"] in OPS4:
                    qs = [iso[(r["source"], r["descriptor"], i)] for i in r["image_ids"]]
                    K[(r["n"], r["predicate"], sum(r["vals"]))][gi[r["group"]]] += (c, e3.ideal(r["predicate"],
                                                                                                r["answer"], qs), 1)

            def acc(key):
                m = W @ S[key]
                return m[:, 0] / m[:, 1]

            v = {}
            for n in range(3, 7):
                for op in OPS4:
                    v[f"IBA|{n}|{op}"] = 0.5 * acc((n, op, "int")) + 0.5 * acc((n, op, "uni"))
                    v[f"EBA|{n}|{op}"] = 0.5 * acc((n, op, "edge")) + 0.5 * acc((n, op, "uni"))
            v["IBA_46"] = np.mean([v[f"IBA|{n}|{op}"] for n in (4, 5, 6) for op in OPS4], axis=0)
            v["IBA_46_ALL"] = np.mean([v[f"IBA|{n}|ALL"] for n in (4, 5, 6)], axis=0)
            v["EBA_46"] = np.mean([v[f"EBA|{n}|{op}"] for n in (4, 5, 6) for op in OPS4], axis=0)
            v["IBA_3"] = np.mean([v[f"IBA|3|{op}"] for op in OPS4], axis=0)
            v["IBA_56"] = np.mean([v[f"IBA|{n}|{op}"] for n in (5, 6) for op in OPS4], axis=0)
            suff = {}
            for n in TRAIN_N.get(arm, ()):
                ops = ("ALL", "ANY") if n == 2 else OPS4
                accs, Es = [], []
                for op in ops:
                    ks = [k for (nn, o, k) in K if nn == n and o == op]
                    a = np.mean([K[(n, op, k)][:, 0].sum() / K[(n, op, k)][:, 2].sum() for k in ks])
                    e = np.mean([K[(n, op, k)][:, 1].sum() / K[(n, op, k)][:, 2].sum() for k in ks])
                    accs.append(a)
                    Es.append(e)
                a, e = float(np.mean(accs)), float(np.mean(Es))
                suff[str(n)] = dict(acc=round(a, 4), E=round(e, 4), ok=a >= min(0.95, e - 0.02))
            per_run.append((run, v, suff))
        if not per_run:
            continue
        keys = per_run[0][1].keys()
        arm_v = {k: np.mean([pr[1][k] for pr in per_run], axis=0) for k in keys}
        res[arm] = dict(
            arm={k: [round(float(x[0]), 4)] + [round(float(q), 4) for q in np.nanpercentile(x[1:], [2.5, 97.5])]
                 for k, x in arm_v.items()},
            seeds={pr[0]: {k: round(float(pr[1][k][0]), 4) for k in keys} for pr in per_run},
            sufficiency={pr[0]: pr[2] for pr in per_run})
        print(arm, "done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "p1.json"), "w") as f:
        json.dump(res, f, indent=1)
    L = ["# P1 position-coverage test (direct answering, E3 grid)", "",
         "| arm | IBA_46 [CI] | IBA_46_ALL [CI] | EBA_46 | IBA_3 | IBA_56 | seeds IBA_46 | seeds IBA_46_ALL |",
         "|---|---|---|---|---|---|---|---|"]
    for arm, x in res.items():
        a = x["arm"]
        L.append(f"| {arm} | {a['IBA_46'][0]:.3f} [{a['IBA_46'][1]:.3f}, {a['IBA_46'][2]:.3f}] | "
                 f"{a['IBA_46_ALL'][0]:.3f} [{a['IBA_46_ALL'][1]:.3f}, {a['IBA_46_ALL'][2]:.3f}] | "
                 f"{a['EBA_46'][0]:.3f} | {a['IBA_3'][0]:.3f} | {a['IBA_56'][0]:.3f} | "
                 + ", ".join(f"{s['IBA_46']:.3f}" for s in x["seeds"].values()) + " | "
                 + ", ".join(f"{s['IBA_46_ALL']:.3f}" for s in x["seeds"].values()) + " |")
    L += ["", "## IBA by n and operator (arm mean)", "", "| arm | n | ALL | ANY | ONE | NONE |", "|---|---|---|---|---|---|"]
    for arm, x in res.items():
        for n in range(3, 7):
            L.append(f"| {arm} | {n} | " + " | ".join(f"{x['arm'][f'IBA|{n}|{op}'][0]:.3f}" for op in OPS4) + " |")
    L += ["", "## Training sufficiency", "", "```", json.dumps({a: x["sufficiency"] for a, x in res.items()}, indent=1),
          "```"]
    with open(os.path.join(OUT, "p1.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
