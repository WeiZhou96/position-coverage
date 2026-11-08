"""P4: confirmation of the position-coverage results on the sealed E3-structured set (the preregistered analysis plan).

Data: data/sealed_e3/eval_requests.jsonl (12 groups, never used before P4); outputs eval_sealed_e3_logit.jsonl.
Metric (fixed before any P4 output): for each run and n in {4, 5, 6}, AUROC of P(Yes) between ALL-F (all present) and
ALL-B1 with the missing image at an interior slot (INT) or at slot 1 / n (EDGE); INT_46, EDGE_46 = means over n;
GAP = EDGE_46 - INT_46. Arm = mean of seeds. 95% CIs: 1,000 cluster-bootstrap resamples of the 12 groups (within
source), seed 20261107, the same draws for every run; contrasts use the same draws on both sides.
Output: outputs/p4/{p4.json, p4.md}. Usage: python scripts/p4_analyze.py
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
OUT = os.path.join(ROOT, "outputs", "p4")
B, SEED = 1000, 20261107
ARMS = {"qwen": {"base": ["base"], "A": ["A_s1", "A_s2"], "A23": ["A23_s1", "A23_s2"],
                 "A3edge": ["A3edge_s1", "A3edge_s2"], "A24": ["A24_s1", "A24_s2"]},
        "internvl": {"base": ["ivbase"], "A": ["ivA_s1", "ivA_s2"], "A23": ["ivA23_s1", "ivA23_s2"]},
        "gemma": {"base": ["gmbase"], "A": ["gmA_s1", "gmA_s2"], "A23": ["gmA23_s1", "gmA23_s2"]}}


def auroc_w(pos, neg, wp, wn):
    """Weighted AUROC (weights = bootstrap multiplicities of the items' groups)."""
    if wp.sum() == 0 or wn.sum() == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return float((wp[:, None] * wn[None, :] * gt).sum() / (wp.sum() * wn.sum()))


def main():
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "sealed_e3", "eval_requests.jsonl")):
        if r["set"] == "e3_grid" and r["mode"] == "logit" and r["predicate"] == "ALL" and r["pattern"] in ("F", "B1"):
            r.pop("segments", None)
            req[r["rid"]] = r
    groups = sorted({r["group"] for r in req.values()})
    gi = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(SEED)
    W = np.ones((B + 1, len(groups)))
    for src in ("clevr", "coco"):
        idx = np.array([gi[g] for g in groups if f"_{src}" in g])
        for b in range(1, B + 1):
            W[b, idx] = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))

    def run_vec(run):
        path = os.path.join(RUNS, run, "eval_sealed_e3_logit.jsonl")
        if not os.path.exists(path):
            return None
        items = defaultdict(lambda: ([], []))
        n_seen = 0
        for o in iter_jsonl(path):
            r = req.get(o["rid"])
            if r is None:
                continue
            n_seen += 1
            n = r["n"]
            if r["pattern"] == "F":
                key = (n, "F")
            else:
                slot = r["vals"].index(0) + 1
                key = (n, "int" if 1 < slot < n else "edge")
            items[key][0].append(o["probs"][0])
            items[key][1].append(gi[r["group"]])
        if n_seen < len(req):
            return None
        out = {}
        for kind in ("int", "edge"):
            per_n = []
            for n in (4, 5, 6):
                pos, gp = np.array(items[(n, "F")][0]), np.array(items[(n, "F")][1])
                neg, gn = np.array(items[(n, kind)][0]), np.array(items[(n, kind)][1])
                per_n.append([auroc_w(pos, neg, W[b, gp], W[b, gn]) for b in range(B + 1)])
            out[kind.upper() + "_46"] = np.mean(per_n, axis=0)
        out["GAP"] = out["EDGE_46"] - out["INT_46"]
        return out

    def ci(v):
        lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
        return [round(float(v[0]), 4), round(float(lo), 4), round(float(hi), 4)]

    res, vecs = {}, {}
    for fam, arms in ARMS.items():
        res[fam] = {}
        for arm, runs in arms.items():
            vs = {r: run_vec(r) for r in runs}
            if any(v is None for v in vs.values()):
                print("incomplete", fam, arm)
                continue
            m = {k: np.mean([v[k] for v in vs.values()], axis=0) for k in ("INT_46", "EDGE_46", "GAP")}
            vecs[(fam, arm)] = m
            res[fam][arm] = dict(mean={k: ci(x) for k, x in m.items()},
                                 seeds={r: {k: round(float(v[k][0]), 4) for k in ("INT_46", "EDGE_46", "GAP")}
                                        for r, v in vs.items()})
        print(fam, "done", flush=True)
    contrasts = {}
    for fam in ARMS:
        if (fam, "A23") in vecs and (fam, "A") in vecs:
            contrasts[f"{fam}: A23 - A (INT_46)"] = ci(vecs[(fam, "A23")]["INT_46"] - vecs[(fam, "A")]["INT_46"])
    if ("qwen", "A3edge") in vecs:
        contrasts["qwen: A3edge - A (INT_46)"] = ci(vecs[("qwen", "A3edge")]["INT_46"] - vecs[("qwen", "A")]["INT_46"])
        contrasts["qwen: A23 - A3edge (INT_46)"] = ci(vecs[("qwen", "A23")]["INT_46"]
                                                      - vecs[("qwen", "A3edge")]["INT_46"])
    if ("qwen", "A24") in vecs:
        contrasts["qwen: A24 - A23 (INT_46)"] = ci(vecs[("qwen", "A24")]["INT_46"] - vecs[("qwen", "A23")]["INT_46"])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "p4.json"), "w") as f:
        json.dump(dict(arms=res, contrasts=contrasts), f, indent=1)
    L = ["# P4 sealed-set confirmation (direct answering)", "",
         "| family | arm | INT_46 [CI] | EDGE_46 | GAP [CI] | seeds INT_46 | seeds GAP |", "|---|---|---|---|---|---|---|"]
    for fam, arms in res.items():
        for arm, x in arms.items():
            m = x["mean"]
            L.append(f"| {fam} | {arm} | {m['INT_46'][0]:.3f} [{m['INT_46'][1]:.3f}, {m['INT_46'][2]:.3f}] | "
                     f"{m['EDGE_46'][0]:.3f} | {m['GAP'][0]:+.3f} [{m['GAP'][1]:+.3f}, {m['GAP'][2]:+.3f}] | "
                     + ", ".join("%.3f" % s["INT_46"] for s in x["seeds"].values()) + " | "
                     + ", ".join("%+.3f" % s["GAP"] for s in x["seeds"].values()) + " |")
    L += ["", "| contrast | estimate [95% CI] |", "|---|---|"]
    L += [f"| {k} | {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] |" for k, v in contrasts.items()]
    with open(os.path.join(OUT, "p4.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
