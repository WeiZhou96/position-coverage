"""P5b post hoc (not preregistered): item-cluster bootstrap CIs on Visual Haystacks.

For each family (Qwen3-VL-8B, InternVL3.5-8B, Gemma-3-12B):
  base VHS_GAP; A23 - A on VHS_INT, VHS_GAP, VHS_ACC; A - base on VHS_INT, VHS_ACC (arm = mean of seeds).
Bootstrap: 2,000 resamples of VHs items (each item keeps its 5 slot variants). 95% percentile intervals.
Output: outputs/p5/p5b_posthoc.{json,md}. Usage: python scripts/p5b_posthoc.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib.io import iter_jsonl  # noqa: E402

RUNS = os.path.join(ROOT, "outputs", "s1", "runs")
OUT = os.path.join(ROOT, "outputs", "p5")
FAM = {"Qwen3-VL-8B": "", "InternVL3.5-8B": "iv", "Gemma-3-12B": "gm"}
B = 2000


def auroc(s, y):
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(s)
    npos = y.sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def main():
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "p5vhs", "eval_requests.jsonl")):
        req[r["rid"]] = (r["item_id"], r["slot"], bool(r["answer"]))
    items = sorted({v[0] for v in req.values()})
    iidx = {it: i for i, it in enumerate(items)}
    rids = sorted(req)
    item_of = np.array([iidx[req[r][0]] for r in rids])
    slot = np.array([req[r][1] for r in rids])
    y = np.array([req[r][2] for r in rids])
    rows_of_item = [np.where(item_of == i)[0] for i in range(len(items))]

    def scores(run):
        out = {o["rid"]: o["probs"][0] for o in iter_jsonl(os.path.join(RUNS, run, "eval_p5vhs_logit.jsonl"))}
        return np.array([out[r] for r in rids])

    def metrics(s, idx):
        ss, yy, sl = s[idx], y[idx], slot[idx]
        inner = (sl > 1) & (sl < 5)
        vi, ve = auroc(ss[inner], yy[inner]), auroc(ss[~inner], yy[~inner])
        return np.array([vi, ve - vi, float(((ss > 0.5) == yy).mean())])   # VHS_INT, VHS_GAP, VHS_ACC

    rng = np.random.default_rng(0)
    boots = [np.concatenate([rows_of_item[i] for i in rng.integers(0, len(items), len(items))]) for _ in range(B)]
    full = np.arange(len(rids))
    res = {}
    for fam, p in FAM.items():
        arms = {"base": [f"{p}base" if p else "base"], "A": [f"{p}A_s1", f"{p}A_s2"],
                "A23": [f"{p}A23_s1", f"{p}A23_s2"]}
        S = {a: [scores(r) for r in rs] for a, rs in arms.items()}

        def arm_m(a, idx):
            return np.mean([metrics(s, idx) for s in S[a]], axis=0)

        point = {a: arm_m(a, full) for a in arms}
        bs = {a: np.array([arm_m(a, idx) for idx in boots]) for a in arms}

        def ci(x):
            return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

        d = {}
        d["base VHS_GAP"] = (float(point["base"][1]), ci(bs["base"][:, 1]))
        for k, j in (("VHS_INT", 0), ("VHS_GAP", 1), ("VHS_ACC", 2)):
            d[f"A23-A {k}"] = (float(point["A23"][j] - point["A"][j]), ci(bs["A23"][:, j] - bs["A"][:, j]))
        for k, j in (("VHS_INT", 0), ("VHS_ACC", 2)):
            d[f"A-base {k}"] = (float(point["A"][j] - point["base"][j]), ci(bs["A"][:, j] - bs["base"][:, j]))
        res[fam] = d
        print(fam, {k: (round(v[0], 4), [round(x, 4) for x in v[1]]) for k, v in d.items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "p5b_posthoc.json"), "w"), indent=1)
    keys = list(next(iter(res.values())).keys())
    L = ["# P5b post hoc: item-cluster bootstrap (2,000 resamples, 95% CI)", "",
         "| quantity | " + " | ".join(FAM) + " |", "|---" * (len(FAM) + 1) + "|"]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{res[f][k][0]:+.3f} [{res[f][k][1][0]:+.3f}, {res[f][k][1][1]:+.3f}]"
                                          for f in FAM) + " |")
    open(os.path.join(OUT, "p5b_posthoc.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
