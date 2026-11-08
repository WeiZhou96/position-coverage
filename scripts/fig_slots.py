"""Figure data (paper): per-slot discrimination on the E3 grid, direct answering.

For each checkpoint and n = 3..6, the ALL items with all images containing the target (F) and with exactly one image
missing it (B1, missing image at slot s): AUROC of P(Yes) between F and B1@s, and the detection rate of B1@s
(P(Yes) < 0.5) and the acceptance rate of F (P(Yes) > 0.5). Same requests as the paper's E3 evaluation.
Output: outputs/paper/fig_slots.json. Usage: python scripts/fig_slots.py
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
FAMS = {"qwen": "", "internvl": "iv", "gemma": "gm"}
ARMS = {"base": ["base"], "A": ["A_s1", "A_s2"], "A23": ["A23_s1", "A23_s2"]}
QWEN_EXTRA = {"A3edge": ["A3edge_s1", "A3edge_s2"], "A24": ["A24_s1", "A24_s2"], "A23q": ["A23q_s1", "A23q_s2"]}


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        if r.get("set") == "e3_grid" and r["mode"] == "logit" and r["predicate"] == "ALL" and r["pattern"] in ("F", "B1"):
            req[r["rid"]] = (r["n"], r["pattern"], (r["vals"].index(0) + 1) if r["pattern"] == "B1" else 0)
    out = {}
    runs = []
    for fam, p in FAMS.items():
        arms = dict(ARMS)
        if fam == "qwen":
            arms.update(QWEN_EXTRA)
        for arm, rs in arms.items():
            names = [(f"{p}{r}" if not (r == "base" and p) else f"{p}base") for r in rs]
            runs.append((fam, arm, names))
    for fam, arm, names in runs:
        per_seed = []
        for run in names:
            path = os.path.join(RUNS, run, "eval_e3_logit.jsonl")
            if not os.path.exists(path):
                print("missing", path, flush=True)
                continue
            sc = defaultdict(list)
            for o in iter_jsonl(path):
                k = req.get(o["rid"])
                if k:
                    sc[k].append(o["probs"][0])
            res = {}
            for n in (3, 4, 5, 6):
                F = sc[(n, "F", 0)]
                res[f"accF|{n}"] = float(np.mean(np.asarray(F) > 0.5))
                for s in range(1, n + 1):
                    B = sc[(n, "B1", s)]
                    res[f"auc|{n}|{s}"] = auroc(F, B)
                    res[f"det|{n}|{s}"] = float(np.mean(np.asarray(B) < 0.5))
                    res[f"nB|{n}|{s}"] = len(B)
                res[f"nF|{n}"] = len(F)
            per_seed.append(res)
            print(fam, arm, run, "auc6", [round(res[f"auc|6|{s}"], 3) for s in range(1, 7)], flush=True)
        if per_seed:
            out[f"{fam}|{arm}"] = dict(seeds=per_seed, runs=names,
                                       mean={k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]})
    os.makedirs(os.path.join(ROOT, "outputs", "paper"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "paper", "fig_slots.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
