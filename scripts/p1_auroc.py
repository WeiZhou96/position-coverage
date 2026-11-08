"""P1 post hoc (not pre-registered): threshold-free discrimination for ALL. AUROC of P(Yes) between F (all present,
truth Yes) and B1 (one image missing, truth No) with the missing image at an interior slot / at slot 1 or n, per n.
Unlike balanced accuracy at 0.5, it is unaffected by a global Yes/No bias. Arm = mean of the per-seed AUROCs.
Output: outputs/p1/posthoc_auroc.md"""
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib.io import iter_jsonl  # noqa: E402

ARMS = ["base", "A", "A23", "A3edge", "A24", "A23q"]


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        if r["set"] == "e3_grid" and r["mode"] == "logit" and r["predicate"] == "ALL" and r["pattern"] in ("B1", "F"):
            r.pop("segments", None)
            req[r["rid"]] = r
    lines = ["# P1 post hoc: AUROC of P(Yes), ALL, F vs B1 (missing image interior / edge); arm = mean of seeds", "",
             "| arm | n=3 interior | n=4 interior | n=5 interior | n=6 interior | mean 4-6 interior | mean 4-6 edge |",
             "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        runs = ["base"] if arm == "base" else [f"{arm}_s1", f"{arm}_s2"]
        vals = defaultdict(list)
        for run in runs:
            sc = defaultdict(list)
            for o in iter_jsonl(os.path.join(ROOT, "outputs", "s1", "runs", run, "eval_e3_logit.jsonl")):
                r = req.get(o["rid"])
                if r is None:
                    continue
                n = r["n"]
                if r["pattern"] == "F":
                    sc[(n, "F")].append(o["probs"][0])
                else:
                    slot = r["vals"].index(0) + 1
                    sc[(n, "int" if 1 < slot < n else "edge")].append(o["probs"][0])
            for n in range(3, 7):
                vals[(n, "int")].append(auroc(sc[(n, "F")], sc[(n, "int")]))
                vals[(n, "edge")].append(auroc(sc[(n, "F")], sc[(n, "edge")]))
        m = {k: float(np.mean(v)) for k, v in vals.items()}
        lines.append(f"| {arm} | " + " | ".join("%.3f" % m[(n, "int")] for n in range(3, 7))
                     + " | %.3f | %.3f |" % (np.mean([m[(n, "int")] for n in (4, 5, 6)]),
                                             np.mean([m[(n, "edge")] for n in (4, 5, 6)])))
    with open(os.path.join(ROOT, "outputs", "p1", "posthoc_auroc.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
