"""P2 (cross-family replication of the position effect) analysis on the E3 grid, direct answering
(the preregistered analysis plan). Families: InternVL3.5-8B (prefix iv), Gemma-3-12B (prefix gm); Qwen3-VL-8B (no
prefix) is shown as the reference from P1.

Per run (base: <prefix>base; trained: <prefix>A_s1/2, <prefix>A23_s1/2):
  AUROC_int(n)  AUROC of P(Yes) between ALL-F (all present) and ALL-B1 with the missing image at an interior slot
  AUROC_edge(n) the same with the missing image at slot 1 or n
  INT_46, EDGE_46 means over n = 4, 5, 6;  GAP = EDGE_46 - INT_46
  IBA_46_ALL    balanced accuracy at 0.5 as in P1;  F(n) accuracy on all-present;  B1 detection by slot (n = 6)
  sufficiency   k-balanced accuracy on trained operators (n = 2: ALL, ANY; n = 3: ALL, ANY, ONE, NONE) against
                min(0.95, E - 0.02), E from the run's own isolated readings
Arm value = mean of seeds; seeds reported. Output: outputs/p2/{p2.json, p2.md}. Usage: python scripts/p2_analyze.py
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
OUT = os.path.join(ROOT, "outputs", "p2")
FAMILIES = {"qwen (P1 reference)": "", "internvl": "iv", "gemma": "gm"}
OPS4 = ("ALL", "ANY", "ONE", "NONE")


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def score_run(run, req):
    path = os.path.join(RUNS, run, "eval_e3_logit.jsonl")
    if not os.path.exists(path):
        return None
    out = {o["rid"]: o for o in iter_jsonl(path)}
    if sum(1 for rid in out if req[rid]["set"] == "e3_grid") < sum(1 for r in req.values() if r["set"] == "e3_grid"):
        return None
    iso = {}
    for rid, o in out.items():
        r = req[rid]
        if r["set"] == "e3_single":
            p1 = sum(p for s, p in zip(r["sems"], o["probs"]) if int(s) == 1)
            iso[(r["source"], r["descriptor"], r["image_id"])] = {1: p1, 0: 1 - p1}
    sc = defaultdict(list)
    acc = defaultdict(list)
    kcell = defaultdict(lambda: [[], []])
    for rid, o in out.items():
        r = req[rid]
        if r["set"] != "e3_grid":
            continue
        n, op, py = r["n"], r["predicate"], o["probs"][0]
        c = float((py > 0.5) == r["answer"])
        if op == "ALL" and r["pattern"] in ("F", "B1"):
            if r["pattern"] == "F":
                sc[(n, "F")].append(py)
                acc[(n, "F")].append(c)
            else:
                slot = r["vals"].index(0) + 1
                sc[(n, "int" if 1 < slot < n else "edge")].append(py)
                acc[(n, "int" if 1 < slot < n else "edge")].append(c)
                acc[(n, "slot", slot)].append(c)
        if op in OPS4 and n in (2, 3):
            qs = [iso[(r["source"], r["descriptor"], i)] for i in r["image_ids"]]
            cell = kcell[(n, op, sum(r["vals"]))]
            cell[0].append(c)
            cell[1].append(e3.ideal(op, r["answer"], qs))
    v = {}
    for n in range(3, 7):
        v[f"AUROC_int|{n}"] = auroc(sc[(n, "F")], sc[(n, "int")])
        v[f"AUROC_edge|{n}"] = auroc(sc[(n, "F")], sc[(n, "edge")])
        v[f"F|{n}"] = float(np.mean(acc[(n, "F")]))
        v[f"IBA_ALL|{n}"] = 0.5 * float(np.mean(acc[(n, "int")])) + 0.5 * v[f"F|{n}"]
    v["INT_46"] = float(np.mean([v[f"AUROC_int|{n}"] for n in (4, 5, 6)]))
    v["EDGE_46"] = float(np.mean([v[f"AUROC_edge|{n}"] for n in (4, 5, 6)]))
    v["GAP"] = v["EDGE_46"] - v["INT_46"]
    v["IBA_46_ALL"] = float(np.mean([v[f"IBA_ALL|{n}"] for n in (4, 5, 6)]))
    v["B1_slots_6"] = [round(float(np.mean(acc[(6, "slot", s)])), 3) for s in range(1, 7)]
    suff = {}
    for n in (2, 3):
        ops = ("ALL", "ANY") if n == 2 else OPS4
        a = np.mean([np.mean([np.mean(kcell[(n, op, k)][0]) for k in range(n + 1) if kcell[(n, op, k)][0]])
                     for op in ops])
        e = np.mean([np.mean([np.mean(kcell[(n, op, k)][1]) for k in range(n + 1) if kcell[(n, op, k)][1]])
                     for op in ops])
        suff[str(n)] = dict(acc=round(float(a), 4), E=round(float(e), 4), ok=bool(a >= min(0.95, e - 0.02)))
    v["sufficiency"] = suff
    return v


def main():
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        if r["mode"] == "logit":
            r.pop("segments", None)
            req[r["rid"]] = r
    res = {}
    for fam, pfx in FAMILIES.items():
        res[fam] = {}
        for arm in ("base", "A", "A23"):
            runs = [f"{pfx}base" if pfx else "base"] if arm == "base" else [f"{pfx}{arm}_s1", f"{pfx}{arm}_s2"]
            vs = {run: score_run(run, req) for run in runs}
            vs = {k: x for k, x in vs.items() if x is not None}
            if not vs:
                continue
            keys = [k for k in next(iter(vs.values())) if isinstance(next(iter(vs.values()))[k], float)]
            res[fam][arm] = dict(mean={k: float(np.mean([x[k] for x in vs.values()])) for k in keys}, seeds=vs)
        print(fam, "done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "p2.json"), "w") as f:
        json.dump(res, f, indent=1)
    L = ["# P2 cross-family position effect (direct answering, E3 grid)", "",
         "| family | arm | INT_46 | EDGE_46 | GAP | IBA_46_ALL | F n=6 | seeds INT_46 | B1 detection by slot, n = 6 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for fam, arms in res.items():
        for arm, x in arms.items():
            m = x["mean"]
            seeds = ", ".join("%.3f" % s["INT_46"] for s in x["seeds"].values())
            slots = " / ".join(" ".join("%.2f" % q for q in s["B1_slots_6"]) for s in x["seeds"].values())
            L.append(f"| {fam} | {arm} | {m['INT_46']:.3f} | {m['EDGE_46']:.3f} | {m['GAP']:+.3f} | "
                     f"{m['IBA_46_ALL']:.3f} | {m['F|6']:.2f} | {seeds} | {slots} |")
    L += ["", "## Training sufficiency", "", "```",
          json.dumps({f"{fam}|{arm}": {run: s["sufficiency"] for run, s in x["seeds"].items()}
                      for fam, arms in res.items() for arm, x in arms.items() if arm != "base"}, indent=1), "```"]
    with open(os.path.join(OUT, "p2.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
