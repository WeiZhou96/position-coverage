"""P5 analysis (the preregistered analysis plan).

P5a (scale; the E3 subset data/p5e3 (scripts/p5_e3subset_build.py), direct answering; large zero-shot models write
eval_p5e3_logit.jsonl, the 8-12B checkpoints are scored on the same subset rows of their eval_e3_logit.jsonl):
  INT_46, EDGE_46, GAP   AUROC of P(Yes) between ALL-F and ALL-B1 with the missing image at an interior slot / at
                         slot 1 or n, mean over n = 4, 5, 6 (as P2, P4); GAP = EDGE_46 - INT_46
  AGG_6                  E_6 - M_6: ideal-integration reference from the model's own isolated readings minus the
                         k-balanced accuracy, mean over the five operators at n = 6 (as gap_6 in E3)
P5b (Visual Haystacks, needle rotated through slots 1..5; outputs eval_p5vhs_logit.jsonl):
  VHS_INT, VHS_EDGE      AUROC of P(Yes) between yes- and no-items with the needle at slots 2..4 / at slot 1 or 5
  VHS_GAP                VHS_EDGE - VHS_INT;  VHS_ACC  accuracy at 0.5, all slots;  acc by slot
Arm = mean of seeds. Output: outputs/p5/{p5.json, p5.md}. Usage: python scripts/p5_analyze.py
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
OUT = os.path.join(ROOT, "outputs", "p5")
LARGE = {"Qwen3-VL-32B": "L_qwen32", "Gemma-3-27B": "L_gemma27", "Mistral-Small-3.2-24B": "L_mistral24"}
SMALL = {"Qwen3-VL-8B": "", "InternVL3.5-8B": "iv", "Gemma-3-12B": "gm"}


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def p5a(run, req):
    path = os.path.join(RUNS, run, "eval_p5e3_logit.jsonl")
    if not os.path.exists(path):
        path = os.path.join(RUNS, run, "eval_e3_logit.jsonl")
    if not os.path.exists(path):
        return None
    out = {o["rid"]: o for o in iter_jsonl(path) if o["rid"] in req}
    if sum(1 for rid in out if rid in req) < len(req):
        return None
    iso = {}
    for rid, o in out.items():
        r = req[rid]
        if r["set"] == "e3_single":
            p1 = sum(p for s, p in zip(r["sems"], o["probs"]) if int(s) == 1)
            iso[(r["source"], r["descriptor"], r["image_id"])] = {1: p1, 0: 1 - p1}
    sc = defaultdict(list)
    cells = defaultdict(lambda: ([], []))
    for rid, o in out.items():
        r = req[rid]
        if r["set"] != "e3_grid":
            continue
        n, py = r["n"], o["probs"][0]
        if r["predicate"] == "ALL" and r["pattern"] in ("F", "B1"):
            if r["pattern"] == "F":
                sc[(n, "F")].append(py)
            else:
                slot = r["vals"].index(0) + 1
                sc[(n, "int" if 1 < slot < n else "edge")].append(py)
        if n == 6:
            qs = [iso[(r["source"], r["descriptor"], i)] for i in r["image_ids"]]
            c = cells[(r["predicate"], sum(r["vals"]))]
            c[0].append(float((py > 0.5) == r["answer"]))
            c[1].append(e3.ideal(r["predicate"], r["answer"], qs))
    v = dict(INT_46=float(np.mean([auroc(sc[(n, "F")], sc[(n, "int")]) for n in (4, 5, 6)])),
             EDGE_46=float(np.mean([auroc(sc[(n, "F")], sc[(n, "edge")]) for n in (4, 5, 6)])))
    v["GAP"] = v["EDGE_46"] - v["INT_46"]
    accs, Es = [], []
    for op in e3.OPS:
        ks = [k for (o, k) in cells if o == op]
        accs.append(np.mean([np.mean(cells[(op, k)][0]) for k in ks]))
        Es.append(np.mean([np.mean(cells[(op, k)][1]) for k in ks]))
    v["M_6"], v["E_6"] = float(np.mean(accs)), float(np.mean(Es))
    v["AGG_6"] = v["E_6"] - v["M_6"]
    return v


def p5b(run, vreq):
    path = os.path.join(RUNS, run, "eval_p5vhs_logit.jsonl")
    if not os.path.exists(path):
        return None
    out = {o["rid"]: o for o in iter_jsonl(path)}
    if len(out) < len(vreq):
        return None
    sc = defaultdict(lambda: ([], []))
    acc = defaultdict(list)
    for rid, o in out.items():
        r = vreq[rid]
        py = o["probs"][0]
        grp = "int" if 1 < r["slot"] < 5 else "edge"
        sc[grp][0 if r["answer"] else 1].append(py)
        acc[r["slot"]].append(float((py > 0.5) == r["answer"]))
    v = dict(VHS_INT=auroc(*sc["int"]), VHS_EDGE=auroc(*sc["edge"]))
    v["VHS_GAP"] = v["VHS_EDGE"] - v["VHS_INT"]
    v["VHS_ACC"] = float(np.mean([x for s in acc for x in acc[s]]))
    v["VHS_ACC_by_slot"] = [round(float(np.mean(acc[s])), 3) for s in range(1, 6)]
    return v


def main():
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "p5e3", "eval_requests.jsonl")):
        r.pop("segments", None)
        req[r["rid"]] = r
    vreq = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "p5vhs", "eval_requests.jsonl")):
        r.pop("segments", None)
        vreq[r["rid"]] = r
    rows = []
    for name, run in LARGE.items():
        rows.append((name, "base (zero-shot)", [run]))
    for name, pfx in SMALL.items():
        rows.append((name, "base", [f"{pfx}base" if pfx else "base"]))
        rows.append((name, "A", [f"{pfx}A_s1", f"{pfx}A_s2"]))
        rows.append((name, "A23", [f"{pfx}A23_s1", f"{pfx}A23_s2"]))
    res = []
    for name, arm, runs in rows:
        a = [x for x in (p5a(r, req) for r in runs) if x]
        b = [x for x in (p5b(r, vreq) for r in runs) if x]
        row = dict(model=name, arm=arm, runs=runs)
        if a:
            row.update({k: float(np.mean([x[k] for x in a])) for k in a[0]})
            row["seeds_GAP"] = [round(x["GAP"], 4) for x in a]
        if b:
            row.update({k: float(np.mean([x[k] for x in b])) for k in b[0] if k != "VHS_ACC_by_slot"})
            row["VHS_ACC_by_slot"] = list(np.round(np.mean([x["VHS_ACC_by_slot"] for x in b], axis=0), 3))
            row["seeds_VHS_INT"] = [round(x["VHS_INT"], 4) for x in b]
        res.append(row)
        print(name, arm, "done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "p5.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)

    def g(r, k, fmt="%.3f"):
        return fmt % r[k] if k in r else "—"

    L = ["# P5 scale and Visual Haystacks", "",
         "| model | arm | INT_46 | EDGE_46 | GAP | AGG_6 | VHS_INT | VHS_EDGE | VHS_GAP | VHS_ACC | VHS acc by slot |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['model']} | {r['arm']} | {g(r, 'INT_46')} | {g(r, 'EDGE_46')} | {g(r, 'GAP', '%+.3f')} | "
                 f"{g(r, 'AGG_6', '%+.3f')} | {g(r, 'VHS_INT')} | {g(r, 'VHS_EDGE')} | {g(r, 'VHS_GAP', '%+.3f')} | "
                 f"{g(r, 'VHS_ACC')} | {' '.join(str(x) for x in r.get('VHS_ACC_by_slot', [])) or '—'} |")
    with open(os.path.join(OUT, "p5.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
