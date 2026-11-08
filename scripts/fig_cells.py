"""Figure data for the paper (post hoc, descriptive; evaluation grid, direct answers). Writes outputs/paper/:

cells.json      decomposition cells for the Qwen checkpoints with in-context readings (base, A_s1, A23_s1, D_s1,
                Dself_s1, Dself23p_s1): for every n, predicate, one-odd pattern (A1/B1) and slot of the odd image,
                the mean over image groups and rotations of E_iso, E_ctx and direct accuracy (as dx_e3_analyze.py).
phase.json      per family / design / seed / n: pooled interior AUROC and edge AUROC (ALL, F vs B1) and the rate at
                which all-present items are accepted (P(Yes) > 0.5).
example.json    one real n = 6 ALL item set: the B1 arrangement in all six rotations. Selection rule (fixed before
                looking at the panel): among the 12 COCO groups, the group whose 2-img interior detection rate
                (B1, slots 2..5, mean of both seeds) is closest to the median over all 24 groups. Contents: image
                paths in slot order per rotation, which image lacks the target, P(Yes) for the ALL question per
                checkpoint and rotation, P(Yes) for the all-present (F) arrangement, in-context readings per slot
                (checkpoints with in-context readings), isolated readings.
Usage: python scripts/fig_cells.py
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
OUT = os.path.join(ROOT, "outputs", "paper")
CTX_RUNS = ["base", "A_s1", "A23_s1", "D_s1", "Dself_s1", "Dself23p_s1"]


def load(run, name):
    p = os.path.join(RUNS, run, name)
    return {o["rid"]: o for o in iter_jsonl(p)} if os.path.exists(p) else {}


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    grid, single = {}, {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        if r["mode"] != "logit":
            continue
        if r["set"] == "e3_grid":
            grid[r["rid"]] = r
        elif r["set"] == "e3_single":
            single[r["rid"]] = r
    dx = [r for r in iter_jsonl(os.path.join(ROOT, "data", "e3dx", "eval_requests.jsonl")) if r["set"] == "dx_ctx"]
    os.makedirs(OUT, exist_ok=True)

    # ---------------- cells
    cells = {}
    for run in CTX_RUNS:
        e3o, dxo = load(run, "eval_e3_logit.jsonl"), load(run, "eval_e3dx_logit.jsonl")
        iso = {}
        for rid, r in single.items():
            if rid in e3o:
                p1 = sum(p for s, p in zip(r["sems"], e3o[rid]["probs"]) if int(s) == 1)
                iso[(r["source"], r["descriptor"], r["image_id"])] = p1
        ctx = {}
        for r in dx:
            if r["rid"] in dxo:
                p = sum(p for s, p in zip(r["sems"], dxo[r["rid"]]["probs"]) if int(s) == 1)
                ctx[(r["group"], r["n"], r["pattern"], r["rot"], r["slot"])] = p
        acc = defaultdict(lambda: defaultdict(list))
        for rid, r in grid.items():
            if r["pattern"] not in ("A1", "B1") or rid not in e3o:
                continue
            g, n, pat, rot = r["group"], r["n"], r["pattern"], r["rot"]
            keys = [(g, n, pat, rot, k) for k in range(1, n + 1)]
            if not all(k in ctx for k in keys):
                continue
            qs_ctx = [{1: ctx[k], 0: 1 - ctx[k]} for k in keys]
            qs_iso = [{1: iso[(r["source"], r["descriptor"], i)], 0: 1 - iso[(r["source"], r["descriptor"], i)]}
                      for i in r["image_ids"]]
            vals = r["vals"]
            odd_val = 0 if pat == "B1" else 1
            slot = vals.index(odd_val) + 1
            ans = r["answer"]
            c = acc[(n, r["predicate"], pat, slot)]
            c["E_iso"].append(e3.ideal(r["predicate"], ans, qs_iso))
            c["E_ctx"].append(e3.ideal(r["predicate"], ans, qs_ctx))
            c["direct"].append(float((e3o[rid]["probs"][0] > 0.5) == ans))
        cells[run] = [dict(n=k[0], op=k[1], pattern=k[2], slot=k[3], cnt=len(v["direct"]),
                           **{m: float(np.mean(x)) for m, x in v.items()}) for k, v in sorted(acc.items())]
        print("cells", run, len(cells[run]), flush=True)
    json.dump(cells, open(os.path.join(OUT, "cells.json"), "w"), indent=1)

    # ---------------- phase plane
    fams = {"qwen": "", "internvl": "iv", "gemma": "gm"}
    designs = {"base": ["base"], "A": ["A_s1", "A_s2"], "A23": ["A23_s1", "A23_s2"]}
    extra = {"A3edge": ["A3edge_s1", "A3edge_s2"], "A24": ["A24_s1", "A24_s2"], "A23q": ["A23q_s1", "A23q_s2"]}
    allb = {rid: r for rid, r in grid.items() if r["predicate"] == "ALL" and r["pattern"] in ("F", "B1")}
    phase = []
    for fam, p in fams.items():
        ds = dict(designs, **(extra if fam == "qwen" else {}))
        for d, runs in ds.items():
            for run in runs:
                name = (p + run) if not (run == "base" and p) else p + "base"
                o = load(name, "eval_e3_logit.jsonl")
                sc = defaultdict(list)
                for rid, r in allb.items():
                    if rid not in o:
                        continue
                    n = r["n"]
                    if r["pattern"] == "F":
                        sc[(n, "F")].append(o[rid]["probs"][0])
                    else:
                        s = r["vals"].index(0) + 1
                        sc[(n, "int" if 1 < s < n else "edge")].append(o[rid]["probs"][0])
                for n in (3, 4, 5, 6):
                    phase.append(dict(family=fam, design=d, run=name, n=n,
                                      auc_int=auroc(sc[(n, "F")], sc[(n, "int")]),
                                      auc_edge=auroc(sc[(n, "F")], sc[(n, "edge")]),
                                      acc_F=float(np.mean(np.asarray(sc[(n, "F")]) > 0.5)),
                                      det_int=float(np.mean(np.asarray(sc[(n, "int")]) < 0.5))))
                print("phase", fam, d, name, flush=True)
    json.dump(phase, open(os.path.join(OUT, "phase.json"), "w"), indent=1)

    # ---------------- example
    oA = [load(r, "eval_e3_logit.jsonl") for r in ("A_s1", "A_s2")]
    det = defaultdict(list)
    for rid, r in allb.items():
        if r["n"] == 6 and r["pattern"] == "B1":
            s = r["vals"].index(0) + 1
            if 1 < s < 6:
                for o in oA:
                    det[r["group"]].append(float(o[rid]["probs"][0] < 0.5))
    rate = {g: float(np.mean(v)) for g, v in det.items()}
    med = float(np.median(list(rate.values())))
    coco = {g: v for g, v in rate.items() if g.startswith("coco")}
    pick = min(coco, key=lambda g: (abs(coco[g] - med), g))
    print("example group", pick, "rate", coco[pick], "median", med, flush=True)
    ck = ["base", "A_s1", "A_s2", "A23_s1", "A23_s2"]
    outs = {c: load(c, "eval_e3_logit.jsonl") for c in ck}
    dxs = {c: load(c, "eval_e3dx_logit.jsonl") for c in ("base", "A_s1", "A23_s1")}
    rows = []
    items = sorted([r for r in allb.values() if r["group"] == pick and r["n"] == 6], key=lambda r: (r["pattern"], r["rot"]))
    for r in items:
        paths = [v for k, v in r["segments"] if k == "image"]
        row = dict(pattern=r["pattern"], rot=r["rot"], vals=r["vals"], image_ids=r["image_ids"], paths=paths,
                   question=[v for k, v in r["segments"] if k == "text"][-1], descriptor=r["descriptor"],
                   p_yes={c: outs[c][r["rid"]]["probs"][0] for c in ck if r["rid"] in outs[c]})
        rd = {}
        for c, o in dxs.items():
            vals = []
            for q in dx:
                if q["group"] == pick and q["n"] == 6 and q["pattern"] == r["pattern"] and q["rot"] == r["rot"] \
                        and q["rid"] in o:
                    vals.append((q["slot"], sum(p for s, p in zip(q["sems"], o[q["rid"]]["probs"]) if int(s) == 1)))
            rd[c] = [v for _, v in sorted(vals)]
        row["ctx_read"] = rd
        rows.append(row)
    json.dump(dict(group=pick, rate=coco[pick], median=med, rates=rate, rows=rows),
              open(os.path.join(OUT, "example.json"), "w"), indent=1)
    print("example rows", len(rows), flush=True)


if __name__ == "__main__":
    main()
