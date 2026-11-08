"""Analysis of the unseen-operator probe (dx_ops_build.py; post hoc screening, descriptive).

Per run and protocol (direct = eval_e3ops_logit, stated = eval_e3ops_gen; unparsed = wrong):
  count-only operators: accuracy balanced over k (number of images with the target), as M_n in E3;
  identity operators:   accuracy balanced over the four (v_i, v_j) cells of the two named slots;
  constant baseline (best of always-Yes / always-No under the same balancing); ideal reference E from the run's own
  isolated readings (e3_single), balanced the same way; AUROC of P(Yes) for direct.
Output: outputs/e3ops/{ops.json, ops.md}. Usage: python scripts/dx_ops_analyze.py [runs ...]
"""
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib.io import iter_jsonl  # noqa: E402

RUNS = os.path.join(ROOT, "outputs", "s1", "runs")
OUT = os.path.join(ROOT, "outputs", "e3ops")
DEFAULT = ["base", "A_s1", "A23_s1", "Aall_s1", "D_s1", "Dself_s1", "Dself23p_s1"]
COUNT = ("EXACTLY2", "ATMOST1", "MAJORITY")
IDENT = ("BOTH_IJ", "ONLY_IJ")
F = {"EXACTLY2": lambda v, p: sum(v) == 2, "ATMOST1": lambda v, p: sum(v) <= 1,
     "MAJORITY": lambda v, p: 2 * sum(v) > len(v),
     "BOTH_IJ": lambda v, p: v[p[0] - 1] == 1 and v[p[1] - 1] == 1,
     "ONLY_IJ": lambda v, p: v[p[0] - 1] == 1 and v[p[1] - 1] == 0}


def load(path):
    return {o["rid"]: o for o in iter_jsonl(path)} if os.path.exists(path) else {}


def ideal(op, pair, truth, qs):
    tot = 0.0
    for vals in itertools.product((0, 1), repeat=len(qs)):
        p = np.prod([q[v] for q, v in zip(qs, vals)])
        tot += p * (F[op](vals, pair) == truth)
    return tot


def auroc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main():
    runs = sys.argv[1:] or DEFAULT
    e3req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        if r["set"] == "e3_single":
            r.pop("segments", None)
            e3req[r["rid"]] = r
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3ops", "eval_requests.jsonl")):
        r.pop("segments", None)
        req[r["rid"]] = r
    res = {}
    for run in runs:
        d = os.path.join(RUNS, run)
        iso = {}
        for o in iter_jsonl(os.path.join(d, "eval_e3_logit.jsonl")):
            r = e3req.get(o["rid"])
            if r:
                p1 = sum(p for s, p in zip(r["sems"], o["probs"]) if int(s) == 1)
                iso[(r["source"], r["descriptor"], r["image_id"])] = {1: p1, 0: 1 - p1}
        for proto, mode in (("direct", "logit"), ("stated", "gen")):
            out = load(os.path.join(d, f"eval_e3ops_{mode}.jsonl"))
            if not out:
                continue
            cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # (n, op) -> cell -> metric -> list
            sc = defaultdict(lambda: ([], []))
            for rid, o in out.items():
                r = req[rid]
                v, pair = r["vals"], tuple(r["pair"]) if r["pair"] else None
                if proto == "direct":
                    py = o["probs"][0]
                    pred = py > 0.5
                    sc[(r["n"], r["predicate"])][0].append(py)
                    sc[(r["n"], r["predicate"])][1].append(r["answer"])
                else:
                    pred = o.get("pred")
                cell = sum(v) if r["predicate"] in COUNT else (v[pair[0] - 1], v[pair[1] - 1])
                qs = [iso[(r["source"], r["descriptor"], i)] for i in r["image_ids"]]
                c = cells[(r["n"], r["predicate"])][cell]
                c["acc"].append(float(pred is not None and pred == r["answer"]))
                c["E"].append(ideal(r["predicate"], pair, r["answer"], qs))
                c["yes"].append(float(r["answer"]))
                if proto == "stated":
                    c["explicit"].append(float(bool(o.get("explicit"))))
            tab = {}
            for (n, op), byc in sorted(cells.items()):
                acc = np.mean([np.mean(c["acc"]) for c in byc.values()])
                E = np.mean([np.mean(c["E"]) for c in byc.values()])
                fy = np.mean([np.mean(c["yes"]) for c in byc.values()])
                tab[f"{n}|{op}"] = dict(acc=float(acc), E=float(E), base=float(max(fy, 1 - fy)),
                                        auroc=auroc(*sc[(n, op)]) if proto == "direct" else None,
                                        cells={str(k): float(np.mean(c["acc"])) for k, c in byc.items()})
            for n in (3, 6):
                for cls, ops in (("count", COUNT), ("ident", IDENT)):
                    xs = [tab[f"{n}|{op}"] for op in ops if f"{n}|{op}" in tab]
                    tab[f"{n}|{cls}"] = {m: float(np.mean([x[m] for x in xs])) for m in ("acc", "E", "base")}
            res[f"{run}|{proto}"] = tab
        print(run, "done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ops.json"), "w") as f:
        json.dump(res, f, indent=1)
    L = ["# Unseen-operator probe (post hoc; E3 images, half of the groups; single seed per arm)", "",
         "Cells: count-only operators balanced over k, identity operators over the four (v_i, v_j) cells.", ""]
    for n in (3, 6):
        L += [f"## n = {n}: accuracy (gap to E in points)", "",
              "| run | protocol | " + " | ".join(COUNT + IDENT) + " | count mean | ident mean |",
              "|" + "---|" * (len(COUNT) + len(IDENT) + 4)]
        for key, tab in res.items():
            run, proto = key.split("|")
            row = []
            for op in COUNT + IDENT + ("count", "ident"):
                x = tab.get(f"{n}|{op}")
                row.append("—" if x is None else f"{x['acc']:.3f} ({100 * (x['E'] - x['acc']):+.1f})")
            L.append(f"| {run} | {proto} | " + " | ".join(row) + " |")
        L.append("")
        base = next(iter(res.values()))
        L.append("Constant baselines: " + ", ".join(f"{op} {base[f'{n}|{op}']['base']:.3f}" for op in COUNT + IDENT
                                                   if f"{n}|{op}" in base))
        L.append("")
    L += ["## Direct AUROC of P(Yes)", "", "| run | " + " | ".join(f"{op} n={n}" for n in (3, 6) for op in COUNT + IDENT)
          + " |", "|" + "---|" * (2 * len(COUNT + IDENT) + 1)]
    for key, tab in res.items():
        if key.endswith("|direct"):
            L.append(f"| {key.split('|')[0]} | " + " | ".join(
                f"{tab[f'{n}|{op}']['auroc']:.2f}" for n in (3, 6) for op in COUNT + IDENT) + " |")
    with open(os.path.join(OUT, "ops.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
