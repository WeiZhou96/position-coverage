"""Analysis of the teacher-forced facts probe (dx_force_eval.py; post hoc, descriptive).

Per run, condition, n and operator (k-balanced over the number of target images, as M_n in E3):
  acc       answer = truth
  follow    answer = the answer implied by the facts in the assistant prefix (img/hidden: = truth; conflict: the
            flipped facts; selfimg/selfhidden: the run's own parsed facts, items with a complete parse only)
Also, for img: P(answer wrong | direct E3 answer wrong) next to the overall error rate.
Output: outputs/e3dx/force.{json,md}. Usage: python scripts/dx_force_analyze.py [runs ...]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from e0lib import e3  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402
from e3_posthoc import parse_facts  # noqa: E402

RUNS = os.path.join(ROOT, "outputs", "s1", "runs")
DEFAULT = ["base", "A23_s1", "D_s1", "Dself_s1", "Dself23p_s1"]
CONDS = ("img", "hidden", "conflict", "selfimg", "selfhidden")


def kbal(rows, key):
    byk = defaultdict(list)
    for r in rows:
        if r[key] is not None:
            byk[sum(r["vals"])].append(r[key])
    return float(np.mean([np.mean(v) for v in byk.values()])) if byk else float("nan")


def main():
    runs = sys.argv[1:] or DEFAULT
    grid = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        if r["set"] == "e3_grid":
            grid[r["rid"]] = (r["group"], r["n"], r["pattern"], r["rot"], r["predicate"], r["mode"])
    by_key = {v: k for k, v in grid.items()}
    res = {}
    for run in runs:
        path = os.path.join(RUNS, run, "eval_e3force.jsonl")
        if not os.path.exists(path):
            print("missing", run)
            continue
        gen = {o["rid"]: o["text"] for o in iter_jsonl(os.path.join(RUNS, run, "eval_e3_gen.jsonl"))}
        lg = {o["rid"]: o for o in iter_jsonl(os.path.join(RUNS, run, "eval_e3_logit.jsonl"))}
        rows = defaultdict(list)
        for o in iter_jsonl(path):
            pred = o["p_yes"] > 0.5
            fa = o["facts_answer"]
            if o["cond"] in ("selfimg", "selfhidden"):
                f = parse_facts(gen[o["grid_rid"]], o["n"])
                fa = e3.answer(o["predicate"], [f[i] for i in range(1, o["n"] + 1)]) if len(f) == o["n"] else None
            key = grid[o["grid_rid"]]
            lrid = by_key[key[:5] + ("logit",)]
            dwrong = (lg[lrid]["probs"][0] > 0.5) != o["answer"]
            rows[(o["cond"], o["n"], o["predicate"])].append(dict(
                vals=o["vals"], acc=float(pred == o["answer"]), follow=None if fa is None else float(pred == fa),
                dwrong=dwrong))
        tab = {}
        for (cond, n, op), rs in rows.items():
            x = dict(acc=kbal(rs, "acc"), follow=kbal(rs, "follow"), items=len(rs))
            if cond == "img":
                wrong = [1 - r["acc"] for r in rs]
                dw = [1 - r["acc"] for r in rs if r["dwrong"]]
                x.update(err=float(np.mean(wrong)), err_given_dwrong=float(np.mean(dw)) if dw else None)
            tab[f"{cond}|{n}|{op}"] = x
        for cond in CONDS:
            for n in (3, 6):
                xs = [tab[f"{cond}|{n}|{op}"] for op in e3.ops_for(n) if f"{cond}|{n}|{op}" in tab]
                if xs:
                    tab[f"{cond}|{n}|M"] = dict(acc=float(np.nanmean([x["acc"] for x in xs])),
                                                follow=float(np.nanmean([x["follow"] for x in xs])))
        res[run] = tab
        print(run, "done", flush=True)
    od = os.path.join(ROOT, "outputs", "e3dx")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "force.json"), "w") as f:
        json.dump(res, f, indent=1)
    ops = list(e3.OPS) + ["M"]
    L = ["# Teacher-forced facts probe (post hoc; single seed per arm)", ""]
    for cond, what in (("img", "acc"), ("hidden", "acc"), ("conflict", "follow"), ("selfimg", "acc"),
                       ("selfhidden", "acc"), ("selfimg", "follow"), ("selfhidden", "follow")):
        L += [f"## {cond}: {what}", "", "| run | n | " + " | ".join(ops) + " |", "|" + "---|" * (len(ops) + 2)]
        for run, tab in res.items():
            for n in (3, 6):
                L.append(f"| {run} | {n} | " + " | ".join(
                    f"{tab[f'{cond}|{n}|{op}'][what]:.3f}" if f"{cond}|{n}|{op}" in tab else "—" for op in ops) + " |")
        L.append("")
    L += ["## img: P(wrong | direct wrong) vs P(wrong), n = 6", "", "| run | " + " | ".join(e3.OPS) + " |",
          "|" + "---|" * (len(e3.OPS) + 1)]
    for run, tab in res.items():
        cells = []
        for op in e3.OPS:
            x = tab.get(f"img|6|{op}")
            cells.append("—" if not x or x["err_given_dwrong"] is None else f"{x['err_given_dwrong']:.2f} vs {x['err']:.2f}")
        L.append(f"| {run} | " + " | ".join(cells) + " |")
    with open(os.path.join(od, "force.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
