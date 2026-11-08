"""S1 metrics and the pre-registered decision rules (the preregistered analysis plan).

Metrics (accuracy; every group is cell-balanced: mean over cells of the mean over rows):
  ID       trained operators on ev_main: clevr AND/OR/GE/GT, coco AND/OR
  ID_hard  AND conflict cells (0,1),(1,0) on clevr and coco
  HO_op    held-out operators on ev_main: clevr XOR, coco XOR, clevr EQ
  HO_3img  three-image questions: {clevr, coco} x {ALL, ANY, ONE, NONE}
  T        mean(HO_op, HO_3img)            -- primary transfer score
  HO_tmpl  held-out paraphrases, all nine (source, operator) groups
  READ     single-image reading accuracy;  TEXT  text-control accuracy (guards)
  E_*      ideal-integrator null model from the run's own single-image readings; X_* = acc - E
Arms D, Dself, Dself23, Dselfclevr and the zero-shot reference ('base:gen') are scored from generated
'Answer:' lines.

Usage: python scripts/s1_analyze.py
"""
import itertools
import json
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib import logic  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402

RUNS = os.path.join(ROOT, "outputs", "s1", "runs")
ID_G = [("clevr", "AND"), ("clevr", "OR"), ("coco", "AND"), ("coco", "OR"), ("clevr", "GE"), ("clevr", "GT")]
HO_G = [("clevr", "XOR"), ("coco", "XOR"), ("clevr", "EQ")]
G3 = [(s, o) for s in ("clevr", "coco") for o in ("ALL", "ANY", "ONE", "NONE")]
TM_G = ID_G[:4] + [("clevr", "XOR"), ("coco", "XOR")] + ID_G[4:] + [("clevr", "EQ")]


def cell_bal(rows, groups, key="acc"):
    """rows: dicts with source, predicate, vals, and `key`. Mean over groups of cell-balanced means."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["source"], r["predicate"])][tuple(r["vals"])].append(r[key])
    out = {}
    for g in groups:
        cells = by.get(g)
        if cells:
            out[g] = st.mean(st.mean(v) for v in cells.values())
    return (st.mean(out.values()) if out else None), out


def cells(rows, source, pred, cell_list):
    v = [r["acc"] for r in rows if r["source"] == source and r["predicate"] == pred and tuple(r["vals"]) in cell_list]
    return st.mean(v) if v else None


def three_ideal(op, truth, qs):
    tot = 0.0
    for vals in itertools.product((0, 1), repeat=len(qs)):
        p = 1.0
        for q, v in zip(qs, vals):
            p *= q.get(v, 0.0)
        n = sum(vals)
        ans = {"ALL": n == len(vals), "ANY": n >= 1, "ONE": n == 1, "NONE": n == 0}[op]
        tot += p * (ans == truth)
    return tot


def evaluate(req, logit, gen=None):
    """req: rid -> request meta; logit/gen: rid -> output row."""
    read = defaultdict(lambda: defaultdict(float))
    single_rows = []
    for rid, o in logit.items():
        r = req[rid]
        if r["set"] != "ev_single":
            continue
        for sem, p in zip(r["sems"], o["probs"]):
            read[(r["source"], r["block"], r["descriptor"], r["image_id"])][int(sem)] += p
    for k, d in read.items():
        z = sum(d.values())
        for v in d:
            d[v] /= z
    gold_single = {(r["source"], r["block"], r["descriptor"], r["image_id"]): r["value"] for r in req.values()
                   if r["set"] == "ev_single"}
    for k, d in read.items():
        single_rows.append(dict(source=k[0], predicate=k[1], vals=[gold_single[k]], acc=float(max(d, key=d.get) == gold_single[k])))

    rows = defaultdict(list)
    src = gen if gen is not None else logit
    for rid, o in src.items():
        r = req[rid]
        if r["set"] in ("ev_single",) or (gen is not None and r["mode"] != "gen"):
            continue
        if gen is None and r["mode"] != "logit":
            continue
        pred = (o["probs"][0] > 0.5) if gen is None else o["pred"]
        row = dict(source=r["source"], predicate=r["predicate"], vals=r["vals"], acc=float(pred == r["answer"]))
        qs = [read.get((r["source"], r["block"], r["descriptor"], i)) for i in r.get("image_ids") or []]
        if qs and all(qs):
            row["E"] = (logic.ideal_integrator_accuracy(r["predicate"], r["answer"], qs[0], qs[1]) if len(qs) == 2
                        else three_ideal(r["predicate"], r["answer"], qs))
        rows[r["set"]].append(row)
    text_rows = []
    for rid, o in logit.items():
        r = req[rid]
        if r["set"] == "ev_text":
            text_rows.append(dict(source=r["source"], predicate=r["predicate"], vals=r["vals"],
                                  acc=float((o["probs"][0] > 0.5) == r["answer"])))

    m = {}
    main, three, tmpl = rows["ev_main"], rows["ev_3img"], rows["ev_tmpl"]
    m["ID"], m["ID_groups"] = cell_bal(main, ID_G)
    m["ID_hard"] = st.mean(x for x in (cells(main, s, "AND", [(0, 1), (1, 0)]) for s in ("clevr", "coco")) if x is not None)
    m["HO_op"], m["HO_op_groups"] = cell_bal(main, HO_G)
    m["HO_3img"], m["HO_3img_groups"] = cell_bal(three, G3)
    m["T"] = st.mean([m["HO_op"], m["HO_3img"]]) if m["HO_op"] is not None and m["HO_3img"] is not None else None
    m["HO_tmpl"], m["HO_tmpl_groups"] = cell_bal(tmpl, TM_G)
    m["HO_tmpl_trainedops"], _ = cell_bal(tmpl, ID_G)
    m["HO_tmpl_heldoutops"], _ = cell_bal(tmpl, HO_G)
    m["XOR11"] = st.mean(x for x in (cells(main, s, "XOR", [(1, 1)]) for s in ("clevr", "coco")) if x is not None)
    m["EQ_ties"] = cells(main, "clevr", "EQ", [(0, 0), (1, 1), (2, 2)])
    for name, rr, g in (("ID", main, ID_G), ("HO_op", main, HO_G), ("HO_3img", three, G3)):
        e_rows = [x for x in rr if "E" in x]
        if e_rows:
            m["E_" + name], _ = cell_bal(e_rows, g, key="E")
            if m["E_" + name] is not None and m[name] is not None:
                m["X_" + name] = m[name] - m["E_" + name]
    m["READ"], m["READ_groups"] = cell_bal(single_rows, [("clevr", "presence"), ("clevr", "count"), ("coco", "presence")])
    m["TEXT"], _ = cell_bal(text_rows, ID_G + HO_G)
    m["n"] = {k: len(v) for k, v in rows.items()}
    m["n"]["single"] = len(single_rows)
    m["n"]["text"] = len(text_rows)
    if gen is not None:
        ex = [o["explicit"] for rid, o in gen.items()]
        m["explicit_rate"] = st.mean(ex) if ex else None
    return m


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def main():
    req = {r["rid"]: {k: v for k, v in r.items() if k != "segments"}
           for r in iter_jsonl(os.path.join(ROOT, "data", "s1", "eval_requests.jsonl"))}
    res = {}
    for run in sorted(os.listdir(RUNS)):
        if run.endswith("_smoke"):
            continue
        lp = os.path.join(RUNS, run, "eval_logit.jsonl")
        if not os.path.exists(lp):
            continue
        logit = {o["rid"]: o for o in iter_jsonl(lp)}
        gp = os.path.join(RUNS, run, "eval_gen.jsonl")
        gen = {o["rid"]: o for o in iter_jsonl(gp)} if os.path.exists(gp) else None
        if run.startswith("D") and gen:          # D, Dself and the E3 Dself arms answer by generation
            res[run] = evaluate(req, logit, gen)
        else:
            res[run] = evaluate(req, logit)
            if run == "base" and gen:
                res["base:gen"] = evaluate(req, logit, gen)
    arms = defaultdict(list)
    for run, m in res.items():
        arms[run.split("_s")[0] if "_s" in run else run].append(m)
    keys = ["ID", "ID_hard", "E_ID", "HO_op", "XOR11", "EQ_ties", "HO_3img", "T", "HO_tmpl", "HO_tmpl_heldoutops",
            "READ", "TEXT"]
    lines = ["| run | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for run, m in sorted(res.items()):
        lines.append(f"| {run} | " + " | ".join(fmt(m.get(k)) for k in keys) + " |")
    lines += ["", "| arm (mean of seeds) | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    mean = {}
    for arm, ms in sorted(arms.items()):
        mean[arm] = {k: (st.mean(m[k] for m in ms) if all(m.get(k) is not None for m in ms) else None) for k in keys}
        lines.append(f"| {arm} (n={len(ms)}) | " + " | ".join(fmt(mean[arm][k]) for k in keys) + " |")

    # pre-registered decision rules
    dec = {}
    for run, m in res.items():
        if run not in ("base", "base:gen") and m.get("ID") is not None and m.get("E_ID") is not None:
            dec.setdefault("converged", {})[run] = m["ID"] >= min(0.95, m["E_ID"] - 0.02)
    if all(a in arms for a in ("C3", "A", "B")):
        best = max(("A", "B"), key=lambda a: mean[a]["T"])
        d = mean["C3"]["T"] - mean[best]["T"]
        seeds_ok = min(m["T"] for m in arms["C3"]) > max(m["T"] for m in arms[best])
        dec["C3_vs_best_of_A_B"] = dict(best=best, delta_T=round(d, 4), seeds_separate=seeds_ok,
                                        verdict="supported" if (d >= 0.05 and seeds_ok) else
                                        ("killed" if d <= 0.02 else "inconclusive"))
        dec["C3_label_free_parity_with_A"] = dict(dID=round(mean["C3"]["ID"] - mean["A"]["ID"], 4),
                                                  dT=round(mean["C3"]["T"] - mean["A"]["T"], 4))
    if all(a in arms for a in ("B", "Bd")):
        dec["blueprint_delta_term"] = {k: round(mean["Bd"][k] - mean["B"][k], 4) for k in ("ID", "T")}
    if all(a in arms for a in ("C3", "C3ans")):
        dec["rep_alignment"] = {k: round(mean["C3"][k] - mean["C3ans"][k], 4) for k in ("ID", "T")}
    if all(a in arms for a in ("C3", "D")):
        dec["D_minus_C3"] = {k: round(mean["D"][k] - mean["C3"][k], 4) for k in ("ID", "T")}
    if "base" in res:
        dec["guards"] = {run: dict(dREAD=round(m["READ"] - res["base"]["READ"], 4),
                                   dTEXT=round(m["TEXT"] - res["base"]["TEXT"], 4))
                         for run, m in res.items() if run not in ("base", "base:gen")}
    out = os.path.join(ROOT, "outputs", "s1")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(dict(runs={k: {kk: (vv if not isinstance(vv, dict) else {str(a): b for a, b in vv.items()})
                                 for kk, vv in v.items()} for k, v in res.items()}, decisions=dec), f, indent=1)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n\n```\n" + json.dumps(dec, indent=1) + "\n```\n")
    print("\n".join(lines))
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
