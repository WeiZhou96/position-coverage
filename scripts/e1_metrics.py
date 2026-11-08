"""E1 remaining metrics on the S1 evaluation set (descriptive re-analysis of existing outputs; no model runs).

For every S1 checkpoint (base; A, B, Bd, C3, C3ans, D, Dself x 2 seeds) under both inference protocols
  direct  first-token Yes/No scoring (eval_logit.jsonl)
  stated  state-then-answer, generated 'Answer:' line (eval_gen.jsonl; unparsed answers count as wrong, as in
          s1_analyze.py)
it reports, for the S1 aggregates (ID, ID_hard, HO_op, EQ, three-image split into trained-operator extension
ALL/ANY and new operators ONE/NONE, T, HO_tmpl):
  acc   cell-balanced accuracy (identical to s1_analyze.py / e1_xproto_analyze.py)
  cls   class-balanced accuracy: per group, mean of the cell-balanced accuracy over yes-cells and over no-cells
  base  constant baseline: best of always-Yes / always-No under the same cell balancing
  E     ideal-integration reference from the checkpoint's own single-image readings (as in s1_analyze.py)
  GC    gap closure (acc - base) / (E - base); only for aggregates whose groups contain both answer classes
  full truth table: accuracy, E and mean P(yes) per (set, source, operator, cell)
95% CIs: cluster bootstrap, 1000 resamples. Clusters: connected components of image sharing among two-image pairs
(ev_main / ev_tmpl), and triples for ev_3img; resampled within (source x pair/triple). Arm values are seed means
computed on the same draws; paired contrasts use the same draws for both sides. The bootstrap p-value is
2 x min(P(diff <= 0), P(diff >= 0)) over the resamples (approximate).
Outputs: outputs/e1/metrics/{metrics.json, truth_table.csv, tables.md}
Usage: python scripts/e1_metrics.py [--boot 1000]
"""
import argparse
import csv
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_analyze import G3, HO_G, ID_G, ROOT, RUNS, TM_G, three_ideal  # noqa: E402
from e0lib import logic  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402

ARMS = ["A", "B", "Bd", "C3", "C3ans", "D", "Dself"]
RUN_LIST = ["base"] + [f"{a}_s{s}" for a in ARMS for s in (1, 2)]
PROTOS = {"direct": "logit", "stated": "gen"}
SETS = ("ev_main", "ev_3img", "ev_tmpl")
SEED = 20261105
TIES = {(0, 0), (1, 1), (2, 2)}
STRICT = {c for c in itertools.product(range(3), repeat=2) if c not in TIES}
G3_EXT = [g for g in G3 if g[1] in ("ALL", "ANY")]
G3_NEW = [g for g in G3 if g[1] in ("ONE", "NONE")]
AGG = {
    "ID": [("ev_main", s, p, None) for s, p in ID_G],
    "ID_hard": [("ev_main", s, "AND", {(0, 1), (1, 0)}) for s in ("clevr", "coco")],
    "HO_op": [("ev_main", s, p, None) for s, p in HO_G],
    "XOR": [("ev_main", s, "XOR", None) for s in ("clevr", "coco")],
    "EQ": [("ev_main", "clevr", "EQ", None)],
    "EQ_ties": [("ev_main", "clevr", "EQ", TIES)],
    "EQ_strict": [("ev_main", "clevr", "EQ", STRICT)],
    "HO_3ext": [("ev_3img", s, p, None) for s, p in G3_EXT],
    "HO_3new": [("ev_3img", s, p, None) for s, p in G3_NEW],
    "HO_3img": [("ev_3img", s, p, None) for s, p in G3],
    "HO_tmpl": [("ev_tmpl", s, p, None) for s, p in TM_G],
    # sensitivity: the S1 NONE wording ("Do none of the three images contain X?") is ambiguous at k = 0
    # (report 10.2, rewritten in E3), so ONE and NONE are also reported separately and T without NONE
    "ONE": [("ev_3img", s, "ONE", None) for s in ("clevr", "coco")],
    "NONE": [("ev_3img", s, "NONE", None) for s in ("clevr", "coco")],
    "HO_3img_xNONE": [("ev_3img", s, p, None) for s, p in G3 if p != "NONE"],
}
T_DEF = {"T": ("HO_op", "HO_3img"), "T_xNONE": ("HO_op", "HO_3img_xNONE")}
ORDER = ["ID", "ID_hard", "HO_op", "XOR", "EQ", "EQ_ties", "EQ_strict", "HO_3ext", "HO_3new", "ONE", "NONE",
         "HO_3img", "HO_3img_xNONE", "T", "T_xNONE", "HO_tmpl"]
PSEUDO = {"ABC3": ["A", "B", "C3"]}       # single-pass reference used in report 10.10


def load_req():
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "s1", "eval_requests.jsonl")):
        r.pop("segments", None)
        req[r["rid"]] = r
    return req


def cluster_ids(req):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in req.values():
        if r["set"] in ("ev_main", "ev_tmpl"):
            for i in r["image_ids"]:
                a, b = find(("pair", r["pair_id"])), find(("img", r["source"], i))
                if a != b:
                    parent[a] = b
    cid = {}
    for rid, r in req.items():
        if r["set"] in ("ev_main", "ev_tmpl"):
            cid[rid] = ("pair", r["source"], find(("pair", r["pair_id"])))
        elif r["set"] == "ev_3img":
            cid[rid] = ("trip", r["source"], r["trip_id"])
    return cid


def readings(req, logit):
    read = defaultdict(lambda: defaultdict(float))
    for rid, o in logit.items():
        r = req[rid]
        if r["set"] == "ev_single":
            for sem, p in zip(r["sems"], o["probs"]):
                read[(r["source"], r["block"], r["descriptor"], r["image_id"])][int(sem)] += p
    for d in read.values():
        z = sum(d.values())
        for v in d:
            d[v] /= z
    return read


def item_E(r, read):
    qs = [read.get((r["source"], r["block"], r["descriptor"], i)) for i in r["image_ids"]]
    if not all(qs):
        return np.nan
    if len(qs) == 2:
        return logic.ideal_integrator_accuracy(r["predicate"], r["answer"], qs[0], qs[1])
    return three_ideal(r["predicate"], r["answer"], qs)


def boot_weights(strata, n_boot, rng):
    """(n_boot + 1, C) cluster multiplicities; row 0 is the full sample."""
    C = len(strata)
    W = np.ones((n_boot + 1, C))
    by = defaultdict(list)
    for c, s in enumerate(strata):
        by[s].append(c)
    for s, idx in by.items():
        idx = np.array(idx)
        for b in range(1, n_boot + 1):
            draw = rng.integers(0, len(idx), len(idx))
            W[b, idx] = np.bincount(draw, minlength=len(idx))
    return W


def summ(arr):
    if arr is None or np.isnan(arr[0]):
        return None
    lo, hi = np.nanpercentile(arr[1:], [2.5, 97.5])
    return [float(arr[0]), float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    req = load_req()
    cid = cluster_ids(req)
    clusters = sorted(set(cid.values()))
    cidx = {c: i for i, c in enumerate(clusters)}
    strata = [(c[0], c[1]) for c in clusters]
    keys, kidx = [], {}
    for rid, r in sorted(req.items()):
        if r["set"] in SETS:
            k = (r["set"], r["source"], r["predicate"], tuple(r["vals"]))
            if k not in kidx:
                kidx[k] = len(keys)
                keys.append(k + (bool(r["answer"]),))
    C, K = len(clusters), len(keys)
    items = {}
    for mode in ("logit", "gen"):
        rids = sorted(rid for rid, r in req.items() if r["set"] in SETS and r["mode"] == mode)
        items[mode] = (rids, np.array([cidx[cid[x]] for x in rids]),
                       np.array([kidx[(req[x]["set"], req[x]["source"], req[x]["predicate"], tuple(req[x]["vals"]))]
                                 for x in rids]), [bool(req[x]["answer"]) for x in rids])
    rng = np.random.default_rng(SEED)
    W = boot_weights(strata, args.boot, rng)
    print(f"clusters {C} ({dict((s, strata.count(s)) for s in sorted(set(strata)))}), keys {K}, "
          f"boot {args.boot}", flush=True)

    # group index cache
    ginfo = {}
    for name, spec in AGG.items():
        groups = []
        for (st_, src, pred, cf) in spec:
            ks = [i for i, k in enumerate(keys) if k[0] == st_ and k[1] == src and k[2] == pred
                  and (cf is None or k[3] in cf)]
            assert ks, (name, st_, src, pred)
            yes = [i for i in ks if keys[i][4]]
            no = [i for i in ks if not keys[i][4]]
            fy = len(yes) / len(ks)
            groups.append(dict(ks=ks, yes=yes, no=no, base=max(fy, 1 - fy)))
        ginfo[name] = groups

    def mat(vals, cl, ki, mask):
        M = np.zeros((C, K))
        np.add.at(M, (cl[mask], ki[mask]), vals[mask])
        return M

    rep, ttab = {}, []
    for run in RUN_LIST:
        d = os.path.join(RUNS, run)
        logit = {o["rid"]: o for o in iter_jsonl(os.path.join(d, "eval_logit.jsonl"))}
        gen = {o["rid"]: o for o in iter_jsonl(os.path.join(d, "eval_gen.jsonl"))}
        read = readings(req, logit)
        for proto, mode in PROTOS.items():
            out = logit if mode == "logit" else gen
            rids, cl, ki, ans = items[mode]
            acc = np.zeros(len(rids))
            E = np.full(len(rids), np.nan)
            py = np.full(len(rids), np.nan)
            has = np.zeros(len(rids), bool)
            for j, rid in enumerate(rids):
                o = out.get(rid)
                if o is None:
                    continue
                has[j] = True
                if mode == "logit":
                    py[j] = o["probs"][0]
                    pred = py[j] > 0.5
                else:
                    pred = o["pred"]
                acc[j] = float(pred == ans[j])
                E[j] = item_E(req[rid], read)
            assert has.all(), f"{run} {proto}: {(~has).sum()} missing outputs"
            assert np.isfinite(E).all(), f"{run} {proto}: missing readings"
            N = mat(np.ones(len(rids)), cl, ki, has)
            WN = W @ N
            CM = (W @ mat(acc, cl, ki, has)) / WN
            CE = (W @ mat(E, cl, ki, has)) / WN
            CP = (W @ mat(np.nan_to_num(py), cl, ki, has)) / WN if mode == "logit" else None
            res = {}
            for name, groups in ginfo.items():
                ga, gc, ge, gec = [], [], [], []
                for g in groups:
                    ga.append(CM[:, g["ks"]].mean(1))
                    ge.append(CE[:, g["ks"]].mean(1))
                    if g["yes"] and g["no"]:
                        gc.append(0.5 * (CM[:, g["yes"]].mean(1) + CM[:, g["no"]].mean(1)))
                        gec.append(0.5 * (CE[:, g["yes"]].mean(1) + CE[:, g["no"]].mean(1)))
                two = len(gc) == len(groups)
                a, e = np.mean(ga, 0), np.mean(ge, 0)
                base = float(np.mean([g["base"] for g in groups]))
                res[name] = dict(acc=a, E=e, base=base,
                                 cls=np.mean(gc, 0) if two else None, Ecls=np.mean(gec, 0) if two else None,
                                 GC=(a - base) / (e - base) if two else None)
            for tname, (p1, p2) in T_DEF.items():
                t = {m: (res[p1][m] + res[p2][m]) / 2 for m in ("acc", "E", "cls", "Ecls")}
                tb = (res[p1]["base"] + res[p2]["base"]) / 2
                res[tname] = dict(t, base=tb, GC=(t["acc"] - tb) / (t["E"] - tb))
            rep[(run, proto)] = res
            ncell = N.sum(0)
            for k, key in enumerate(keys):
                ttab.append(dict(run=run, protocol=proto, set=key[0], source=key[1], op=key[2],
                                 cell="".join(map(str, key[3])), answer=int(key[4]), n=int(ncell[k]),
                                 acc=round(float(CM[0, k]), 4), E=round(float(CE[0, k]), 4),
                                 p_yes=(round(float(CP[0, k]), 4) if CP is not None else "")))
        print(f"{run} done", flush=True)

    # arms (seed means on the same draws)
    arm_runs = {"base": ["base"]}
    arm_runs.update({a: [f"{a}_s1", f"{a}_s2"] for a in ARMS})
    arm_runs.update({p: [f"{a}_s{s}" for a in members for s in (1, 2)] for p, members in PSEUDO.items()})

    def arm_vec(arm, proto, name, m):
        vs = [rep[(r, proto)][name][m] for r in arm_runs[arm]]
        return None if any(v is None for v in vs) else np.mean(vs, 0)

    # validation against the existing cross-protocol table
    xp = json.load(open(os.path.join(ROOT, "outputs", "e1", "xproto.json")))
    maxdiff = 0.0
    for (run, proto), res in rep.items():
        old = xp.get(f"{run}|{proto}")
        if old:
            for k in ("ID", "ID_hard", "HO_op", "EQ_ties", "HO_3img", "T", "HO_tmpl"):
                if old.get(k) is not None:
                    maxdiff = max(maxdiff, abs(old[k] - float(res[k]["acc"][0])))
    print(f"max |diff| vs outputs/e1/xproto.json: {maxdiff:.2e}", flush=True)
    assert maxdiff < 1e-9

    best_ab = max(("A", "B"), key=lambda a: float(arm_vec(a, "direct", "T", "acc")[0]))
    contrasts = [(f"protocol switch: {a}", (a, "stated"), (a, "direct"))
                 for a in ("base", "A", "B", "Bd", "C3", "C3ans", "ABC3")]
    contrasts += [
        ("format training: D vs ABC3, both stated", ("D", "stated"), ("ABC3", "stated")),
        ("format training: Dself vs ABC3, both stated", ("Dself", "stated"), ("ABC3", "stated")),
        ("total: D stated vs ABC3 direct", ("D", "stated"), ("ABC3", "direct")),
        ("total: Dself stated vs ABC3 direct", ("Dself", "stated"), ("ABC3", "direct")),
        ("same targets: Dself stated vs C3ans direct", ("Dself", "stated"), ("C3ans", "direct")),
        ("same targets, same protocol: Dself vs C3ans, both stated", ("Dself", "stated"), ("C3ans", "stated")),
        ("no internalisation: D direct vs base direct", ("D", "direct"), ("base", "direct")),
        ("no internalisation: Dself direct vs base direct", ("Dself", "direct"), ("base", "direct")),
        (f"S1 prereg: C3 vs best of A/B ({best_ab}), direct", ("C3", "direct"), (best_ab, "direct")),
        ("label source: Dself vs D, both stated", ("Dself", "stated"), ("D", "stated")),
    ]
    cmetrics = ["T", "T_xNONE", "HO_op", "HO_3img", "HO_3ext", "HO_3new", "ONE", "NONE", "EQ", "ID", "ID_hard",
                "HO_tmpl"]
    cres = []
    for label, x, y in contrasts:
        row = dict(label=label, x=list(x), y=list(y), metrics={})
        for name in cmetrics:
            for m in ("acc", "cls", "GC"):
                vx, vy = arm_vec(*x, name, m), arm_vec(*y, name, m)
                if vx is None or vy is None:
                    continue
                dd = vx - vy
                p = float(min(1.0, 2 * min(np.mean(dd[1:] <= 0), np.mean(dd[1:] >= 0))))
                sx = [float(rep[(r, x[1])][name][m][0]) for r in arm_runs[x[0]]]
                sy = [float(rep[(r, y[1])][name][m][0]) for r in arm_runs[y[0]]]
                sep = (min(sx) > max(sy)) or (max(sx) < min(sy))
                row["metrics"][f"{name}.{m}"] = dict(delta=summ(dd), p=p, seeds_separate=bool(sep))
        cres.append(row)

    # outputs
    outd = os.path.join(ROOT, "outputs", "e1", "metrics")
    os.makedirs(outd, exist_ok=True)
    js = dict(meta=dict(boot=args.boot, seed=SEED, clusters=C, keys=K, best_of_A_B=best_ab,
                        cluster_strata={f"{a}|{b}": strata.count((a, b)) for a, b in sorted(set(strata))}),
              runs={}, arms={}, contrasts=cres)
    for (run, proto), res in rep.items():
        js["runs"][f"{run}|{proto}"] = {n: {m: (summ(v[m]) if m != "base" else v[m]) for m in v} for n, v in res.items()}
    for arm in arm_runs:
        for proto in PROTOS:
            js["arms"][f"{arm}|{proto}"] = {
                n: dict({m: summ(arm_vec(arm, proto, n, m)) for m in ("acc", "cls", "E", "Ecls", "GC")},
                        base=rep[(arm_runs[arm][0], proto)][n]["base"],
                        seeds=[float(rep[(r, proto)][n]["acc"][0]) for r in arm_runs[arm]])
                for n in ORDER}
    with open(os.path.join(outd, "metrics.json"), "w") as f:
        json.dump(js, f, indent=1)
    with open(os.path.join(outd, "truth_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ttab[0].keys()))
        w.writeheader()
        w.writerows(ttab)

    def f3(s):
        return "—" if s is None else f"{s[0]:.3f} [{s[1]:.3f}, {s[2]:.3f}]"

    L = ["# E1 remaining metrics (S1 evaluation set)", "",
         f"Cluster bootstrap {args.boot} resamples, seed {SEED}; clusters {js['meta']['cluster_strata']}.", ""]
    cols = ["ID", "ID_hard", "HO_op", "EQ", "HO_3ext", "ONE", "NONE", "HO_3img", "T", "T_xNONE", "HO_tmpl"]
    for m, title in (("acc", "Cell-balanced accuracy"), ("cls", "Class-balanced accuracy"),
                     ("GC", "Gap closure (acc - base) / (E - base)")):
        L += [f"## {title} (arm = mean of seeds, 95% CI)", "", "| arm | protocol | " + " | ".join(cols) + " |",
              "|" + "---|" * (len(cols) + 2)]
        for arm in arm_runs:
            for proto in PROTOS:
                a = js["arms"][f"{arm}|{proto}"]
                L.append(f"| {arm} | {proto} | " + " | ".join(f3(a[c][m]) for c in cols) + " |")
        L.append("")
    L += ["## Constant baseline and ideal-integration reference E (cell-balanced; E is the base model's)", "",
          "| quantity | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1),
          "| constant baseline | " + " | ".join(f"{js['arms']['base|direct'][c]['base']:.3f}" for c in cols) + " |",
          "| E (base) | " + " | ".join(f3(js["arms"]["base|direct"][c]["E"]) for c in cols) + " |",
          "| E (base, class-balanced) | " + " | ".join(f3(js["arms"]["base|direct"][c]["Ecls"]) for c in cols)
          + " |", ""]
    L += ["## Paired contrasts (x - y; arm = mean of seeds; same bootstrap draws)", "",
          "| contrast | metric | Δ acc [CI] | p | seeds sep. | Δ cls [CI] | Δ GC [CI] |", "|---|---|---|---|---|---|---|"]
    for row in cres:
        for name in ("T", "T_xNONE", "HO_op", "HO_3ext", "ONE", "NONE", "EQ"):
            a = row["metrics"].get(f"{name}.acc")
            c = row["metrics"].get(f"{name}.cls")
            g = row["metrics"].get(f"{name}.GC")
            if a is None:
                continue
            L.append(f"| {row['label']} | {name} | {f3(a['delta'])} | {a['p']:.3f} | "
                     f"{'yes' if a['seeds_separate'] else 'no'} | {f3(c['delta']) if c else '—'} | "
                     f"{f3(g['delta']) if g else '—'} |")
    L.append("")
    L += ["## Truth tables (accuracy per cell; arm = mean of seeds)", ""]
    tt = defaultdict(list)
    for r in ttab:
        arm = r["run"].split("_s")[0] if "_s" in r["run"] else r["run"]
        tt[(arm, r["protocol"], r["set"], r["source"], r["op"], r["cell"])].append(r)
    show = ["base", "A", "B", "C3", "C3ans", "D", "Dself"]
    for st_, title in (("ev_main", "two-image (ev_main)"), ("ev_3img", "three-image (ev_3img)")):
        for proto in PROTOS:
            L += [f"### {title}, {proto}", "", "| source | op | cell | truth | E(base) | " + " | ".join(show) + " |",
                  "|" + "---|" * (len(show) + 5)]
            for key in sorted(keys):
                if key[0] != st_:
                    continue
                cell = "".join(map(str, key[3]))
                vals = []
                for arm in show:
                    rs = tt[(arm, proto, st_, key[1], key[2], cell)]
                    vals.append(f"{np.mean([x['acc'] for x in rs]):.2f}")
                e0 = tt[("base", proto, st_, key[1], key[2], cell)][0]["E"]
                L.append(f"| {key[1]} | {key[2]} | {cell} | {'Y' if key[4] else 'N'} | {e0:.2f} | "
                         + " | ".join(vals) + " |")
            L.append("")
    with open(os.path.join(outd, "tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", outd, flush=True)


if __name__ == "__main__":
    with np.errstate(invalid="ignore", divide="ignore"):
        main()
