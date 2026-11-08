"""E3 metrics and the pre-registered hypotheses (the preregistered analysis plan).

For every run directory with eval_e3_logit.jsonl, two protocols are scored:
  direct  first-token Yes/No (eval_e3_logit.jsonl, set e3_grid)
  stated  state-then-answer generation, 'Answer:' parsed (eval_e3_gen.jsonl); an unparsed answer counts as wrong
The ideal-integration reference E uses the run's own single-image readings (set e3_single).

Aggregation (fixed before any E3 output existed):
  acc(source, n, op)  mean over the k values present (k = number of target images), each k a mean over its rows
  M_n                 mean over (source, op) of acc(source, n, op); E_n likewise; gap_n = E_n - M_n
  position            odd-one cells: ALL on pattern B1 (odd = the image without the target), ANY/ONE/NONE on A1
                      (odd = the image with it); FSA = acc(odd in slot 1) - mean acc(odd in slots 2..n)
  paired              same group, rotation and operator; patterns that differ in exactly one image; success = both
                      answers correct; 'change' when the truth differs, 'invariance' when it does not
  arm value           mean over seeds; pooled hypotheses (H1, H3, H4, H10) average the arm values
  CIs                 1,000 cluster-bootstrap resamples of the evaluation groups (within source), shared by all runs
Output: outputs/e3/summary.md, summary.json. Usage: python scripts/e3_analyze.py
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib import e3  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402

RUNS = os.environ.get("E3_RUNS", os.path.join(ROOT, "outputs", "s1", "runs"))      # overridable for tests
OUT = os.environ.get("E3_OUT", os.path.join(ROOT, "outputs", "e3"))
B = 1000
TRAINED3 = e3.TRAIN3_OPS
SINGLE_PASS = ("A", "B", "C3")


def load_requests():
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        r.pop("segments")
        req[r["rid"]] = r
    return req


def readings(req, logit):
    q = {}
    for rid, o in logit.items():
        r = req[rid]
        if r["set"] == "e3_single":
            p1 = sum(p for sem, p in zip(r["sems"], o["probs"]) if int(sem) == 1)
            q[(r["source"], r["descriptor"], r["image_id"])] = {1: p1, 0: 1 - p1}
    return q


def rows_for(req, q, outputs, protocol):
    rows = []
    for rid, o in outputs.items():
        r = req[rid]
        if r["set"] != "e3_grid":
            continue
        pred = (o["probs"][0] > 0.5) if protocol == "direct" else o.get("pred")
        qs = [q[(r["source"], r["descriptor"], i)] for i in r["image_ids"]]
        rows.append(dict(source=r["source"], group=r["group"], n=r["n"], op=r["predicate"], k=sum(r["vals"]),
                         pattern=r["pattern"], rot=r["rot"], answer=r["answer"],
                         correct=float(pred is not None and pred == r["answer"]),
                         E=e3.ideal(r["predicate"], r["answer"], qs)))
    return rows


class Tables:
    """Per-group sums for every cell, so that every metric can be recomputed under group resampling."""

    def __init__(self, groups):
        self.groups = groups
        self.gi = {g: i for i, g in enumerate(groups)}
        self.keys, self.ki = [], {}
        self.data = defaultdict(lambda: np.zeros((len(groups), 3)))   # correct, E, count

    def add(self, key, group, correct, E=0.0):
        if key not in self.ki:
            self.ki[key] = len(self.keys)
            self.keys.append(key)
        self.data[key][self.gi[group]] += (correct, E, 1.0)

    def matrix(self):
        return np.stack([self.data[k] for k in self.keys], axis=1)   # [G, C, 3]


def build_tables(rows, groups):
    acc, pos, pair = Tables(groups), Tables(groups), Tables(groups)
    index = {}
    for x in rows:
        acc.add((x["source"], x["n"], x["op"], x["k"]), x["group"], x["correct"], x["E"])
        odd = (x["op"] == "ALL" and x["pattern"] == "B1") or (x["op"] in ("ANY", "ONE", "NONE") and x["pattern"] == "A1")
        if odd:
            pos.add((x["n"], x["op"], x["rot"] + 1), x["group"], x["correct"])
        index[(x["group"], x["n"], x["rot"], x["op"], x["pattern"])] = x
    for (g, n, rot, op, pname), x in index.items():
        pats = e3.base_patterns(n)
        for pname2, v2 in pats.items():
            v1 = pats[pname]
            if pname2 <= pname or sum(a != b for a, b in zip(v1, v2)) != 1:
                continue
            y = index.get((g, n, rot, op, pname2))
            if y is None:
                continue
            kind = "change" if x["answer"] != y["answer"] else "invariance"
            pt = "-".join(sorted((pname, pname2)))
            pair.add((n, op, pt, kind), g, x["correct"] * y["correct"])
    return acc, pos, pair


def weights(groups, seed=0):
    rng = np.random.default_rng(seed)
    by_src = defaultdict(list)
    for i, g in enumerate(groups):
        by_src[g[:-2]].append(i)
    W = np.zeros((B + 1, len(groups)))
    W[0] = 1.0
    for b in range(1, B + 1):
        for idx in by_src.values():
            for j in rng.choice(idx, size=len(idx), replace=True):
                W[b, j] += 1
    return W


class Scored:
    """All metrics of one (run, protocol) under every resample (row 0 = point estimate)."""

    def __init__(self, rows, groups, W):
        self.acc, self.pos, self.pair = build_tables(rows, groups)
        self.W = W
        self._cache = {}

    def _cells(self, t):
        key = id(t)
        if key not in self._cache:
            m = np.einsum("bg,gcj->bcj", self.W, t.matrix())
            with np.errstate(invalid="ignore", divide="ignore"):
                self._cache[key] = (m[..., 0] / m[..., 2], m[..., 1] / m[..., 2])
        return self._cache[key]

    def M(self, ns, ops=None, sources=("clevr", "coco"), what="acc"):
        """Mean over (source, n, op) of the k-balanced accuracy (or E); vector over resamples."""
        a, e = self._cells(self.acc)
        v = a if what == "acc" else e
        grp = defaultdict(list)
        for c, (s, n, op, k) in enumerate(self.acc.keys):
            if n in ns and s in sources and (ops is None or op in ops):
                grp[(s, n, op)].append(c)
        return np.mean([np.nanmean(v[:, idx], axis=1) for idx in grp.values()], axis=0)

    def gap(self, ns, ops=None, sources=("clevr", "coco")):
        return self.M(ns, ops, sources, "E") - self.M(ns, ops, sources)

    def fsa(self, ns, op="ALL"):
        a, _ = self._cells(self.pos)
        out = []
        for n in ns:
            col = {slot: c for c, (nn, o, slot) in enumerate(self.pos.keys) if nn == n and o == op}
            if 1 in col and len(col) > 1:
                out.append(a[:, col[1]] - np.mean([a[:, c] for s, c in col.items() if s != 1], axis=0))
        return np.mean(out, axis=0)

    def slots(self, n, op):
        a, _ = self._cells(self.pos)
        return {slot: float(a[0, c]) for c, (nn, o, slot) in enumerate(self.pos.keys) if nn == n and o == op}

    def paired(self, ns, op, kind, ptype=None):
        a, _ = self._cells(self.pair)
        cols = [c for c, (n, o, pt, kd) in enumerate(self.pair.keys)
                if n in ns and o == op and kd == kind and (ptype is None or pt == ptype)]
        if not cols:
            return np.full(self.W.shape[0], np.nan)
        m = np.einsum("bg,gcj->bcj", self.W, self.pair.matrix()[:, cols])
        return m[..., 0].sum(1) / m[..., 2].sum(1)


def ci(v):
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
    return dict(est=round(float(v[0]), 4), lo=round(float(lo), 4), hi=round(float(hi), 4))


def main():
    req = load_requests()
    groups = sorted({r["group"] for r in req.values() if r["set"] == "e3_grid"})
    W = weights(groups)
    S = {}
    for run in sorted(os.listdir(RUNS)):
        lp, gp = (os.path.join(RUNS, run, f"eval_e3_{m}.jsonl") for m in ("logit", "gen"))
        if run.endswith("_smoke") or not os.path.exists(lp):
            continue
        logit = {o["rid"]: o for o in iter_jsonl(lp)}
        n_grid = sum(1 for r in req.values() if r["set"] == "e3_grid" and r["mode"] == "logit")
        if sum(1 for rid in logit if req[rid]["set"] == "e3_grid") < n_grid:
            print("skip incomplete", run)
            continue
        q = readings(req, logit)
        S[(run, "direct")] = Scored(rows_for(req, q, logit, "direct"), groups, W)
        if os.path.exists(gp):
            gen = {o["rid"]: o for o in iter_jsonl(gp)}
            if len(gen) == n_grid:
                S[(run, "stated")] = Scored(rows_for(req, q, gen, "stated"), groups, W)
    arms = defaultdict(list)
    for (run, proto) in S:
        arms[(run.split("_s")[0] if "_s" in run else run, proto)].append(run)

    def armv(arm, proto, f):
        """Mean over the arm's seeds of f(Scored) (vector over resamples)."""
        runs = arms.get((arm, proto))
        return None if not runs else np.mean([f(S[(r, proto)]) for r in runs], axis=0)

    def seeds(arm, proto, f):
        return [float(f(S[(r, proto)])[0]) for r in arms.get((arm, proto), [])]

    # ------------------------------------------------------------- descriptive tables
    lines = ["| run | protocol | " + " | ".join(f"M{n}" for n in range(2, 7)) + " | " +
             " | ".join(f"gap{n}" for n in range(2, 7)) + " | FSA_ALL(3-6) |", "|" + "---|" * 13]
    desc = {}
    for (run, proto), sc in sorted(S.items()):
        m = [float(sc.M([n])[0]) for n in range(2, 7)]
        g = [float(sc.gap([n])[0]) for n in range(2, 7)]
        f = float(sc.fsa(range(3, 7))[0])
        desc[f"{run}|{proto}"] = dict(M=m, gap=g, FSA_ALL=f,
                                      by_op={op: [float(sc.M([n], [op])[0]) for n in range(2, 7) if op in e3.ops_for(n)]
                                             for op in e3.OPS},
                                      slots_ALL={n: sc.slots(n, "ALL") for n in range(2, 7)},
                                      paired_B1F={kind: {op: float(sc.paired([6], op, kind, "B1-F")[0])
                                                         for op in e3.OPS} for kind in ("change", "invariance")})
        lines.append(f"| {run} | {proto} | " + " | ".join(f"{x:.3f}" for x in m) + " | " +
                     " | ".join(f"{x:.3f}" for x in g) + f" | {f:+.3f} |")

    # ------------------------------------------------------------- pre-registered hypotheses
    H = {}
    have = lambda a, p: (a, p) in arms  # noqa: E731
    pool_sp = [a for a in ("base",) + SINGLE_PASS if have(a, "direct")]

    def pooled(arms_, proto, f):
        vals = [armv(a, proto, f) for a in arms_ if have(a, proto)]
        return np.mean(vals, axis=0) if vals else None

    d63 = lambda sc: sc.gap([6]) - sc.gap([3])  # noqa: E731
    v = pooled(pool_sp, "direct", d63)
    if v is not None:
        H["H1"] = dict(pooled=ci(v), per_arm={a: ci(armv(a, "direct", d63)) for a in pool_sp}, threshold=0.05,
                       holds=bool(v[0] >= 0.05))
    if have("D", "stated") and have("Dself", "stated"):
        g6 = {a: ci(armv(a, "stated", lambda sc: sc.gap([6]))) for a in ("D", "Dself")}
        H["H2a"] = dict(per_arm=g6, threshold=0.06, holds=all(x["est"] <= 0.06 for x in g6.values()))
    v = pooled(SINGLE_PASS, "stated", lambda sc: sc.gap([6]))
    if v is not None:
        H["H2b"] = dict(pooled=ci(v), threshold=0.10, holds=bool(v[0] <= 0.10))
    if all(have(a, p) for a in pool_sp for p in ("direct", "stated")):
        vals = [armv(a, "stated", lambda sc: sc.M([6])) - armv(a, "direct", lambda sc: sc.M([6]))
                - armv(a, "stated", lambda sc: sc.M([3])) + armv(a, "direct", lambda sc: sc.M([3])) for a in pool_sp]
        v = np.mean(vals, axis=0)
        H["H3"] = dict(pooled=ci(v), threshold=0.05, holds=bool(v[0] >= 0.05))
    v = pooled(pool_sp, "direct", lambda sc: sc.fsa(range(3, 7)))
    if v is not None:
        H["H4a"] = dict(pooled=ci(v), per_arm={a: ci(armv(a, "direct", lambda sc: sc.fsa(range(3, 7))))
                                               for a in pool_sp}, threshold=0.10, holds=bool(v[0] >= 0.10))
    if have("D", "stated") and have("Dself", "stated"):
        f4 = {a: ci(armv(a, "stated", lambda sc: sc.fsa(range(3, 7)))) for a in ("D", "Dself")}
        H["H4b"] = dict(per_arm=f4, threshold=0.05, holds=all(abs(x["est"]) <= 0.05 for x in f4.values()))
    if have("A23", "direct"):
        g3 = seeds("A23", "direct", lambda sc: sc.gap([3], TRAINED3))
        g6 = seeds("A23", "direct", lambda sc: sc.gap([6], TRAINED3))
        strong = all(a <= 0.03 and b - a >= 0.07 for a, b in zip(g3, g6))
        down = all(b - a <= 0.03 for a, b in zip(g3, g6))
        H["H5"] = dict(gap3_seeds=g3, gap6_seeds=g6, verdict="count-bound" if strong else
                       ("extrapolates" if down else "intermediate"))
    if have("Dself23", "stated") and have("A23", "direct"):
        f46 = lambda sc: sc.M([4, 5, 6])  # noqa: E731
        d = armv("Dself23", "stated", f46) - armv("A23", "direct", f46)
        sep = min(seeds("Dself23", "stated", f46)) > max(seeds("A23", "direct", f46))
        H["H6a"] = dict(diff=ci(d), seeds_separate=sep, holds=bool(d[0] >= 0.05 and sep))
        if have("A23", "stated"):
            d2 = armv("Dself23", "stated", f46) - armv("A23", "stated", f46)
            H["H6b"] = dict(diff=ci(d2), holds=bool(d2[0] >= 0.03))
        if have("Dself23p", "stated"):                # contingency arm (amendment): reported, not a test
            d3 = armv("Dself23p", "stated", f46) - armv("A23", "direct", f46)
            H["H6_supp_Dself23p"] = dict(diff_vs_A23_direct=ci(d3), seeds_separate=min(seeds("Dself23p", "stated", f46))
                                         > max(seeds("A23", "direct", f46)))
    if have("A23", "direct"):
        v = armv("A23", "direct", lambda sc: sc.gap([4, 5, 6], ("ATLEAST2",)))
        H["H7"] = dict(gap=ci(v), threshold=0.10, holds=bool(v[0] >= 0.10))
    if have("Aall", "direct") and have("A", "direct"):
        f36 = lambda sc: sc.M([3, 4, 5, 6])  # noqa: E731
        d = armv("Aall", "direct", f36) - armv("A", "direct", f36)
        H["H8"] = dict(diff=ci(d), holds=bool(abs(d[0]) <= 0.03))
    coco = lambda sc: sc.M([2, 3, 4, 5, 6], sources=("coco",))  # noqa: E731
    if have("Aclevr", "direct") and have("A", "direct"):
        d = armv("Aclevr", "direct", coco) - armv("A", "direct", coco)
        H["H9a"] = dict(diff=ci(d), holds=bool(abs(d[0]) <= 0.03))
    if have("Dselfclevr", "stated") and have("Dself", "stated"):
        d = armv("Dselfclevr", "stated", coco) - armv("Dself", "stated", coco)
        H["H9b"] = dict(diff=ci(d), holds=bool(abs(d[0]) <= 0.03))
    ch = pooled(pool_sp, "direct", lambda sc: sc.paired([6], "ALL", "change", "B1-F"))
    inv = pooled(pool_sp, "direct", lambda sc: sc.paired([6], "ANY", "invariance", "B1-F"))
    if ch is not None:
        H["H10a"] = dict(ALL_change=ci(ch), ANY_invariance=ci(inv), holds=bool(ch[0] <= 0.6 and inv[0] >= 0.85))
    if have("D", "stated") and have("Dself", "stated"):
        c = {a: ci(armv(a, "stated", lambda sc: sc.paired([6], "ALL", "change", "B1-F"))) for a in ("D", "Dself")}
        H["H10b"] = dict(per_arm=c, holds=all(x["est"] >= 0.85 for x in c.values()))
    # sufficiency of the new arms on E3 (S1-set sufficiency comes from outputs/s1/summary.json)
    suff = {}
    for arm, proto in (("A23", "direct"), ("Dself23", "stated"), ("Dself23p", "stated")):
        if have(arm, proto):
            for r in arms[(arm, proto)]:
                sc = S[(r, proto)]
                acc3, e3v = float(sc.M([3], TRAINED3)[0]), float(sc.M([3], TRAINED3, what="E")[0])
                suff[r] = dict(acc_n3_trained=round(acc3, 4), E=round(e3v, 4), ok=acc3 >= min(0.95, e3v - 0.02))
    H["sufficiency_E3"] = suff
    if "H5" in H and "H4a" in H:
        H["claim_version"] = ("strong" if H["H5"]["verdict"] == "count-bound" and H["H4a"]["holds"] else
                              "downgraded" if H["H5"]["verdict"] == "extrapolates" else "main")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(dict(descriptive=desc, hypotheses=H, runs=sorted(f"{r}|{p}" for r, p in S)), f, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n\n```\n" + json.dumps(H, indent=1, default=str) + "\n```\n")
    print("\n".join(lines))
    print(json.dumps(H, indent=1, default=str))


if __name__ == "__main__":
    main()
