"""E0 analysis (design v2, sections 7-8): per-item quantities, null model, tests, gates."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import logic

EPS = 1e-12


# --------------------------------------------------------------------------- loading
def sem_probs(req: Dict, res: Dict) -> Dict:
    return {s: p for s, p in zip(req["sems"], res["probs"])}


def avg_dist(dists: List[Dict]) -> Optional[Dict]:
    """Average of log-probabilities per semantic option, renormalised."""
    dists = [d for d in dists if d]
    if not dists:
        return None
    keys = dists[0].keys()
    lp = {k: float(np.mean([math.log(max(d[k], EPS)) for d in dists])) for k in keys}
    m = max(lp.values())
    z = sum(math.exp(v - m) for v in lp.values())
    return {k: math.exp(v - m) / z for k, v in lp.items()}


def argmax(d: Dict):
    return max(d.items(), key=lambda kv: kv[1])[0]


def index(requests: Iterable[Dict], results: Dict[str, Dict]):
    rel, none_rel, text, ps, pc, same = (defaultdict(list) for _ in range(6))
    for r in requests:
        res = results.get(r["rid"])
        if res is None:
            continue
        d = sem_probs(r, res)
        k = r["kind"]
        if k == "rel" and r.get("item_id"):
            rel[(r["item_id"], r["fmt"], r["visible"], r["order"])].append((r["template"], d))
        elif k == "rel":
            none_rel[(r["source"], r["block"], r["descriptor"], r["predicate"], r["fmt"])].append((r["template"], d))
        elif k == "text":
            text[r["item_id"]].append((r["template"], d))
        elif k == "probe_single":
            ps[(r["source"], r["block"], r["descriptor"], r["image_id"])].append(d)
        elif k == "probe_ctx":
            pc[(r["pair_id"], r["orig_slot"])].append(d)
        elif k == "rel_same":
            same[r["item_id"]].append((r["template"], d))
    return dict(rel=rel, none_rel=none_rel, text=text, ps=ps, pc=pc, same=same)


def by_template(entries: List[Tuple[int, Dict]]) -> Dict[int, Dict]:
    out = defaultdict(list)
    for t, d in entries:
        out[t].append(d)
    return {t: avg_dist(ds) for t, ds in out.items()}


# --------------------------------------------------------------------------- per item
def per_item(pairs: List[Dict], items: List[Dict], ix: Dict, stable_thr: float = 0.9) -> List[Dict]:
    pair_of = {p["pair_id"]: p for p in pairs}
    out = []
    for it in items:
        p = pair_of[it["pair_id"]]
        blk, pred = it["block"], it["predicate"]
        dom = logic.BLOCK_DOMAIN[blk]
        v1, v2 = it["v1"], it["v2"]
        # readings ---------------------------------------------------------
        s_dist = [avg_dist(ix["ps"].get((p["source"], blk, p["descriptor"], img["image_id"]), []))
                  for img in (p["img1"], p["img2"])]
        c_raw = [ix["pc"].get((p["pair_id"], k), []) for k in (1, 2)]
        c_dist = [avg_dist(c) for c in c_raw]
        if None in s_dist or None in c_dist:
            continue
        truth = (v1, v2)
        single_ok = [argmax(s_dist[i]) == truth[i] for i in (0, 1)]
        ctx_ok = [argmax(c_dist[i]) == truth[i] for i in (0, 1)]
        all_var_ok = [all(argmax(d) == truth[i] for d in ix["ps"][(p["source"], blk, p["descriptor"], img["image_id"])])
                      and all(argmax(d) == truth[i] for d in c_raw[i])
                      for i, img in enumerate((p["img1"], p["img2"]))]
        conf = [float(np.mean([d[truth[i]] for d in ix["ps"][(p["source"], blk, p["descriptor"], img["image_id"])]]
                              + [d[truth[i]] for d in c_raw[i]])) for i, img in enumerate((p["img1"], p["img2"]))]
        stable = all(all_var_ok) and min(conf) >= stable_thr
        q = [{v: c_dist[i].get(v, 0.0) for v in dom} for i in (0, 1)]
        # relation answers (binary, both visible) ---------------------------
        prior = by_template(ix["none_rel"].get((p["source"], blk, p["descriptor"], pred, "bin"), []))
        lg = lambda d: math.log(max(d["yes"], EPS)) - math.log(max(d["no"], EPS))
        obs = {}
        for order in ("normal", "swap"):
            tv = by_template(ix["rel"].get((it["item_id"], "bin", "both", order), []))
            if not tv:
                continue
            a1, a2 = logic.presented(v1, v2, order)
            ans = logic.answer(pred, a1, a2)
            qa, qb = (q[0], q[1]) if order == "normal" else (q[1], q[0])
            E = logic.ideal_integrator_accuracy(pred, ans, qa, qb)
            r1, r2 = (argmax(c_dist[0]), argmax(c_dist[1])) if order == "normal" else (argmax(c_dist[1]), argmax(c_dist[0]))
            own = logic.answer(pred, r1, r2)
            exp_incons = 1.0 - logic.ideal_integrator_accuracy(pred, own, qa, qb)
            accs, incons, slog, cal = [], [], [], []
            for t, d in tv.items():
                yes = d["yes"] > 0.5
                accs.append(float(yes == ans))
                incons.append(float(yes != own))
                slog.append(lg(d) if ans else -lg(d))          # log-odds toward the correct answer
                if t in prior:                                  # content-free calibration (section 7.4)
                    cal.append(float((lg(d) - lg(prior[t]) > 0) == ans))
            s1o, s2o = logic.sufficiency(pred, a1, a2, dom)
            obs[order] = dict(acc=float(np.mean(accs)), E=E, d=float(np.mean(accs)) - E, cell=(a1, a2),
                              type=logic.suff_type(s1o, s2o), incons=float(np.mean(incons)),
                              exp_incons=exp_incons, slogit=float(np.mean(slog)),
                              cal=float(np.mean(cal)) if cal else None,
                              cal_d=(float(np.mean(cal)) - E) if cal else None, answer=ans)
        if "normal" not in obs:
            continue
        cal = [o["cal"] for o in obs.values() if o["cal"] is not None]
        # text control ------------------------------------------------------
        tt = by_template(ix["text"].get(it["item_id"], []))
        text_acc = float(np.mean([float((d["yes"] > 0.5) == it["answer"]) for d in tt.values()])) if tt else None
        # ternary partial views ---------------------------------------------
        tern = {}
        for vis in ("only1", "only2", "both"):
            tv = by_template(ix["rel"].get((it["item_id"], "ter", vis, "normal"), []))
            if tv:
                target = logic.partial_target(pred, v1, v2, dom, vis)
                d = avg_dist(list(tv.values()))
                tern[vis] = dict(pred=argmax(d), target=target, p_undet=d["undetermined"])
        # binary partial views (subsample) for the empirical interaction -----
        delta_hat = None
        parts = {vis: by_template(ix["rel"].get((it["item_id"], "bin", vis, "normal"), [])) for vis in ("only1", "only2")}
        if all(parts.values()) and prior:
            full = by_template(ix["rel"].get((it["item_id"], "bin", "both", "normal"), []))
            vals = []
            for t in full:
                if t in parts["only1"] and t in parts["only2"] and t in prior:
                    vals.append(0.5 * (lg(full[t]) - lg(parts["only1"][t]) - lg(parts["only2"][t]) + lg(prior[t])))
            delta_hat = float(np.mean(vals)) if vals else None
        pipe = float(logic.answer(pred, argmax(s_dist[0]), argmax(s_dist[1])) == it["answer"])
        out.append(dict(item_id=it["item_id"], pair_id=it["pair_id"], source=it["source"], block=blk,
                        split=it["split"], descriptor=it["descriptor"], predicate=pred, v1=v1, v2=v2,
                        answer=it["answer"], type=it["type"], n_logic=1 - int(it["s1"]) - int(it["s2"]),
                        cluster1=p["img1"].get("cluster"), cluster2=p["img2"].get("cluster"),
                        package1=p["img1"].get("package"), package2=p["img2"].get("package"),
                        single_ok=single_ok, ctx_ok=ctx_ok, AC=all(single_ok), stable=stable,
                        stability=min(conf), obs=obs, cal_acc=float(np.mean(cal)) if cal else None,
                        text_acc=text_acc, tern=tern, delta_hat=delta_hat, pipeline_acc=pipe))
    return out


# --------------------------------------------------------------------------- contrasts
def _obs_d(row, order, field="d"):
    o = row["obs"].get(order)
    return None if o is None else o[field]


def contrast_units(rows: List[Dict], test: str, field: str = "d") -> List[Tuple[str, float]]:
    """Per-pair contrast values (pair_id, value) for T1-T4, averaged over presentation orders."""
    by_pair = defaultdict(dict)
    for r in rows:
        by_pair[r["pair_id"]][r["predicate"]] = r
    units = []
    for pid, preds in by_pair.items():
        vals = []
        for order in ("normal", "swap"):
            if field == "text" and order == "swap":   # the text control exists in the original order only
                continue
            any_row = next(iter(preds.values()))
            a1, a2 = logic.presented(any_row["v1"], any_row["v2"], order)
            pair = _test_pair(test, a1, a2)
            if pair is None:
                continue
            hi, lo = pair
            if hi not in preds or lo not in preds:
                continue
            if field == "text":
                x, y = preds[hi]["text_acc"], preds[lo]["text_acc"]
            else:
                x, y = _obs_d(preds[hi], order, field), _obs_d(preds[lo], order, field)
            if x is None or y is None:
                continue
            vals.append(x - y)
        if vals:
            units.append((pid, float(np.mean(vals))))
    return units


def _test_pair(test: str, a1: int, a2: int):
    """(non-synergistic predicate, synergistic predicate) for a presented cell, or None."""
    if test == "T1":
        return {(1, 1): ("OR", "AND"), (0, 0): ("AND", "OR")}.get((a1, a2))
    if test == "T2":
        return ("OR", "XOR") if a1 != a2 else None
    if test == "T3":
        if a1 > a2:
            return ("GE", "GT")
        if a1 < a2:
            return ("GT", "GE")
        return None
    if test == "T4":
        return ("GE", "EQ") if (a1, a2) in ((0, 0), (2, 2)) else None
    raise ValueError(test)


def bootstrap(units: List[Tuple[str, float]], n_boot: int = 2000, seed: int = 0,
              cluster_of: Optional[Dict[str, str]] = None):
    """Cluster bootstrap over pairs (or over clusters if cluster_of is given)."""
    if not units:
        return dict(n=0, est=None, lo=None, hi=None, p=None, boots=None)
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for pid, v in units:
        groups[cluster_of.get(pid, pid) if cluster_of else pid].append(v)
    keys = list(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    cnts = np.array([len(groups[k]) for k in keys])
    est = sums.sum() / cnts.sum()
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return dict(n=int(cnts.sum()), n_clusters=len(keys), est=float(est), p=float(min(1.0, p)), boots=boots)


def ci(boots: np.ndarray, alpha: float):
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, float]:
    """Per-test alpha after Holm ordering (used for the CI level)."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out = {}
    for i, k in enumerate(order):
        out[k] = alpha / (m - i)
    return out


def dose_curve(rows: List[Dict], test: str, n_bins: int = 5) -> List[Optional[float]]:
    stab = {r["pair_id"]: r["stability"] for r in rows}
    units = contrast_units(rows, test)
    if len(units) < n_bins:
        return [None] * n_bins
    vals = np.array([stab[pid] for pid, _ in units])
    edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = [v for (pid, v), s in zip(units, vals) if (s >= lo and (s <= hi if b == n_bins - 1 else s < hi))]
        out.append(float(np.mean(sel)) if sel else None)
    return out


# --------------------------------------------------------------------------- summaries
def subset(rows: List[Dict], name: str) -> List[Dict]:
    rows = [r for r in rows if r["split"] == "main"]
    if name == "all":
        return rows
    if name == "AC":
        keep = {r["pair_id"] for r in rows if r["AC"]}
    elif name == "stable":
        keep = {r["pair_id"] for r in rows if r["stable"]}
    else:
        raise ValueError(name)
    return [r for r in rows if r["pair_id"] in keep]


def attribution_error(rows: List[Dict]) -> Dict:
    """M2: in-context probe error on heterogeneous minus homogeneous pairs (pair-level bootstrap)."""
    seen = {}
    for r in rows:
        if r["split"] == "main":
            seen[r["pair_id"]] = r
    units = []
    for pid, r in seen.items():
        err = 1 - float(np.mean(r["ctx_ok"]))
        units.append((pid, err, r["v1"] != r["v2"]))
    het = [(p, e) for p, e, h in units if h]
    hom = [(p, e) for p, e, h in units if not h]
    if not het or not hom:
        return dict(est=None)
    rng = np.random.default_rng(1)
    a, b = np.array([e for _, e in het]), np.array([e for _, e in hom])
    boots = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)])
    lo, hi = ci(boots, 0.05)
    return dict(est=float(a.mean() - b.mean()), lo=lo, hi=hi, n_het=len(a), n_hom=len(b),
                single_acc=float(np.mean([np.mean(r["single_ok"]) for r in seen.values()])),
                ctx_acc=float(np.mean([np.mean(r["ctx_ok"]) for r in seen.values()])))


def sufficiency_discrimination(rows: List[Dict]) -> Dict:
    """M6: balanced accuracy on ternary partial views; over-abstention on full inputs."""
    det, und, full = [], [], []
    for r in rows:
        if r["split"] != "main":
            continue
        for vis in ("only1", "only2"):
            t = r["tern"].get(vis)
            if not t:
                continue
            if t["target"] == "undetermined":
                und.append(float(t["pred"] == "undetermined"))
            else:
                det.append(float(t["pred"] == t["target"]))
        if "both" in r["tern"]:
            full.append(float(r["tern"]["both"]["pred"] == "undetermined"))
    if not det or not und:
        return dict(ba=None)
    rng = np.random.default_rng(2)
    a, b = np.array(det), np.array(und)
    boots = np.array([0.5 * (rng.choice(a, len(a)).mean() + rng.choice(b, len(b)).mean()) for _ in range(2000)])
    lo, hi = ci(boots, 0.05)
    return dict(ba=float(0.5 * (a.mean() + b.mean())), lo=lo, hi=hi, det_acc=float(a.mean()),
                undet_choose_C=float(b.mean()), over_abstain_full=float(np.mean(full)) if full else None,
                n_det=len(a), n_und=len(b))


def type_table(rows: List[Dict]) -> List[Dict]:
    """M3 / M4 / M5 descriptive table by type x answer x predicate (normal order)."""
    g = defaultdict(list)
    for r in rows:
        o = r["obs"]["normal"]
        g[(r["predicate"], o["type"], o["answer"])].append(r)
    out = []
    for (pred, typ, ans), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        out.append(dict(predicate=pred, type=typ, answer=ans, n=len(rs),
                        acc=float(np.mean([r["obs"]["normal"]["acc"] for r in rs])),
                        null_E=float(np.mean([r["obs"]["normal"]["E"] for r in rs])),
                        excess_d=float(np.mean([r["obs"]["normal"]["d"] for r in rs])),
                        text_acc=_mean([r["text_acc"] for r in rs]),
                        cal_acc=_mean([r["cal_acc"] for r in rs]),
                        incons=float(np.mean([r["obs"]["normal"]["incons"] for r in rs])),
                        exp_incons=float(np.mean([r["obs"]["normal"]["exp_incons"] for r in rs])),
                        pipeline_acc=float(np.mean([r["pipeline_acc"] for r in rs])),
                        delta_hat=_mean([r["delta_hat"] for r in rs])))
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def position_bias(rows: List[Dict]) -> Dict:
    u1 = [r["obs"]["normal"]["d"] for r in rows if r["obs"]["normal"]["type"] == "unique1"]
    u2 = [r["obs"]["normal"]["d"] for r in rows if r["obs"]["normal"]["type"] == "unique2"]
    cons = []
    for r in rows:
        if r["predicate"] in logic.SYMMETRIC and "swap" in r["obs"]:
            n, s = r["obs"]["normal"], r["obs"]["swap"]
            cons.append(float((n["acc"] >= 0.5) == (s["acc"] >= 0.5)))
    return dict(unique1_minus_unique2=_mean(u1) - _mean(u2) if u1 and u2 else None,
                swap_consistency=_mean(cons))
