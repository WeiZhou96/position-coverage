"""Analyses after the E3 verdicts (report 10.13). Nothing here changes a pre-registered verdict.

1. H9, S1 part (completes the pre-registered operationalisation). The preregistered analysis plan defines H9 on the COCO part of
   the S1 evaluation set and of the E3 grid; the frozen scripts/e3_analyze.py uses only the E3 COCO groups. Here the
   same contrasts (Aclevr - A, direct; Dselfclevr - Dself, stated) on the S1 COCO items: cell-balanced accuracy,
   mean over the COCO groups of ev_main and ev_3img (held-out templates reported separately); cluster bootstrap and
   seed as in e1_metrics.py; arm = mean of seeds.
2. Stated-protocol audit on the E3 grid, by n (post hoc, descriptive): explicit 'Answer:' rate, unparsed answers,
   completeness of the written per-image facts, fact accuracy against the truth and agreement with the run's own
   isolated single-image reading, isolated-reading accuracy on the same images, and whether the answer follows the
   written facts.
Output: outputs/e3/posthoc/{posthoc.json, posthoc.md}. Usage: python scripts/e3_posthoc.py
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_metrics import SEED, boot_weights, cluster_ids, load_req, summ  # noqa: E402
from s1_analyze import G3, HO_G, ID_G, ROOT, RUNS, TM_G  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "e3", "posthoc")
H9_PAIRS = [("Aclevr", "A", "direct"), ("Dselfclevr", "Dself", "stated")]
COMP = {
    "S1_COCO": [("ev_main", "coco", p) for s, p in list(ID_G) + list(HO_G) if s == "coco"]
               + [("ev_3img", "coco", p) for s, p in G3 if s == "coco"],
    "S1_COCO_2img": [("ev_main", "coco", p) for s, p in list(ID_G) + list(HO_G) if s == "coco"],
    "S1_COCO_3img": [("ev_3img", "coco", p) for s, p in G3 if s == "coco"],
    "S1_COCO_tmpl": [("ev_tmpl", "coco", p) for s, p in TM_G if s == "coco"],
}
MODE = {"direct": "logit", "stated": "gen"}
AUDIT_ARMS = ["base", "A", "B", "C3", "D", "Dself", "A23", "Dself23", "Dself23p"]
OPS = {"ALL": lambda v: all(v), "ANY": lambda v: any(v), "ONE": lambda v: sum(v) == 1,
       "NONE": lambda v: sum(v) == 0, "ATLEAST2": lambda v: sum(v) >= 2}
LINE = re.compile(r"^[\s\-\*#>\d\.\)]*Image\s+(\d+)[\*\s]*(:)?[\*\s]*(.*)$", re.I)
NEG = re.compile(r"\b(not|no|none|without|neither|nor)\b|n't", re.I)


def runs_of(arm):
    return ["base"] if arm == "base" else [f"{arm}_s1", f"{arm}_s2"]


# ----------------------------------------------------------------------------------------------- H9, S1 part
def h9_s1():
    req = load_req()
    cid = cluster_ids(req)
    clusters = sorted(set(cid.values()))
    cidx = {c: i for i, c in enumerate(clusters)}
    W = boot_weights([(c[0], c[1]) for c in clusters], 1000, np.random.default_rng(SEED))
    sets = {k[0] for spec in COMP.values() for k in spec}
    kidx = {}
    for rid, r in sorted(req.items()):
        if r["set"] in sets and r["source"] == "coco":
            kidx.setdefault((r["set"], r["source"], r["predicate"], tuple(r["vals"])), len(kidx))
    C, K = len(clusters), len(kidx)

    def comps(run, proto):
        mode = MODE[proto]
        out = {o["rid"]: o for o in iter_jsonl(os.path.join(RUNS, run, f"eval_{mode}.jsonl"))}
        S, N = np.zeros((C, K)), np.zeros((C, K))
        for rid, r in req.items():
            if r["mode"] != mode or r["source"] != "coco" or r["set"] not in sets:
                continue
            o = out[rid]
            pred = (o["probs"][0] > 0.5) if mode == "logit" else o.get("pred")
            k = kidx[(r["set"], r["source"], r["predicate"], tuple(r["vals"]))]
            S[cidx[cid[rid]], k] += float(pred is not None and pred == r["answer"])
            N[cidx[cid[rid]], k] += 1
        cells = (W @ S) / (W @ N)
        res = {}
        for name, spec in COMP.items():
            vals = []
            for st_, src, p in spec:
                ks = [i for k, i in kidx.items() if k[0] == st_ and k[1] == src and k[2] == p]
                assert ks, (name, st_, p)
                vals.append(np.nanmean(cells[:, ks], axis=1))
            res[name] = np.mean(vals, axis=0)
        return res

    out = {}
    for x, y, proto in H9_PAIRS:
        vx = [comps(r, proto) for r in runs_of(x)]
        vy = [comps(r, proto) for r in runs_of(y)]
        row = {}
        for name in COMP:
            ax, ay = np.mean([v[name] for v in vx], 0), np.mean([v[name] for v in vy], 0)
            row[name] = dict(x=summ(ax), y=summ(ay), diff=summ(ax - ay),
                             seeds_x=[float(v[name][0]) for v in vx], seeds_y=[float(v[name][0]) for v in vy],
                             within_3pp=bool(abs(float(ax[0] - ay[0])) <= 0.03))
        out[f"{x} - {y} ({proto})"] = row
    return out


# ----------------------------------------------------------------------------------------- stated audit on E3
def parse_facts(text, n):
    body = text.split("Answer:")[0]
    facts = {}
    for line in body.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        i = int(m.group(1))
        if not 1 <= i <= n or i in facts:
            continue
        rest = m.group(3).strip()
        first = re.split(r"[.;]", rest, maxsplit=1)[0]
        word = re.match(r"[A-Za-z]+", first)
        if m.group(2) and word and word.group(0).lower() in ("yes", "no"):
            facts[i] = int(word.group(0).lower() == "yes")
        elif first:
            facts[i] = 0 if NEG.search(first) else 1
    return facts


def audit():
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        r.pop("segments", None)
        req[r["rid"]] = r
    res = {}
    for arm in AUDIT_ARMS:
        per_run = []
        for run in runs_of(arm):
            d = os.path.join(RUNS, run)
            iso = {}
            for o in iter_jsonl(os.path.join(d, "eval_e3_logit.jsonl")):
                r = req[o["rid"]]
                if r["set"] == "e3_single":
                    p1 = sum(p for sem, p in zip(r["sems"], o["probs"]) if int(sem) == 1)   # as e3_analyze.py
                    iso[(r["source"], r["descriptor"], r["image_id"])] = int(p1 > 0.5)
            acc = defaultdict(lambda: defaultdict(list))
            for o in iter_jsonl(os.path.join(d, "eval_e3_gen.jsonl")):
                r = req[o["rid"]]
                if r["set"] != "e3_grid":
                    continue
                n, pred = r["n"], o.get("pred")
                a = acc[n]
                a["acc"].append(float(pred is not None and pred == r["answer"]))
                a["explicit"].append(float(bool(o.get("explicit"))))
                a["unparsed"].append(float(pred is None))
                facts = parse_facts(o["text"], n)
                complete = len(facts) == n
                a["complete"].append(float(complete))
                isov = [iso[(r["source"], r["descriptor"], i)] for i in r["image_ids"]]
                for i, f in facts.items():
                    a["fact_truth"].append(float(f == r["vals"][i - 1]))
                    a["fact_iso"].append(float(f == isov[i - 1]))
                a["iso_truth"].extend(float(v == t) for v, t in zip(isov, r["vals"]))
                if complete:
                    v = [facts[i] for i in range(1, n + 1)]
                    a["facts_all_true"].append(float(v == list(r["vals"])))
                    if pred is not None:
                        a["follows"].append(float(OPS[r["predicate"]](v) == pred))
            per_run.append({n: {k: float(np.mean(v)) for k, v in a.items()} for n, a in acc.items()})
        res[arm] = {n: {k: float(np.mean([pr[n][k] for pr in per_run if k in pr[n]]))
                        for k in per_run[0][n]} for n in sorted(per_run[0])}
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    h9 = h9_s1()
    au = audit()
    with open(os.path.join(OUT, "posthoc.json"), "w") as f:
        json.dump(dict(h9_s1=h9, stated_audit=au), f, indent=1)

    def f3(s):
        return f"{s[0]:+.3f} [{s[1]:+.3f}, {s[2]:+.3f}]"

    L = ["# E3 post-hoc analyses", "", "## H9, S1 part (COCO items of the S1 evaluation set; x - y, seed means, 95% CI)",
         "", "| contrast | component | x | y | x - y | within 3 pp |", "|---|---|---|---|---|---|"]
    for lab, row in h9.items():
        for name, v in row.items():
            L.append(f"| {lab} | {name} | {v['x'][0]:.3f} | {v['y'][0]:.3f} | {f3(v['diff'])} | "
                     f"{'yes' if v['within_3pp'] else 'no'} |")
    cols = ["acc", "explicit", "unparsed", "complete", "fact_truth", "iso_truth", "fact_iso", "facts_all_true",
            "follows"]
    L += ["", "## Stated protocol on the E3 grid, by n (arm = mean of seeds)", "",
          "| arm | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for arm, byn in au.items():
        for n, v in byn.items():
            L.append(f"| {arm} | {n} | " + " | ".join(f"{v.get(c, float('nan')):.3f}" for c in cols) + " |")
    with open(os.path.join(OUT, "posthoc.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    with np.errstate(invalid="ignore", divide="ignore"):
        main()
