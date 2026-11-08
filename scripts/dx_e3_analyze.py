"""Analysis of the diagnostic probes built by dx_e3_build.py (direction re-diagnosis; post hoc, descriptive).

For each run with eval_e3dx_logit.jsonl:
  reading   accuracy of the in-context reading ("Does Image k contain X?" with all n images shown) against the
            truth, next to the isolated reading of the same image (e3_single), by n, by slot, odd vs other images.
  decomp    on the arrangements with one odd image (patterns A1, B1; every rotation; every operator):
              E_iso   ideal integration from the isolated readings (as in E3)
              E_ctx   ideal integration from the in-context readings of the same arrangement
              direct  E3 single-pass accuracy on the same items; stated  E3 state-then-answer accuracy
            E_iso - E_ctx = loss from reading in context; E_ctx - direct = loss in single-pass aggregation.
  text      text-only facts (no images): k-balanced accuracy by n and operator (as M_n in e3_analyze.py).
Output: outputs/e3dx/{dx.json, dx.md}. Usage: python scripts/dx_e3_analyze.py [runs ...]
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
OUT = os.path.join(ROOT, "outputs", "e3dx")
DEFAULT = ["base", "A_s1", "A23_s1", "D_s1", "Dself_s1", "Dself23p_s1"]


def load(path):
    return {o["rid"]: o for o in iter_jsonl(path)} if os.path.exists(path) else {}


def mean(x):
    return float(np.mean(x)) if len(x) else float("nan")


def main():
    runs = sys.argv[1:] or DEFAULT
    e3req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        r.pop("segments", None)
        e3req[r["rid"]] = r
    dxreq = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3dx", "eval_requests.jsonl")):
        r.pop("segments", None)
        dxreq[r["rid"]] = r
    grid_by_arr = defaultdict(list)          # (group, n, pattern, rot) -> grid logit rids
    gen_rid = {}
    for rid, r in e3req.items():
        if r["set"] == "e3_grid":
            key = (r["group"], r["n"], r["pattern"], r["rot"], r["predicate"])
            if r["mode"] == "logit":
                grid_by_arr[key[:4]].append(rid)
            else:
                gen_rid[key] = rid
    res = {}
    for run in runs:
        d = os.path.join(RUNS, run)
        dx = load(os.path.join(d, "eval_e3dx_logit.jsonl"))
        if not dx:
            print("no probe output for", run)
            continue
        lg, gn = load(os.path.join(d, "eval_e3_logit.jsonl")), load(os.path.join(d, "eval_e3_gen.jsonl"))
        iso = {}
        for rid, o in lg.items():
            r = e3req[rid]
            if r["set"] == "e3_single":
                p1 = sum(p for s, p in zip(r["sems"], o["probs"]) if int(s) == 1)
                iso[(r["source"], r["descriptor"], r["image_id"])] = p1
        ctx = {}
        rd = defaultdict(lambda: defaultdict(list))
        for rid, o in dx.items():
            r = dxreq[rid]
            if r["set"] != "dx_ctx":
                continue
            p = o["probs"][0]
            ctx[(r["group"], r["n"], r["pattern"], r["rot"], r["slot"])] = p
            odd = (r["pattern"] == "A1" and r["value"] == 1) or (r["pattern"] == "B1" and r["value"] == 0)
            pi = iso[(r["source"], r["descriptor"], r["image_id"])]
            c_ctx, c_iso = float((p > 0.5) == bool(r["value"])), float((pi > 0.5) == bool(r["value"]))
            for key in (("n", r["n"]), ("n_odd" if odd else "n_other", r["n"])):
                rd[key]["ctx"].append(c_ctx)
                rd[key]["iso"].append(c_iso)
            if r["n"] == 6:
                rd[("slot6_odd" if odd else "slot6_other", r["slot"])]["ctx"].append(c_ctx)
        reading = {f"{k[0]}={k[1]}": {m: mean(v) for m, v in vv.items()} for k, vv in sorted(rd.items())}
        # decomposition on A1/B1 arrangements
        dec = defaultdict(lambda: defaultdict(list))
        for (g, n, pat, rot), rids in grid_by_arr.items():
            if pat not in ("A1", "B1"):
                continue
            qs_ctx = [{1: ctx[(g, n, pat, rot, k)], 0: 1 - ctx[(g, n, pat, rot, k)]} for k in range(1, n + 1)]
            for rid in rids:
                r = e3req[rid]
                qs_iso = [{1: iso[(r["source"], r["descriptor"], i)], 0: 1 - iso[(r["source"], r["descriptor"], i)]}
                          for i in r["image_ids"]]
                op, ans = r["predicate"], r["answer"]
                cells = dict(E_iso=e3.ideal(op, ans, qs_iso), E_ctx=e3.ideal(op, ans, qs_ctx),
                             direct=float((lg[rid]["probs"][0] > 0.5) == ans))
                gr = gen_rid.get((g, n, pat, rot, op))
                if gr in gn:
                    cells["stated"] = float(gn[gr].get("pred") is not None and gn[gr]["pred"] == ans)
                odd_slot = r["vals"].index(1 if pat == "A1" else 0) + 1
                for key in (("n", n), ("n_op", n, op)) + ((("odd6", op, odd_slot),) if n == 6 else ()):
                    for m, v in cells.items():
                        dec[key][m].append(v)
        decomp = {"|".join(map(str, k)): {m: mean(v) for m, v in vv.items()} for k, vv in sorted(dec.items(),
                                                                                                key=str)}
        # text-only, k-balanced
        tx = defaultdict(lambda: defaultdict(list))
        for rid, o in dx.items():
            r = dxreq[rid]
            if r["set"] == "dx_text":
                tx[(r["n"], r["predicate"])][sum(r["vals"])].append(float((o["probs"][0] > 0.5) == r["answer"]))
        text = {}
        for (n, op), byk in sorted(tx.items()):
            text[f"{n}|{op}"] = mean([mean(v) for v in byk.values()])
        for n in range(2, 7):
            text[f"{n}|M"] = mean([text[f"{n}|{op}"] for op in e3.ops_for(n)])
        res[run] = dict(reading=reading, decomp=decomp, text=text)
        print(run, "done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "dx.json"), "w") as f:
        json.dump(res, f, indent=1)
    L = ["# E3 diagnostic probes (post hoc; single seed per arm)", "",
         "## Reading accuracy: in context vs isolated (all images / odd image / other images)", "",
         "| run | n | ctx all | iso all | ctx odd | iso odd | ctx other | iso other |", "|---|---|---|---|---|---|---|---|"]
    for run, v in res.items():
        R = v["reading"]
        for n in range(2, 7):
            a, o, t = R.get(f"n={n}", {}), R.get(f"n_odd={n}", {}), R.get(f"n_other={n}", {})
            L.append(f"| {run} | {n} | {a.get('ctx', np.nan):.3f} | {a.get('iso', np.nan):.3f} | "
                     f"{o.get('ctx', np.nan):.3f} | {o.get('iso', np.nan):.3f} | {t.get('ctx', np.nan):.3f} | "
                     f"{t.get('iso', np.nan):.3f} |")
    L += ["", "## In-context reading of the odd image at n = 6, by slot", "", "| run | slot 1 | 2 | 3 | 4 | 5 | 6 |",
          "|---|---|---|---|---|---|---|"]
    for run, v in res.items():
        L.append(f"| {run} | " + " | ".join(f"{v['reading'].get(f'slot6_odd={s}', {}).get('ctx', np.nan):.3f}"
                                            for s in range(1, 7)) + " |")
    L += ["", "## Decomposition on one-odd-image arrangements (mean over operators and items)", "",
          "| run | n | E_iso | E_ctx | direct | stated | reading loss (E_iso-E_ctx) | aggregation loss (E_ctx-direct) |",
          "|---|---|---|---|---|---|---|---|"]
    for run, v in res.items():
        for n in range(2, 7):
            c = v["decomp"].get(f"n|{n}", {})
            L.append(f"| {run} | {n} | {c.get('E_iso', np.nan):.3f} | {c.get('E_ctx', np.nan):.3f} | "
                     f"{c.get('direct', np.nan):.3f} | {c.get('stated', np.nan):.3f} | "
                     f"{c.get('E_iso', np.nan) - c.get('E_ctx', np.nan):+.3f} | "
                     f"{c.get('E_ctx', np.nan) - c.get('direct', np.nan):+.3f} |")
    L += ["", "## n = 6, ALL with the odd (missing) image by slot: E_ctx / direct", "",
          "| run | slot 1 | 2 | 3 | 4 | 5 | 6 |", "|---|---|---|---|---|---|---|"]
    for run, v in res.items():
        cells = []
        for s in range(1, 7):
            c = v["decomp"].get(f"odd6|ALL|{s}", {})
            cells.append(f"{c.get('E_ctx', np.nan):.2f} / {c.get('direct', np.nan):.2f}")
        L.append(f"| {run} | " + " | ".join(cells) + " |")
    L += ["", "## Text-only facts: k-balanced accuracy", "",
          "| run | n | ALL | ANY | ONE | NONE | ATLEAST2 | M |", "|---|---|---|---|---|---|---|---|"]
    for run, v in res.items():
        for n in range(2, 7):
            T = v["text"]
            L.append(f"| {run} | {n} | " + " | ".join(f"{T.get(f'{n}|{op}', np.nan):.3f}" for op in e3.OPS)
                     + f" | {T[f'{n}|M']:.3f} |")
    with open(os.path.join(OUT, "dx.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
