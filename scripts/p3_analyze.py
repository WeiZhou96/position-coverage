"""P3 analysis (the preregistered analysis plan): is the interior missing image represented at the answer position
but not read out, or not gathered at all?

Inputs: outputs/p3/<run>.npz / .meta.jsonl from scripts/p3_extract.py (E3 ALL items, F and B1, n = 4 and 6).
  OUT_int        AUROC of the model's own P(Yes) (from the same extraction pass): F vs B1 with the missing image at an
                 interior slot, n = 4 and 6 pooled
  PROBE_int(l)   out-of-fold AUROC of a linear probe on the layer-l hidden state at the answer position, same items;
                 standardised features, logistic regression (C = 0.1), 6-fold cross-validation grouped by image group
  PROBE_LATE     mean of PROBE_int over the last third of the layers (l > 2L/3); READOUT_GAP = PROBE_LATE - OUT_int
  ATT_RATIO      on F items: attention per image token from the answer position, interior slots / edge slots,
                 averaged over the last-third layers (heads already averaged)
  the same quantities with the missing image at the edge (EDGE) are reported for reference
Output: outputs/p3/p3.json, p3.md. Usage: python scripts/p3_analyze.py [runs ...]
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3 = os.path.join(ROOT, "outputs", "p3")
RUNS = ["base", "A_s1", "A_s2", "A23_s1", "A23_s2", "A3edge_s1", "A3edge_s2"]


def probe_auc(X, y, g):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=6).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    runs = sys.argv[1:] or RUNS
    res = {}
    for run in runs:
        z = np.load(os.path.join(P3, f"{run}.npz"))
        meta = [json.loads(line) for line in open(os.path.join(P3, f"{run}.meta.jsonl"))]
        hid, att, ntok = z["hidden"].astype(np.float32), z["attn"], z["ntok"]
        L = att.shape[1]
        late = [l for l in range(hid.shape[1]) if l > 2 * (hid.shape[1] - 1) / 3]
        late_att = [l for l in range(L) if l >= 2 * L / 3]
        out = {}
        for kind in ("int", "edge"):
            sel, y = [], []
            for i, m in enumerate(meta):
                if m["pattern"] == "F":
                    sel.append(i)
                    y.append(1)
                else:
                    slot = m["vals"].index(0) + 1
                    if (1 < slot < m["n"]) == (kind == "int"):
                        sel.append(i)
                        y.append(0)
            sel, y = np.array(sel), np.array(y)
            g = np.array([meta[i]["group"] for i in sel])
            p = np.array([meta[i]["p_yes"] for i in sel])
            out[f"OUT_{kind}"] = float(roc_auc_score(y, p))
            curve = [probe_auc(hid[sel, l], y, g) for l in range(hid.shape[1])]
            out[f"PROBE_{kind}_by_layer"] = [round(c, 4) for c in curve]
            out[f"PROBE_LATE_{kind}"] = float(np.mean([curve[l] for l in late]))
            out[f"PROBE_MAX_{kind}"] = float(np.max(curve))
            out[f"READOUT_GAP_{kind}"] = out[f"PROBE_LATE_{kind}"] - out[f"OUT_{kind}"]
        ratios = []
        for i, m in enumerate(meta):
            if m["pattern"] != "F":
                continue
            n = m["n"]
            per_tok = att[i][:, :n] / ntok[i][:n]
            share = per_tok / per_tok.sum(1, keepdims=True)
            interior = share[late_att][:, 1:n - 1].mean()
            edge = share[late_att][:, [0, n - 1]].mean()
            ratios.append(interior / edge)
        out["ATT_RATIO"] = float(np.mean(ratios))
        res[run] = out
        print(run, {k: round(v, 3) for k, v in out.items() if isinstance(v, float)}, flush=True)
    with open(os.path.join(P3, "p3.json"), "w") as f:
        json.dump(res, f, indent=1)
    keys = ["OUT_int", "PROBE_LATE_int", "PROBE_MAX_int", "READOUT_GAP_int", "OUT_edge", "PROBE_LATE_edge",
            "READOUT_GAP_edge", "ATT_RATIO"]
    lines = ["# P3 mechanism probe (E3 ALL items, n = 4 and 6)", "", "| run | " + " | ".join(keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for run, out in res.items():
        lines.append(f"| {run} | " + " | ".join("%.3f" % out[k] for k in keys) + " |")
    lines += ["", "## Interior probe AUROC by layer", ""]
    for run, out in res.items():
        lines.append(f"- {run}: " + " ".join("%.2f" % v for v in out["PROBE_int_by_layer"]))
    with open(os.path.join(P3, "p3.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
