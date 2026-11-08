"""End-to-end check of the analysis on simulated models.

1. An ideal integrator with noisy (but probe-calibrated) readings must show a positive raw
   contrast (the artefact of design v2 section 2.3) and a near-zero excess contrast d.
2. A model that ignores Image 2 on 30 % of synergistic questions must show a positive d contrast.
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e0lib import analysis as A, items as I, logic  # noqa: E402


def fake_atoms(n_img=6000, seed=0):
    rng = random.Random(seed)
    return [dict(source="sim", image_id=f"im{i}", path=f"im{i}.png", descriptor="thing", block="presence",
                 value=rng.random() < 0.5, cluster=f"im{i}") for i in range(n_img)]


def simulate(skip_rate=0.0, seed=1, n_main=300):
    atoms = fake_atoms()
    for a in atoms:
        a["value"] = int(a["value"])
    pairs, _ = I.build_pairs(atoms, "sim", "presence", n_main, 0, {"thing": {"singular": "a thing", "plural": "things"}})
    items = I.build_items(pairs)
    reqs = I.build_requests(pairs, items, {"sim": "/x"})
    rng = np.random.default_rng(seed)
    val = {a["image_id"]: a["value"] for a in atoms}
    noise = {a["image_id"]: float(rng.uniform(0.0, 0.12)) for a in atoms}
    pair_of = {p["pair_id"]: p for p in pairs}
    item_of = {i["item_id"]: i for i in items}
    results = {}
    for r in reqs:
        k = r["kind"]
        if k == "probe_single":
            v, e = val[r["image_id"]], noise[r["image_id"]]
            p_yes = 1 - e if v == 1 else e
            probs = [p_yes, 1 - p_yes]
        elif k == "probe_ctx":
            p = pair_of[r["pair_id"]]
            img = p["img1"] if r["orig_slot"] == 1 else p["img2"]
            v, e = val[img["image_id"]], noise[img["image_id"]]
            p_yes = 1 - e if v == 1 else e
            probs = [p_yes, 1 - p_yes]
        elif k == "text":
            it = item_of[r["item_id"]]
            probs = [0.99, 0.01] if it["answer"] else [0.01, 0.99]
        elif k == "rel" and r.get("item_id") and r["fmt"] == "bin" and r["visible"] == "both":
            it = item_of[r["item_id"]]
            p = pair_of[it["pair_id"]]
            ids = (p["img1"]["image_id"], p["img2"]["image_id"])
            if r["order"] == "swap":
                ids = ids[::-1]
            reads = [val[i] if rng.random() > noise[i] else 1 - val[i] for i in ids]
            if skip_rate and it["type"] == "synergistic" and rng.random() < skip_rate:
                reads[1] = int(rng.random() < 0.5)          # guesses instead of reading image 2
            yes = logic.answer(it["predicate"], *reads)
            probs = [0.99, 0.01] if yes else [0.01, 0.99]
        else:
            probs = [1 / len(r["cands"])] * len(r["cands"])
        results[r["rid"]] = dict(rid=r["rid"], probs=probs, logp=[float(np.log(x)) for x in probs], mass=1.0)
    rows = A.per_item(pairs, items, A.index(reqs, results), stable_thr=0.0)
    out = {}
    for field in ("acc", "d"):
        for t in ("T1", "T2"):
            out[(field, t)] = A.bootstrap(A.contrast_units(A.subset(rows, "all"), t, field))
    return out


def test_null_removes_reading_noise_artefact():
    res = simulate()
    raw, exc = res[("acc", "T1")], res[("d", "T1")]
    assert raw["est"] > 0.03, raw["est"]                      # artefact present in raw accuracy
    assert abs(exc["est"]) < 0.02 and exc["p"] > 0.01, exc     # removed by the null model


def test_detects_true_integration_deficit():
    res = simulate(skip_rate=0.3)
    assert res[("d", "T1")]["est"] > 0.05 and res[("d", "T1")]["p"] < 0.01, res[("d", "T1")]


if __name__ == "__main__":
    r0 = simulate()
    print({k: (round(v["est"], 4), round(v["p"], 4)) for k, v in r0.items()})
    r1 = simulate(skip_rate=0.3)
    print({k: (round(v["est"], 4), round(v["p"], 4)) for k, v in r1.items()})
    test_null_removes_reading_noise_artefact()
    test_detects_true_integration_deficit()
    print("ok")
