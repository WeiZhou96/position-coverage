"""Sealed test set (report 10.5, E1): fresh images reserved for the final confirmatory evaluation.

No image appears in E0, S1 (training or evaluation) or E3 (training or evaluation). CLEVR images come from the
val split (all training uses the CLEVR train split); COCO images come from the S1 evaluation pool (15 % hash split
of train2017 images unused by E0), which no training set draws from. The set is built once, its hashes are recorded
in the report, and it is not evaluated until the final checkpoints are fixed and a pre-registration for it is
written. Later training builds must exclude data/sealed_image_ids.json.

  data/sealed_s1/eval_requests.jsonl   same structure and set names as data/s1 (ev_main, ev_tmpl, ev_text,
                                       ev_3img, ev_single) at half size; the ambiguous S1 NONE wording (template 0)
                                       is replaced by the E3 wording
  data/sealed_e3/eval_requests.jsonl   same structure and set names as data/e3 (e3_grid, e3_single) with 6 groups
                                       per source instead of 12
Both run with the existing evaluator: python scripts/s1_eval.py --run X --data sealed_s1|sealed_e3 ...

Usage: python scripts/sealed_build.py [--extract]
"""
import argparse
import glob
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0lib import e3, items as I, logic, prompts, s1  # noqa: E402
from e0lib.io import in_subsample, iter_jsonl, load_yaml, read_jsonl, stable_id, write_jsonl  # noqa: E402
import e3_build  # noqa: E402
from s1_build import phrases, three_triples  # noqa: E402

SEED = 2026110501
DS1, DE3 = os.path.join(ROOT, "data", "sealed_s1"), os.path.join(ROOT, "data", "sealed_e3")
IMG = os.path.join(ROOT, "data", "sealed_images")
N_PAIR = {"presence": 60, "count": 30}      # pairs per (source, cell); S1 used 120 / 60
N_TMPL = {"presence": 15, "count": 10}      # held-out paraphrase pairs per (source, cell); S1 used 30 / 20
N_TRIP = 25                                 # triples per (source, pattern); S1 used 50
N_GROUPS = 6                                # n-image groups per source; E3 used 12
NONE_T0 = "Is it true that none of the three images contains {X}?"   # = E3 wording for n = 3


def used_images():
    used = defaultdict(set)
    for p in read_jsonl(os.path.join(ROOT, "data/pairs.jsonl")):
        used[p["source"]] |= {p["img1"]["image_id"], p["img2"]["image_id"]}
    files = sorted(glob.glob(os.path.join(ROOT, "data/s1/*.jsonl")) + glob.glob(os.path.join(ROOT, "data/e3/*.jsonl")))
    for f in files:
        for r in iter_jsonl(f):
            src = r.get("source")
            if not src:
                continue
            used[src] |= set(r.get("image_ids") or [])
            if r.get("image_id"):
                used[src].add(r["image_id"])
            for k in ("img1", "img2"):
                if isinstance(r.get(k), dict):
                    used[src].add(r[k]["image_id"])
            for im in (r.get("imgs") or []) + (r.get("P") or []) + (r.get("N") or []):
                used[src].add(im["image_id"])
    return used, [os.path.relpath(f, ROOT) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(os.path.join(ROOT, "configs/sources.yaml"))
    ph = phrases(cfg)
    kept = json.load(open(os.path.join(ROOT, "data/build_report.json")))["clevr:presence_kept"]
    used, scanned = used_images()
    rep = dict(scanned=scanned, used={k: len(v) for k, v in used.items()})
    root = {"clevr": os.path.join(IMG, "clevr"), "coco": os.path.join(IMG, "coco")}
    prefix = {"clevr": "val/", "coco": ""}

    clevr = [a for a in read_jsonl(os.path.join(ROOT, "data/atoms_clevr.jsonl"))
             if not a.get("report") and a["image_id"] not in used["clevr"]
             and (a["block"] == "count" or a["descriptor"] in kept)]
    coco = [a for a in read_jsonl(os.path.join(ROOT, "data/atoms_coco.jsonl"))
            if a.get("split") == "train" and a["block"] == "presence" and not a.get("report")
            and in_subsample(a["image_id"] + "|s1eval", 0.15) and a["image_id"] not in used["coco"]]
    taken = defaultdict(set)

    # ---------------------------------------------------------------- two-image pairs (presence first, then count)
    pairs = []
    for src, blk, atoms, match in (("clevr", "presence", clevr, False), ("coco", "presence", coco, True),
                                   ("clevr", "count", clevr, False)):
        pool = [a for a in atoms if a["block"] == blk and a["image_id"] not in taken[src]]
        pr, short = I.build_pairs(pool, src, blk, N_PAIR[blk], 0, ph, seed=SEED, use_matching=match)
        for p in pr:
            for k in ("img1", "img2"):
                p[k]["path"] = prefix[src] + p[k]["path"]
                taken[src].add(p[k]["image_id"])
        pairs += pr
        rep[f"pairs:{src}:{blk}"] = dict(n=len(pr), shortfall=short)
    by_cell = defaultdict(list)
    for p in pairs:
        by_cell[(p["source"], p["block"], p["v1"], p["v2"])].append(p)
    rng = random.Random(SEED)
    tmpl_pairs = []
    for k, ps in sorted(by_cell.items()):
        ps = sorted(ps, key=lambda p: p["pair_id"])
        rng.shuffle(ps)
        tmpl_pairs += ps[: N_TMPL[k[1]]]

    # ---------------------------------------------------------------- triples and n-image groups
    pres = {"clevr": [a for a in clevr if a["block"] == "presence"], "coco": coco}
    triples = []
    for src, salt in (("clevr", 1), ("coco", 2)):
        tr = three_triples(pres[src], src, set(taken[src]), random.Random(SEED + salt), N_TRIP, prefix[src])
        for t in tr:
            t["trip_id"] = stable_id("sealedtrip", t["trip_id"])
            taken[src] |= {im["image_id"] for im in t["imgs"]}
        triples += tr
    e3_build.N_GROUPS = N_GROUPS
    groups = []
    for src, salt in (("clevr", 3), ("coco", 4)):
        gs = e3_build.eval_groups(pres[src], src, set(taken[src]), random.Random(SEED + salt), prefix[src])
        for g in gs:
            g["group"] = "sealed_" + g["group"]
            taken[src] |= {im["image_id"] for im in g["P"] + g["N"]}
        groups += gs

    # ---------------------------------------------------------------- requests
    def adder(reqs):
        def add(set_, mode, segs, cands, sems, **meta):
            reqs.append(dict(rid=stable_id("sealed", set_, mode, meta, n=20), set=set_, mode=mode, segments=segs,
                             cands=cands, sems=sems, **meta))
        return add

    rs1, re3 = [], []
    add = adder(rs1)

    def path_of(src, im):
        return f"{root[src]}/{im['path']}"

    def rel_requests(set_, p, templates, kind):
        paths = [path_of(p["source"], p[k]) for k in ("img1", "img2")]
        for pred in logic.BLOCK_PREDICATES[p["block"]]:
            for t in templates:
                tt = (prompts.REL_TEMPLATES if kind == "train" else s1.HO_TEMPLATES)[pred][t]
                q = prompts.fill(tt, p["phrase"])
                meta = dict(source=p["source"], block=p["block"], descriptor=p["descriptor"], predicate=pred,
                            vals=[p["v1"], p["v2"]], answer=logic.answer(pred, p["v1"], p["v2"]), template=t,
                            pair_id=p["pair_id"], image_ids=[p["img1"]["image_id"], p["img2"]["image_id"]])
                add(set_, "logit", s1.student_segments(paths, q), ["Yes", "No"], ["yes", "no"], **meta)
                add(set_, "gen", s1.srt_segments(paths, q, p["block"], p["phrase"]), ["Yes", "No"], ["yes", "no"],
                    **meta)

    single = {}
    for p in pairs:
        rel_requests("ev_main", p, (0, 1), "train")
        for pred in logic.BLOCK_PREDICATES[p["block"]]:
            segs, c, sm = prompts.text_control_request(pred, p["phrase"], 0, p["block"], p["v1"], p["v2"])
            add("ev_text", "logit", segs, c, sm, source=p["source"], block=p["block"], descriptor=p["descriptor"],
                predicate=pred, vals=[p["v1"], p["v2"]], answer=logic.answer(pred, p["v1"], p["v2"]), template=0,
                pair_id=p["pair_id"])
        for k, v in (("img1", p["v1"]), ("img2", p["v2"])):
            single[(p["source"], p["block"], p["descriptor"], p[k]["image_id"])] = (path_of(p["source"], p[k]), v,
                                                                                   p["phrase"])
    for p in tmpl_pairs:
        rel_requests("ev_tmpl", p, range(5), "heldout")
    for t in triples:
        phr = ph[t["descriptor"]]
        paths = [path_of(t["source"], im) for im in t["imgs"]]
        for op in s1.THREE_OPS:
            for ti in (0, 1):
                tt = NONE_T0 if (op == "NONE" and ti == 0) else s1.THREE_TEMPLATES[op][ti]
                q = prompts.fill(tt, phr)
                meta = dict(source=t["source"], block="presence", descriptor=t["descriptor"], predicate=op,
                            vals=t["vals"], answer=s1.three_answer(op, t["vals"]), template=ti, trip_id=t["trip_id"],
                            image_ids=[im["image_id"] for im in t["imgs"]])
                add("ev_3img", "logit", s1.student_segments(paths, q), ["Yes", "No"], ["yes", "no"], **meta)
                add("ev_3img", "gen", s1.srt_segments(paths, q, "presence", phr), ["Yes", "No"], ["yes", "no"], **meta)
        for path, im, v in zip(paths, t["imgs"], t["vals"]):
            single[(t["source"], "presence", t["descriptor"], im["image_id"])] = (path, v, phr)
    for (src, blk, desc, iid), (path, v, phr) in sorted(single.items()):
        for perm in ((0,) if blk == "presence" else (0, 1, 2)):
            segs, c, sm = prompts.probe_single_request(blk, phr, 0, path, perm)
            add("ev_single", "logit", segs, c, sm, source=src, block=blk, descriptor=desc, image_id=iid, value=v,
                template=0, perm=perm)

    add = adder(re3)
    for g in groups:
        phr = ph[g["descriptor"]]
        for n in range(2, 7):
            for pname, pat in e3.base_patterns(n).items():
                for r in range(n):
                    idx = e3.arrange(n, r)
                    vals = [pat[i] for i in idx]
                    ims = [g["P"][i] if pat[i] else g["N"][i] for i in idx]
                    paths = [path_of(g["source"], im) for im in ims]
                    for op in e3.ops_for(n):
                        q = e3.question(op, n, phr)
                        meta = dict(source=g["source"], block="presence", descriptor=g["descriptor"], predicate=op,
                                    n=n, vals=vals, answer=e3.answer(op, vals), template=0, group=g["group"],
                                    pattern=pname, rot=r, image_ids=[im["image_id"] for im in ims])
                        add("e3_grid", "logit", s1.student_segments(paths, q), ["Yes", "No"], ["yes", "no"], **meta)
                        add("e3_grid", "gen", s1.srt_segments(paths, q, "presence", phr), ["Yes", "No"],
                            ["yes", "no"], **meta)
        for v, key in ((1, "P"), (0, "N")):
            for im in g[key]:
                segs, c, sm = prompts.probe_single_request("presence", phr, 0, path_of(g["source"], im))
                add("e3_single", "logit", segs, c, sm, source=g["source"], block="presence",
                    descriptor=g["descriptor"], image_id=im["image_id"], value=v, template=0, group=g["group"])

    # ---------------------------------------------------------------- isolation checks
    ids = defaultdict(set)
    for r in rs1 + re3:
        for i in (r.get("image_ids") or [r.get("image_id")]):
            if i:
                ids[r["source"]].add(i)
    for src in ids:
        assert not (ids[src] & used[src]), f"{src}: sealed image already used"
        assert ids[src] == taken[src], f"{src}: request images differ from sampled images"
    assert all(i.startswith("CLEVR_val_") for i in ids["clevr"])
    assert all(in_subsample(i + "|s1eval", 0.15) for i in ids["coco"])
    trip_ids = {s: {im["image_id"] for t in triples if t["source"] == s for im in t["imgs"]} for s in ids}
    grp_ids = {s: {im["image_id"] for g in groups if g["source"] == s for im in g["P"] + g["N"]} for s in ids}
    pair_ids = {s: {p[k]["image_id"] for p in pairs if p["source"] == s for k in ("img1", "img2")} for s in ids}
    for s in ids:
        assert not (trip_ids[s] & grp_ids[s]) and not (trip_ids[s] & pair_ids[s]) and not (grp_ids[s] & pair_ids[s])
    for reqs in (rs1, re3):
        assert len({r["rid"] for r in reqs}) == len(reqs)

    rep["requests_sealed_s1"] = Counter(f"{r['set']}|{r['mode']}" for r in rs1)
    rep["requests_sealed_e3"] = Counter(f"{r['set']}|{r['mode']}" for r in re3)
    rep["images"] = {s: len(v) for s, v in ids.items()}
    rep["triples"] = Counter(f"{t['source']}|{''.join(map(str, t['vals']))}" for t in triples)
    rep["groups"] = [g["group"] + ":" + g["descriptor"] for g in groups]
    os.makedirs(DS1, exist_ok=True)
    os.makedirs(DE3, exist_ok=True)
    write_jsonl(os.path.join(DS1, "pairs.jsonl"), pairs)
    write_jsonl(os.path.join(DS1, "triples.jsonl"), triples)
    write_jsonl(os.path.join(DS1, "eval_requests.jsonl"), rs1)
    write_jsonl(os.path.join(DE3, "groups.jsonl"), groups)
    write_jsonl(os.path.join(DE3, "eval_requests.jsonl"), re3)
    with open(os.path.join(ROOT, "data", "sealed_image_ids.json"), "w") as f:
        json.dump({s: sorted(v) for s, v in ids.items()}, f)
    with open(os.path.join(DS1, "build_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=dict)
    print(json.dumps(rep, indent=1, default=dict))

    if args.extract:
        need = defaultdict(set)
        for r in rs1 + re3:
            for k, v in r["segments"]:
                if k == "image":
                    need[r["source"]].add(v)
        for src, paths in need.items():
            todo = sorted(p for p in paths if not os.path.exists(p))
            print(f"{src}: {len(paths)} images, {len(todo)} to extract", flush=True)
            if not todo:
                continue
            zpath = cfg["clevr"]["zip"] if src == "clevr" else cfg["coco"]["image_zips"]["train"]
            with zipfile.ZipFile(zpath) as z:
                for p in todo:
                    rel = os.path.relpath(p, root[src])
                    member = (cfg["clevr"]["images_prefix"].replace("images/val/", "images/") + rel) \
                        if src == "clevr" else rel
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "wb") as f:
                        f.write(z.read(member))


if __name__ == "__main__":
    main()
