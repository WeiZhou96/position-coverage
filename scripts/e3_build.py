"""E3 data (the preregistered analysis plan): image-count generalisation, position and paired interventions.

Evaluation (data/e3/eval_requests.jsonl), images disjoint from E0, S1 and every E3 training item:
  e3_grid    n = 2..6; per group 6 positive and 6 negative images of one descriptor; base patterns
             Z, A1, A2, B2, B1, F over image indices, each under all n cyclic rotations; ops ALL, ANY, ONE, NONE,
             ATLEAST2 (n > 2); 'logit' (direct answer) and 'gen' (state-then-answer) modes
  e3_single  single-image reading of every evaluation image (ideal-integration reference)
Training items (all with explicit 'question' and 'vals'):
  train_23.jsonl     8,400 stratified S1 two-image items + 3,200 three-image items (ALL/ANY/ONE/NONE)  -> A23, Dself23
  train_all.jsonl    11,600 stratified two-image items over all six S1 operators (incl. XOR, EQ)       -> Aall
  train_clevr.jsonl  the 7,600 S1 CLEVR items + 4,000 new CLEVR presence items (AND/OR)                  -> Aclevr, Dselfclevr

Usage: python scripts/e3_build.py [--extract]
"""
import argparse
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0lib import atoms_clevr, e3, items as I, logic, prompts, s1  # noqa: E402
from e0lib.io import in_subsample, load_yaml, read_jsonl, stable_id, write_jsonl  # noqa: E402
from s1_build import e0_used_images, phrases, relabel  # noqa: E402

SEED = 20261105
D = os.path.join(ROOT, "data", "e3")
IMG = os.path.join(D, "images")
S1D = os.path.join(ROOT, "data", "s1")
N_GROUPS = 12          # evaluation groups per source
N_POS = 6              # positive and negative images per group (n <= 6)
N_TRIPLES = 50         # three-image training triples per (source, pattern)
N_TOTAL = 11600        # training items per arm, as in S1
N_TWO_IN_23 = 8400


def stratified(items, n_keep, key, salt):
    """Proportional allocation over strata (largest remainder); inside a stratum the order is a fixed hash."""
    strata = defaultdict(list)
    for it in items:
        strata[key(it)].append(it)
    total = len(items)
    quota = {k: n_keep * len(v) / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (alloc[k] - quota[k], str(k)))[: n_keep - sum(alloc.values())]:
        alloc[k] += 1
    out = []
    for k, v in sorted(strata.items(), key=lambda kv: str(kv[0])):
        v = sorted(v, key=lambda it: stable_id(salt, it["item_id"]))
        out += v[: alloc[k]]
    return out


def with_question(it):
    it = dict(it)
    it.setdefault("question", prompts.fill(prompts.REL_TEMPLATES[it["predicate"]][it["template"]], it["phrase"]))
    it.setdefault("vals", [it["v1"], it["v2"]])
    return it


def pools_by(atoms, exclude):
    pools = defaultdict(list)
    for a in sorted(atoms, key=lambda a: (a["descriptor"], a["image_id"])):
        if a["image_id"] not in exclude and not a.get("report"):
            pools[(a["descriptor"], a["value"])].append(a)
    return pools


def eval_groups(atoms, source, exclude, rng, prefix):
    pools = pools_by(atoms, exclude)
    for k in pools:
        rng.shuffle(pools[k])
    descs = sorted({d for d, _ in pools})
    taken, groups = set(), []
    for g in range(N_GROUPS):
        desc = descs[g % len(descs)]
        imgs = {}
        for v in (1, 0):
            got = []
            pool = pools[(desc, v)]
            while pool and len(got) < N_POS:
                a = pool.pop()
                if a["image_id"] not in taken:
                    taken.add(a["image_id"])
                    got.append(dict(image_id=a["image_id"], path=prefix + a["path"]))
            assert len(got) == N_POS, f"pool too small: {source} {desc} {v}"
            imgs["P" if v else "N"] = got
        groups.append(dict(group=f"{source}{g:02d}", source=source, descriptor=desc, **imgs))
    return groups


def train_triples(atoms, source, exclude, rng, prefix):
    pools = pools_by(atoms, exclude)
    for k in pools:
        rng.shuffle(pools[k])
    descs = sorted({d for d, _ in pools})
    taken, out = set(), []
    for pat in sorted({(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)}):
        for j in range(N_TRIPLES):
            desc = descs[j % len(descs)]
            imgs = []
            for v in pat:
                pool = pools[(desc, v)]
                while pool and pool[-1]["image_id"] in taken:
                    pool.pop()
                if not pool:
                    break
                a = pool.pop()
                taken.add(a["image_id"])
                imgs.append(a)
            assert len(imgs) == 3, f"triple pool too small: {source} {desc} {pat}"
            out.append(dict(trip_id=stable_id("e3trip", source, desc, [a["image_id"] for a in imgs]), source=source,
                            descriptor=desc, vals=list(pat),
                            imgs=[dict(image_id=a["image_id"], path=prefix + a["path"]) for a in imgs]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="materialise the new images used by E3")
    args = ap.parse_args()
    cfg = load_yaml(os.path.join(ROOT, "configs/sources.yaml"))
    ph = phrases(cfg)
    kept = json.load(open(os.path.join(ROOT, "data/build_report.json")))["clevr:presence_kept"]
    root = {"clevr": os.path.join(IMG, "clevr"), "coco": os.path.join(IMG, "coco")}
    rep = {}

    # ---------------------------------------------------------------- images already used anywhere
    used = e0_used_images()
    s1_items = read_jsonl(os.path.join(S1D, "train_items.jsonl"))
    s1_pairs = read_jsonl(os.path.join(S1D, "train_pairs.jsonl"))
    for it in s1_items:
        used[it["source"]] |= set(it["image_ids"])
    for p in s1_pairs:
        used[p["source"]] |= {p["img1"]["image_id"], p["img2"]["image_id"]}
    s1_eval_ids = defaultdict(set)
    for r in read_jsonl(os.path.join(S1D, "eval_requests.jsonl")):
        s1_eval_ids[r["source"]] |= set(r.get("image_ids") or [r.get("image_id")])
    for src in s1_eval_ids:
        used[src] |= s1_eval_ids[src]

    # ---------------------------------------------------------------- evaluation groups
    clevr_val = [a for a in read_jsonl(os.path.join(ROOT, "data/atoms_clevr.jsonl"))
                 if a["block"] == "presence" and a["descriptor"] in kept]
    coco_all = [a for a in read_jsonl(os.path.join(ROOT, "data/atoms_coco.jsonl"))
                if a.get("split") == "train" and a["block"] == "presence"]
    coco_eval_pool = [a for a in coco_all if in_subsample(a["image_id"] + "|s1eval", 0.15)]
    coco_train_pool = [a for a in coco_all if not in_subsample(a["image_id"] + "|s1eval", 0.15)]
    groups = eval_groups(clevr_val, "clevr", used["clevr"], random.Random(SEED), "val/")
    groups += eval_groups(coco_eval_pool, "coco", used["coco"], random.Random(SEED + 1), "")

    reqs = []

    def add(set_, mode, segs, cands, sems, **meta):
        reqs.append(dict(rid=stable_id("e3ev", set_, mode, meta, n=20), set=set_, mode=mode, segments=segs,
                         cands=cands, sems=sems, **meta))

    for g in groups:
        phr = ph[g["descriptor"]]
        for n in range(2, 7):
            for pname, pat in e3.base_patterns(n).items():
                for r in range(n):
                    idx = e3.arrange(n, r)
                    vals = [pat[i] for i in idx]
                    ims = [g["P"][i] if pat[i] else g["N"][i] for i in idx]
                    paths = [f"{root[g['source']]}/{im['path']}" for im in ims]
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
                segs, c, sm = prompts.probe_single_request("presence", phr, 0, f"{root[g['source']]}/{im['path']}")
                add("e3_single", "logit", segs, c, sm, source=g["source"], block="presence",
                    descriptor=g["descriptor"], image_id=im["image_id"], value=v, template=0, group=g["group"])
    assert len({r["rid"] for r in reqs}) == len(reqs)
    eval_ids = defaultdict(set)
    for g in groups:
        eval_ids[g["source"]] |= {im["image_id"] for im in g["P"] + g["N"]}

    # ---------------------------------------------------------------- training: three-image triples (A23, Dself23)
    ccfg = dict(cfg["clevr"], scenes_member="CLEVR_v1.0/scenes/CLEVR_train_scenes.json")
    clevr_train = [a for a in atoms_clevr.build(ccfg) if a["block"] == "count" or a["descriptor"] in kept]
    excl = {s: used[s] | eval_ids[s] for s in ("clevr", "coco")}
    triples = train_triples([a for a in clevr_train if a["block"] == "presence"], "clevr", excl["clevr"],
                            random.Random(SEED + 2), "train/")
    triples += train_triples(coco_train_pool, "coco", excl["coco"], random.Random(SEED + 3), "")
    three = []
    for tr in triples:
        phr = ph[tr["descriptor"]]
        paths = [f"{root[tr['source']]}/{im['path']}" for im in tr["imgs"]]
        for op in e3.TRAIN3_OPS:
            three.append(dict(item_id=stable_id("e3item3", tr["trip_id"], op), split="train", source=tr["source"],
                              block="presence", descriptor=tr["descriptor"], phrase=phr, predicate=op, n=3,
                              vals=tr["vals"], answer=e3.answer(op, tr["vals"]), template=0,
                              question=e3.question(op, 3, phr), paths=paths,
                              image_ids=[im["image_id"] for im in tr["imgs"]]))
    s1_train = [with_question(it) for it in s1_items if it["split"] == "train"]
    assert len(s1_train) == N_TOTAL
    strat = lambda it: (it["source"], it["block"], it["predicate"])  # noqa: E731
    train_23 = stratified(s1_train, N_TWO_IN_23, strat, "e3sub23") + three

    # ---------------------------------------------------------------- training: all six two-image operators (Aall)
    by_pair = {}
    for it in s1_train:
        by_pair.setdefault(it["pair_id"], it)
    all_items = []
    for pid, it in sorted(by_pair.items()):
        for pred in logic.BLOCK_PREDICATES[it["block"]]:
            iid = stable_id("s1item", pid, pred)
            t = int(in_subsample(iid + "|tmpl", 0.5))
            new = {k: it[k] for k in ("pair_id", "split", "source", "block", "descriptor", "phrase", "v1", "v2",
                                      "paths", "image_ids")}
            new.update(item_id=iid, predicate=pred, template=t, answer=logic.answer(pred, it["v1"], it["v2"]))
            all_items.append(with_question(new))
    train_all = stratified(all_items, N_TOTAL, strat, "e3suball")

    # ---------------------------------------------------------------- training: CLEVR only (Aclevr, Dselfclevr)
    trip_ids = {im["image_id"] for tr in triples if tr["source"] == "clevr" for im in tr["imgs"]}
    fresh = [a for a in clevr_train if a["block"] == "presence" and a["image_id"] not in (excl["clevr"] | trip_ids)]
    new_pairs, short = I.build_pairs(fresh, "clevr", "presence", 500, 0, ph, seed=SEED + 4)
    new_pairs = relabel(new_pairs, "train/")
    rep["clevr_extra_pairs"] = dict(n=len(new_pairs), shortfall=short)
    extra = []
    for p in new_pairs:
        paths = [f"{root['clevr']}/{p[k]['path']}" for k in ("img1", "img2")]
        for pred in s1.TRAIN_OPS["presence"]:
            iid = stable_id("e3item", p["pair_id"], pred)
            extra.append(with_question(dict(
                item_id=iid, pair_id=p["pair_id"], split="train", source="clevr", block="presence",
                descriptor=p["descriptor"], phrase=p["phrase"], predicate=pred, v1=p["v1"], v2=p["v2"],
                answer=logic.answer(pred, p["v1"], p["v2"]), template=int(in_subsample(iid + "|tmpl", 0.5)),
                paths=paths, image_ids=[p["img1"]["image_id"], p["img2"]["image_id"]])))
    train_clevr = [it for it in s1_train if it["source"] == "clevr"] + extra
    assert len(train_clevr) == N_TOTAL, len(train_clevr)

    # ---------------------------------------------------------------- leakage checks
    eval_paths = {v for r in reqs for k, v in r["segments"] if k == "image"}
    for name, items in (("train_23", train_23), ("train_all", train_all), ("train_clevr", train_clevr)):
        assert len(items) == N_TOTAL, (name, len(items))
        assert len({it["item_id"] for it in items}) == N_TOTAL, name
        tp = {p for it in items for p in it["paths"]}
        ti = {(it["source"], i) for it in items for i in it["image_ids"]}
        assert not (tp & eval_paths), f"{name}: training image in evaluation set"
        assert not (ti & {(s, i) for s in eval_ids for i in eval_ids[s]}), f"{name}: training id in evaluation set"
        rep[f"{name}_items"] = Counter(f"{it['source']}|{it['block']}|{it['predicate']}|n{len(it['paths'])}"
                                       for it in items)
    for s in ("clevr", "coco"):
        assert not (eval_ids[s] & used[s]), f"{s}: evaluation image already used by E0 or S1"

    rep["groups"] = Counter(f"{g['source']}|{g['descriptor']}" for g in groups)
    rep["eval_requests"] = Counter(f"{r['set']}|{r['mode']}" for r in reqs)
    rep["eval_requests_by_n"] = Counter(f"{r['mode']}|n{r['n']}" for r in reqs if r["set"] == "e3_grid")
    os.makedirs(D, exist_ok=True)
    write_jsonl(os.path.join(D, "groups.jsonl"), groups)
    write_jsonl(os.path.join(D, "train_triples.jsonl"), triples)
    write_jsonl(os.path.join(D, "train_23.jsonl"), train_23)
    write_jsonl(os.path.join(D, "train_all.jsonl"), train_all)
    write_jsonl(os.path.join(D, "train_clevr.jsonl"), train_clevr)
    write_jsonl(os.path.join(D, "eval_requests.jsonl"), reqs)
    with open(os.path.join(D, "build_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=dict)
    print(json.dumps(rep, indent=1, default=dict))

    if args.extract:
        need = defaultdict(set)
        for r in reqs:
            for k, v in r["segments"]:
                if k == "image":
                    need[r["source"]].add(v)
        for it in train_23 + train_clevr:
            need[it["source"]] |= {p for p in it["paths"] if p.startswith(IMG)}
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
