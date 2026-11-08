"""S1 data: training items (new images only) and evaluation requests (preregistered).

Training images never appear in E0:
  CLEVR  train split (E0 used val)
  COCO   train2017 images not used by any E0 pair, hash-split 85 / 15 into an S1 training pool and an S1 eval pool
Evaluation sets (no training image appears in any of them):
  ev_main    E0 main pairs, both images, templates 0/1, all six operators
  ev_tmpl    subset of ev_main pairs, five held-out paraphrases per operator
  ev_3img    three-image presence questions (ALL / ANY / ONE / NONE) on CLEVR val images unused by E0 and the COCO eval pool
  ev_single  single-image readings of every evaluation image (null model, reading guard)
  ev_text    E0 text control for ev_main items (logic guard)
Relation sets are also written in 'gen' mode (state-then-answer instruction) for arm D and the zero-shot reference.

Usage: python scripts/s1_build.py [--extract]
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
from e0lib import atoms_clevr, items as I, logic, prompts, s1  # noqa: E402
from e0lib.io import in_subsample, load_yaml, read_jsonl, stable_id, write_jsonl  # noqa: E402

SEED = 20261104
D = os.path.join(ROOT, "data", "s1")
IMG = os.path.join(D, "images")
N_TRAIN = {"presence": 500, "count": 200}      # pairs per pattern / cell
N_DEV = {"presence": 12, "count": 6}
N_EV = {"presence": 120, "count": 60}          # ev_main pairs per (source, cell)
N_TMPL = {"presence": 30, "count": 20}         # ev_tmpl pairs per (source, cell)
N_3IMG = 50                                    # triples per (source, pattern)


def e0_used_images():
    used = defaultdict(set)
    for p in read_jsonl(os.path.join(ROOT, "data/pairs.jsonl")):
        used[p["source"]] |= {p["img1"]["image_id"], p["img2"]["image_id"]}
    return used


def phrases(cfg):
    ph = {d["name"]: dict(singular=d["singular"], plural=d["plural"])
          for d in cfg["clevr"]["presence_descriptors"] + cfg["clevr"]["count_descriptors"]}
    ph.update({c["name"]: dict(singular=c["singular"], plural=c["plural"]) for c in cfg["coco"]["categories"]})
    return ph


def relabel(pairs, prefix):
    for p in pairs:
        p["split"] = {"pilot": "dev", "main": "train"}[p["split"]]
        for k in ("img1", "img2"):
            p[k]["path"] = prefix + p[k]["path"]
    return pairs


def bin_targets(pairs):
    """Blueprint arm B: exact conditional P(yes) of each partial view from the training histogram."""
    hist = defaultdict(Counter)
    for p in pairs:
        if p["split"] != "train":
            continue
        for pred in s1.TRAIN_OPS[p["block"]]:
            a = logic.answer(pred, p["v1"], p["v2"])
            key = (p["source"], p["descriptor"], pred)
            for vk in (("none",), ("only1", p["v1"]), ("only2", p["v2"])):
                hist[key + vk][a] += 1
    return {k: c[True] / (c[True] + c[False]) for k, c in hist.items()}


def three_triples(atoms, source, used_ids, rng, n, path_prefix):
    pools = defaultdict(list)
    descs = sorted({a["descriptor"] for a in atoms})
    for a in sorted(atoms, key=lambda a: (a["descriptor"], a["image_id"])):
        if a["image_id"] not in used_ids:
            pools[(a["descriptor"], a["value"])].append(a)
    for k in pools:
        rng.shuffle(pools[k])
    taken, out = set(), []
    for pat in sorted({(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)}):
        for j in range(n):
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
            if len(imgs) < 3:
                continue
            out.append(dict(trip_id=stable_id("trip", source, desc, [a["image_id"] for a in imgs]), source=source,
                            descriptor=desc, vals=list(pat),
                            imgs=[dict(image_id=a["image_id"], path=path_prefix + a["path"]) for a in imgs]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="materialise the images used by S1")
    args = ap.parse_args()
    cfg = load_yaml(os.path.join(ROOT, "configs/sources.yaml"))
    ph = phrases(cfg)
    used = e0_used_images()
    kept = json.load(open(os.path.join(ROOT, "data/build_report.json")))["clevr:presence_kept"]
    rep = {}

    # ---------------------------------------------------------------- training pairs
    ccfg = dict(cfg["clevr"], scenes_member="CLEVR_v1.0/scenes/CLEVR_train_scenes.json")
    clevr_train = [a for a in atoms_clevr.build(ccfg) if a["block"] == "count" or a["descriptor"] in kept]
    coco_all = [a for a in read_jsonl(os.path.join(ROOT, "data/atoms_coco.jsonl"))
                if a.get("split") == "train" and a["image_id"] not in used["coco"]]
    coco_eval_pool = [a for a in coco_all if in_subsample(a["image_id"] + "|s1eval", 0.15)]
    coco_train_pool = [a for a in coco_all if not in_subsample(a["image_id"] + "|s1eval", 0.15)]
    rep["coco_pool"] = dict(train=len({a["image_id"] for a in coco_train_pool}),
                            eval=len({a["image_id"] for a in coco_eval_pool}))
    pairs = []
    for src, atoms, blocks, prefix in (("clevr", clevr_train, ("presence", "count"), "train/"),
                                       ("coco", coco_train_pool, ("presence",), "")):
        for blk in blocks:
            pr, short = I.build_pairs(atoms, src, blk, N_TRAIN[blk], N_DEV[blk], ph, seed=SEED,
                                      use_matching=(src == "coco"))
            pairs += relabel(pr, prefix)
            rep[f"train_pairs:{src}:{blk}"] = dict(Counter(p["split"] for p in pr), shortfall=short)
    tgt = bin_targets(pairs)
    root = {"clevr": os.path.join(IMG, "clevr"), "coco": os.path.join(IMG, "coco")}
    train_items = []
    for p in pairs:
        paths = [f"{root[p['source']]}/{p[k]['path']}" for k in ("img1", "img2")]
        for pred in s1.TRAIN_OPS[p["block"]]:
            iid = stable_id("s1item", p["pair_id"], pred)
            t = int(in_subsample(iid + "|tmpl", 0.5))
            key = (p["source"], p["descriptor"], pred)
            train_items.append(dict(
                item_id=iid, pair_id=p["pair_id"], split=p["split"], source=p["source"], block=p["block"],
                descriptor=p["descriptor"], phrase=p["phrase"], predicate=pred, v1=p["v1"], v2=p["v2"],
                answer=logic.answer(pred, p["v1"], p["v2"]), template=t, paths=paths,
                image_ids=[p["img1"]["image_id"], p["img2"]["image_id"]],
                p_yes=dict(none=tgt[key + ("none",)], only1=tgt[key + ("only1", p["v1"])],
                           only2=tgt[key + ("only2", p["v2"])], both=float(logic.answer(pred, p["v1"], p["v2"])))))
    rep["train_items"] = Counter(f"{i['split']}|{i['source']}|{i['block']}|{i['predicate']}" for i in train_items)

    # ---------------------------------------------------------------- evaluation pairs (E0 main)
    rng = random.Random(SEED)
    e0_pairs = read_jsonl(os.path.join(ROOT, "data/pairs.jsonl"))
    path_of = {}
    for r in read_jsonl(os.path.join(ROOT, "data/requests.jsonl")):
        if r["kind"] == "probe_ctx" and r["order"] == "normal" and r["asked_slot"] == 1 and r["template"] == 0 \
                and r.get("perm", 0) == 0:
            path_of[r["pair_id"]] = [v for k, v in r["segments"] if k == "image"]
    groups = defaultdict(list)
    for p in e0_pairs:
        if p["split"] == "main" and p["source"] in ("clevr", "coco"):
            groups[(p["source"], p["block"], p["v1"], p["v2"])].append(p)
    ev_pairs, tmpl_pairs = [], []
    for k, ps in sorted(groups.items()):
        ps = sorted(ps, key=lambda p: p["pair_id"])
        rng.shuffle(ps)
        ev_pairs += ps[: N_EV[k[1]]]
        tmpl_pairs += ps[: N_TMPL[k[1]]]
    reqs = []

    def add(set_, mode, segs, cands, sems, **meta):
        reqs.append(dict(rid=stable_id("s1ev", set_, mode, meta, n=20), set=set_, mode=mode, segments=segs,
                         cands=cands, sems=sems, **meta))

    def rel_requests(set_, p, paths, templates, tmpl_kind):
        for pred in logic.BLOCK_PREDICATES[p["block"]]:
            for t in templates:
                tt = (prompts.REL_TEMPLATES if tmpl_kind == "train" else s1.HO_TEMPLATES)[pred][t]
                q = prompts.fill(tt, p["phrase"])
                meta = dict(source=p["source"], block=p["block"], descriptor=p["descriptor"], predicate=pred,
                            vals=[p["v1"], p["v2"]], answer=logic.answer(pred, p["v1"], p["v2"]), template=t,
                            pair_id=p["pair_id"], image_ids=[p["img1"]["image_id"], p["img2"]["image_id"]])
                add(set_, "logit", s1.student_segments(paths, q), ["Yes", "No"], ["yes", "no"], **meta)
                add(set_, "gen", s1.srt_segments(paths, q, p["block"], p["phrase"]), ["Yes", "No"], ["yes", "no"],
                    **meta)

    single = {}
    for p in ev_pairs:
        paths = path_of[p["pair_id"]]
        rel_requests("ev_main", p, paths, (0, 1), "train")
        for pred in logic.BLOCK_PREDICATES[p["block"]]:
            segs, c, sm = prompts.text_control_request(pred, p["phrase"], 0, p["block"], p["v1"], p["v2"])
            add("ev_text", "logit", segs, c, sm, source=p["source"], block=p["block"], descriptor=p["descriptor"],
                predicate=pred, vals=[p["v1"], p["v2"]], answer=logic.answer(pred, p["v1"], p["v2"]),
                template=0, pair_id=p["pair_id"])
        for k, img, v in ((0, p["img1"], p["v1"]), (1, p["img2"], p["v2"])):
            single[(p["source"], p["block"], p["descriptor"], img["image_id"])] = (paths[k], v, p["phrase"])
    for p in tmpl_pairs:
        rel_requests("ev_tmpl", p, path_of[p["pair_id"]], range(5), "heldout")

    # ---------------------------------------------------------------- three-image questions
    clevr_val = [a for a in read_jsonl(os.path.join(ROOT, "data/atoms_clevr.jsonl"))
                 if a["block"] == "presence" and a["descriptor"] in kept and not a.get("report")]
    triples = three_triples(clevr_val, "clevr", used["clevr"], random.Random(SEED + 1), N_3IMG, "val/")
    triples += three_triples([a for a in coco_eval_pool if a["block"] == "presence"], "coco", set(),
                             random.Random(SEED + 2), N_3IMG, "")
    for tr in triples:
        phr = ph[tr["descriptor"]]
        paths = [f"{root[tr['source']]}/{im['path']}" for im in tr["imgs"]]
        for op in s1.THREE_OPS:
            for t in (0, 1):
                q = prompts.fill(s1.THREE_TEMPLATES[op][t], phr)
                meta = dict(source=tr["source"], block="presence", descriptor=tr["descriptor"], predicate=op,
                            vals=tr["vals"], answer=s1.three_answer(op, tr["vals"]), template=t,
                            trip_id=tr["trip_id"], image_ids=[im["image_id"] for im in tr["imgs"]])
                add("ev_3img", "logit", s1.student_segments(paths, q), ["Yes", "No"], ["yes", "no"], **meta)
                add("ev_3img", "gen", s1.srt_segments(paths, q, "presence", phr), ["Yes", "No"], ["yes", "no"],
                    **meta)
        for path, im, v in zip(paths, tr["imgs"], tr["vals"]):
            single[(tr["source"], "presence", tr["descriptor"], im["image_id"])] = (path, v, phr)
    for (src, blk, desc, iid), (path, v, phr) in sorted(single.items()):
        for perm in ((0,) if blk == "presence" else (0, 1, 2)):
            segs, c, sm = prompts.probe_single_request(blk, phr, 0, path, perm)
            add("ev_single", "logit", segs, c, sm, source=src, block=blk, descriptor=desc, image_id=iid, value=v,
                template=0, perm=perm)

    # leakage check: no training image in any evaluation request
    train_paths = {pp for it in train_items for pp in it["paths"]}
    eval_paths = {v for r in reqs for k, v in r["segments"] if k == "image"}
    assert not (train_paths & eval_paths), "training image found in evaluation set"
    train_ids = {(it["source"], i) for it in train_items for i in it["image_ids"]}
    eval_ids = {(r["source"], i) for r in reqs for i in (r.get("image_ids") or [r.get("image_id")]) if i}
    assert not (train_ids & eval_ids), "training image id found in evaluation set"
    assert len({r["rid"] for r in reqs}) == len(reqs)

    rep["ev_requests"] = Counter(f"{r['set']}|{r['mode']}" for r in reqs)
    rep["triples"] = Counter(f"{t['source']}|{''.join(map(str, t['vals']))}" for t in triples)
    os.makedirs(D, exist_ok=True)
    write_jsonl(os.path.join(D, "train_pairs.jsonl"), pairs)
    write_jsonl(os.path.join(D, "train_items.jsonl"), train_items)
    write_jsonl(os.path.join(D, "triples.jsonl"), triples)
    write_jsonl(os.path.join(D, "eval_requests.jsonl"), reqs)
    with open(os.path.join(D, "build_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=dict)
    print(json.dumps(rep, indent=1, default=dict))

    if args.extract:
        need = defaultdict(set)
        for it in train_items:
            need[it["source"]] |= set(it["paths"])
        for tr in triples:
            need[tr["source"]] |= {f"{root[tr['source']]}/{im['path']}" for im in tr["imgs"]}
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
