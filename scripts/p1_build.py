"""P1 training sets: does the position of decisive evidence in training, not the number of images, set which slots
single-pass answering reads? (report 10.14, the preregistered analysis plan)

All arms: 11,600 answer-level items, built from data/e3/train_23.jsonl (the A23 set) so that images and two-image
items are shared with A23.
  train_3edge.jsonl  A23 with the slot order of some three-image items rotated (same images, same k, same answer) so
                     that, within the training set, every three-image answer is determined by slots 1 and 3 alone:
                       ALL: pattern 101 is never shown; ANY, NONE: 010 never shown; ONE: 010, 011, 110 never shown.
                     Rotated items keep their triple and are moved to an allowed rotation (about 19% of them).
  train_24.jsonl     A23's 8,400 two-image items + 3,200 four-image items (ALL/ANY/ONE/NONE, all 16 patterns,
                     25 items per source x operator x pattern) assembled from the images of A23's triples
                     (same descriptor, required presence values, no image twice in an item).
  train_23q.jsonl    dose arm: 384 of A23's three-image items (6 per source x operator x pattern) + 11,216 two-image
                     items (A23's 8,400 + 2,816 more S1 training items, stratified as in e3_build.py).
Checks: no image in data/sealed_image_ids.json or in the E3 evaluation set; 11,600 unique item ids per arm;
answers recomputed from vals; edge-solvability of the three-image part of train_3edge.
Output: data/p1/{train_3edge,train_24,train_23q}.jsonl, data/p1/build_report.json. Usage: python scripts/p1_build.py
"""
import hashlib
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from e0lib import e3  # noqa: E402
from e0lib.io import read_jsonl  # noqa: E402
from e3_build import stratified, with_question  # noqa: E402

OUT = os.path.join(ROOT, "data", "p1")
SEED = 2026110601
OPS4 = ("ALL", "ANY", "ONE", "NONE")
EDGE_ALLOWED = {
    "ALL": {p for p in itertools.product((0, 1), repeat=3)} - {(1, 0, 1)},
    "ANY": {p for p in itertools.product((0, 1), repeat=3)} - {(0, 1, 0)},
    "NONE": {p for p in itertools.product((0, 1), repeat=3)} - {(0, 1, 0)},
    "ONE": {(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1)},
}


def h(*parts):
    return hashlib.md5(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def rotate(it, s):
    """Cyclic rotation: new slot i holds old slot (i - s) mod n."""
    n = len(it["vals"])
    idx = [(i - s) % n for i in range(n)]
    new = dict(it)
    for k in ("vals", "paths", "image_ids"):
        new[k] = [it[k][j] for j in idx]
    return new


def edge_solvable(items):
    """Within each operator, items that agree on slots 1 and 3 must have the same answer."""
    seen = {}
    for it in items:
        key = (it["predicate"], it["vals"][0], it["vals"][-1])
        if seen.setdefault(key, it["answer"]) != it["answer"]:
            return False
    return True


def main():
    a23 = read_jsonl(os.path.join(ROOT, "data", "e3", "train_23.jsonl"))
    two = [it for it in a23 if len(it["image_ids"]) == 2]
    three = [it for it in a23 if len(it["image_ids"]) == 3]
    assert len(two) == 8400 and len(three) == 3200
    rep = {}

    # ---------------------------------------------------------------- A3edge
    edge, moved = [], Counter()
    for it in three:
        op, v = it["predicate"], tuple(it["vals"])
        if v in EDGE_ALLOWED[op]:
            edge.append(it)
            continue
        opts = [s for s in (1, 2) if tuple(rotate(it, s)["vals"]) in EDGE_ALLOWED[op]]
        assert opts, (op, v)
        s = opts[int(h("p1edge", it["item_id"]), 16) % len(opts)]
        new = rotate(it, s)
        assert e3.answer(op, new["vals"]) == it["answer"]
        new["item_id"] = it["item_id"] + "e"
        edge.append(new)
        moved[(op, v, tuple(new["vals"]))] += 1
    assert all(tuple(it["vals"]) in EDGE_ALLOWED[it["predicate"]] for it in edge)
    assert edge_solvable(edge) and not edge_solvable(three)
    train_3edge = two + edge
    rep["3edge"] = dict(rotated=sum(moved.values()), moves={str(k): v for k, v in sorted(moved.items())},
                        patterns={op: dict(Counter("".join(map(str, it["vals"])) for it in edge
                                                   if it["predicate"] == op)) for op in OPS4})

    # ---------------------------------------------------------------- A24
    pool = defaultdict(dict)                 # (source, descriptor, value) -> image_id -> path
    phrase = {}
    for it in three:
        for v, iid, p in zip(it["vals"], it["image_ids"], it["paths"]):
            pool[(it["source"], it["descriptor"], v)][iid] = p
        phrase[(it["source"], it["descriptor"])] = it["phrase"]
    rng = random.Random(SEED)
    four = []
    for src in ("clevr", "coco"):
        descs = sorted({d for (s, d, _) in pool if s == src})
        j = 0
        for op in OPS4:
            for pat in itertools.product((0, 1), repeat=4):
                for r in range(25):
                    for _ in range(len(descs)):
                        d = descs[j % len(descs)]
                        j += 1
                        need = Counter(pat)
                        if all(len(pool[(src, d, v)]) >= c for v, c in need.items()):
                            break
                    else:
                        raise AssertionError(f"no descriptor with enough images: {src} {pat}")
                    chosen = {v: rng.sample(sorted(pool[(src, d, v)]), need[v]) for v in need}
                    ids = [chosen[v].pop() for v in pat]
                    vals = list(pat)
                    four.append(dict(item_id=h("p1four", src, op, pat, r), split="train", source=src,
                                     block="presence", descriptor=d, phrase=phrase[(src, d)], predicate=op, n=4,
                                     vals=vals, answer=e3.answer(op, vals), template=0,
                                     question=e3.question(op, 4, phrase[(src, d)]),
                                     paths=[pool[(src, d, v)][i] for v, i in zip(vals, ids)], image_ids=ids))
    assert len(four) == 3200 and all(len(set(it["image_ids"])) == 4 for it in four)
    train_24 = two + four
    rep["24"] = dict(four=len(four), images=len({i for it in four for i in it["image_ids"]}),
                     answers={f"{k[0]}|{k[1]}": v for k, v in Counter((it["predicate"], it["answer"]) for it in four).items()})

    # ---------------------------------------------------------------- A23q (dose)
    s1_train = [with_question(it) for it in read_jsonl(os.path.join(ROOT, "data", "s1", "train_items.jsonl"))
                if it["split"] == "train"]
    in23 = {it["item_id"] for it in two}
    rest = [it for it in s1_train if it["item_id"] not in in23]
    strat = lambda it: (it["source"], it["block"], it["predicate"])  # noqa: E731
    extra = stratified(rest, 2816, strat, "p1q")
    by_cell = defaultdict(list)
    for it in three:
        by_cell[(it["source"], it["predicate"], tuple(it["vals"]))].append(it)
    three_q = []
    for k, v in sorted(by_cell.items(), key=lambda kv: str(kv[0])):
        three_q += sorted(v, key=lambda it: h("p1q3", it["item_id"]))[:6]
    assert len(three_q) == 384
    train_23q = two + extra + three_q
    rep["23q"] = dict(three=len(three_q), two=len(two) + len(extra))

    # ---------------------------------------------------------------- checks and output
    sealed = json.load(open(os.path.join(ROOT, "data", "sealed_image_ids.json")))
    sealed_ids = {(s, i) for s, ids in sealed.items() for i in ids} if isinstance(sealed, dict) else set(sealed)
    eval_ids = set()
    for it in read_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        for i in it.get("image_ids") or [it.get("image_id")]:
            eval_ids.add((it["source"], i))
    os.makedirs(OUT, exist_ok=True)
    for name, items in (("3edge", train_3edge), ("24", train_24), ("23q", train_23q)):
        assert len(items) == 11600 and len({it["item_id"] for it in items}) == 11600, name
        imgs = {(it["source"], i) for it in items for i in it["image_ids"]}
        assert not imgs & eval_ids, f"{name}: overlap with E3 evaluation images"
        if isinstance(sealed, dict):
            assert not imgs & sealed_ids, f"{name}: overlap with sealed images"
        else:
            assert not {i for _, i in imgs} & sealed_ids, f"{name}: overlap with sealed images"
        for it in items:
            if it["predicate"] in OPS4:
                assert e3.answer(it["predicate"], it["vals"]) == it["answer"]
        with open(os.path.join(OUT, f"train_{name}.jsonl"), "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        rep[name]["n_by_count"] = dict(Counter(len(it["image_ids"]) for it in items))
    with open(os.path.join(OUT, "build_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1)[:3000])


if __name__ == "__main__":
    main()
