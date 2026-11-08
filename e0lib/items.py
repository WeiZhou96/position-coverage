"""Pairs -> items -> model requests (E0 design v2, sections 4 and 5)."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from . import logic, prompts
from .io import in_subsample, stable_id


# --------------------------------------------------------------------------- pairs
def build_pairs(atoms: List[Dict], source: str, block: str, n_main: int, n_pilot: int,
                phrases: Dict[str, Dict[str, str]], seed: int = 0, use_matching: bool = False) -> Tuple[List[Dict], Dict]:
    """Sample image pairs for every atom pattern/cell, without reusing an image inside (source, block).

    n_main / n_pilot are per pattern (presence) or per cell (count), spread evenly over descriptors.
    """
    rng = random.Random(f"{seed}|{source}|{block}")
    rows = [a for a in atoms if a.get("source") == source and a.get("block") == block and not a.get("report")]
    by_id = {}
    pools: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    matched_neg: Dict[Tuple[str, str], str] = {}
    for a in rows:
        key = (a["descriptor"], a["image_id"])
        by_id[key] = a
        pools[(a["descriptor"], a["value"])].append(a["image_id"])
    if use_matching:
        # COCO: matched_to stores the numeric id of the positive; file names are zero-padded ids
        num_to_img = {}
        for a in rows:
            if a["value"] == 1:
                num_to_img[(a["descriptor"], int(a["image_id"].split("/")[-1].split(".")[0]))] = a["image_id"]
        for a in rows:
            if a["value"] == 0 and a.get("matched_to") is not None:
                p = num_to_img.get((a["descriptor"], int(a["matched_to"])))
                if p:
                    matched_neg[(a["descriptor"], p)] = a["image_id"]
    for k in pools:
        rng.shuffle(pools[k])
    descriptors = sorted({d for d, _ in pools})
    used = set()

    def take(desc: str, value: int, avoid: Optional[str] = None) -> Optional[str]:
        pool = pools[(desc, value)]
        while pool:
            iid = pool.pop()
            if iid not in used and iid != avoid:
                used.add(iid)
                return iid
        return None

    out, shortfall = [], defaultdict(int)
    for split, n in (("pilot", n_pilot), ("main", n_main)):
        for (v1, v2) in logic.iter_block_cells(block):
            for j in range(n):
                desc = descriptors[j % len(descriptors)]
                mixed = use_matching and {v1, v2} == {0, 1}
                first_v, second_v = ((1, 0) if mixed else (v1, v2))
                ia = take(desc, first_v)
                ib = None
                if ia is not None and mixed:
                    m = matched_neg.get((desc, ia))
                    if m and m not in used:
                        used.add(m)
                        ib = m
                if ia is not None and ib is None:
                    ib = take(desc, second_v, avoid=ia)
                if ia is None or ib is None:
                    shortfall[f"{split}:{desc}:{v1}{v2}"] += 1
                    if ia is not None:
                        used.discard(ia)
                    continue
                i1, i2 = (ia, ib) if (not mixed or v1 == 1) else (ib, ia)
                a1, a2 = by_id[(desc, i1)], by_id[(desc, i2)]
                pid = stable_id(source, block, desc, i1, i2)
                out.append(dict(pair_id=pid, source=source, block=block, split=split, descriptor=desc,
                                phrase=phrases[desc], v1=v1, v2=v2,
                                img1=_img(a1), img2=_img(a2)))
    return out, dict(shortfall)


def _img(a: Dict) -> Dict:
    keep = ("image_id", "path", "cluster", "tar", "member", "package", "intersection", "raw_count")
    return {k: a[k] for k in keep if k in a}


# --------------------------------------------------------------------------- items
def build_items(pairs: List[Dict]) -> List[Dict]:
    items = []
    for p in pairs:
        dom = logic.BLOCK_DOMAIN[p["block"]]
        for pred in logic.BLOCK_PREDICATES[p["block"]]:
            s1, s2 = logic.sufficiency(pred, p["v1"], p["v2"], dom)
            items.append(dict(item_id=stable_id(p["pair_id"], pred), pair_id=p["pair_id"], source=p["source"],
                              block=p["block"], split=p["split"], descriptor=p["descriptor"], predicate=pred,
                              v1=p["v1"], v2=p["v2"], answer=logic.answer(pred, p["v1"], p["v2"]),
                              s1=s1, s2=s2, type=logic.suff_type(s1, s2)))
    return items


# --------------------------------------------------------------------------- requests
def _req(kind: str, segs, cands, sems, **key) -> Dict:
    rid = stable_id(kind, key, n=20)
    return dict(rid=rid, kind=kind, segments=segs, cands=cands, sems=sems, **key)


def build_requests(pairs: List[Dict], items: List[Dict], image_root: Dict[str, str],
                   frac_ter_full: float = 0.2, frac_bin_partial: float = 0.2,
                   same_image_twice: bool = False) -> List[Dict]:
    """All unique model inputs. image_root maps source -> directory that holds materialised images."""
    pair_of = {p["pair_id"]: p for p in pairs}
    reqs: Dict[str, Dict] = {}

    def add(r):
        reqs.setdefault(r["rid"], r)

    def full_path(src, img):
        return f"{image_root[src].rstrip('/')}/{img['path']}"

    for it in items:
        p = pair_of[it["pair_id"]]
        src, blk, desc, pred, ph = p["source"], p["block"], p["descriptor"], it["predicate"], p["phrase"]
        paths = (full_path(src, p["img1"]), full_path(src, p["img2"]))
        base = dict(source=src, block=blk, descriptor=desc, predicate=pred, split=p["split"])
        for t in (0, 1):
            for order in ("normal", "swap"):
                pp = paths if order == "normal" else paths[::-1]
                segs, c, s = prompts.relation_request(pred, ph, t, pp, (True, True), "bin")
                add(_req("rel", segs, c, s, item_id=it["item_id"], template=t, fmt="bin", visible="both",
                         order=order, perm=0, **base))
            for vis, mask in (("only1", (True, False)), ("only2", (False, True))):
                for perm in range(3):
                    segs, c, s = prompts.relation_request(pred, ph, t, paths, mask, "ter", perm)
                    add(_req("rel", segs, c, s, item_id=it["item_id"], template=t, fmt="ter", visible=vis,
                             order="normal", perm=perm, **base))
                if in_subsample(it["item_id"] + "|binpartial", frac_bin_partial):
                    segs, c, s = prompts.relation_request(pred, ph, t, paths, mask, "bin")
                    add(_req("rel", segs, c, s, item_id=it["item_id"], template=t, fmt="bin", visible=vis,
                             order="normal", perm=0, **base))
            if in_subsample(it["item_id"] + "|terfull", frac_ter_full):
                for perm in range(3):
                    segs, c, s = prompts.relation_request(pred, ph, t, paths, (True, True), "ter", perm)
                    add(_req("rel", segs, c, s, item_id=it["item_id"], template=t, fmt="ter", visible="both",
                             order="normal", perm=perm, **base))
            # hidden-both inputs depend only on (descriptor, predicate, template): shared, no item_id
            for perm in range(3):
                segs, c, s = prompts.relation_request(pred, ph, t, paths, (False, False), "ter", perm)
                add(_req("rel", segs, c, s, item_id=None, template=t, fmt="ter", visible="none",
                         order="normal", perm=perm, **base))
            segs, c, s = prompts.relation_request(pred, ph, t, paths, (False, False), "bin")
            add(_req("rel", segs, c, s, item_id=None, template=t, fmt="bin", visible="none",
                     order="normal", perm=0, **base))
            segs, c, s = prompts.text_control_request(pred, ph, t, blk, p["v1"], p["v2"])
            add(_req("text", segs, c, s, item_id=it["item_id"], template=t, fmt="bin", **base))
            if same_image_twice and p["v1"] == p["v2"] and blk == "presence":
                segs, c, s = prompts.relation_request(pred, ph, t, (paths[0], paths[0]), (True, True), "bin")
                add(_req("rel_same", segs, c, s, item_id=it["item_id"], template=t, fmt="bin", **base))

    for p in pairs:
        src, blk, desc, ph = p["source"], p["block"], p["descriptor"], p["phrase"]
        paths = (full_path(src, p["img1"]), full_path(src, p["img2"]))
        perms = range(3) if blk == "count" else (0,)
        for slot, img in ((1, p["img1"]), (2, p["img2"])):
            for t in (0, 1):
                for perm in perms:
                    segs, c, s = prompts.probe_single_request(blk, ph, t, paths[slot - 1], perm)
                    add(_req("probe_single", segs, c, s, source=src, block=blk, descriptor=desc,
                             image_id=img["image_id"], split=p["split"], template=t, perm=perm))
        for order in ("normal", "swap"):
            pp = paths if order == "normal" else paths[::-1]
            for k in (1, 2):
                orig_slot = k if order == "normal" else 3 - k
                for t in (0, 1):
                    for perm in perms:
                        segs, c, s = prompts.probe_ctx_request(blk, ph, t, pp, k, perm)
                        add(_req("probe_ctx", segs, c, s, source=src, block=blk, descriptor=desc,
                                 pair_id=p["pair_id"], split=p["split"], order=order, asked_slot=k,
                                 orig_slot=orig_slot, template=t, perm=perm))
    return list(reqs.values())
