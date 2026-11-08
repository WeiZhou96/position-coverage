"""COCO presence atoms with caption-checked negatives and scene-matched negatives.

positive : >=1 non-crowd instance whose area >= min_area_frac of the image
negative : zero instances (crowd included) and no caption mentions any configured name
ambiguous: everything else (excluded)
Each positive is paired with its most scene-similar unused negative (cosine over the
80-dim category-presence vector, target category removed); unmatched negatives are dropped.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np


def _mention_regex(names: List[str]) -> re.Pattern:
    alts = []
    for n in names:
        n = re.escape(n.lower())
        alts.append(n + r"(?:s|es)?")
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b")


def build(cfg: Dict, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []
    for split in cfg["splits"]:
        inst = json.load(open(cfg["instances_json"].format(split=split)))
        caps = json.load(open(cfg["captions_json"].format(split=split)))
        cat_ids = sorted(c["id"] for c in inst["categories"])
        cid_index = {c: i for i, c in enumerate(cat_ids)}
        name_of = {c["id"]: c["name"] for c in inst["categories"]}
        imgs = {im["id"]: im for im in inst["images"]}
        anns = defaultdict(list)
        for a in inst["annotations"]:
            anns[a["image_id"]].append(a)
        captions = defaultdict(list)
        for c in caps["annotations"]:
            captions[c["image_id"]].append(c["caption"].lower())
        vec = {}
        for iid in imgs:
            v = np.zeros(len(cat_ids), dtype=np.float32)
            for a in anns[iid]:
                v[cid_index[a["category_id"]]] = 1.0
            vec[iid] = v
        for cat in cfg["categories"]:
            cid = next(k for k, v in name_of.items() if v == cat["name"])
            rx = _mention_regex([cat["name"]] + cat.get("synonyms", []))
            pos, neg = [], []
            n_amb = 0
            for iid, im in imgs.items():
                area = im["width"] * im["height"]
                mine = [a for a in anns[iid] if a["category_id"] == cid]
                if any((not a.get("iscrowd", 0)) and a["area"] / area >= cfg["min_area_frac"] for a in mine):
                    pos.append(iid)
                elif not mine and not any(rx.search(c) for c in captions[iid]):
                    neg.append(iid)
                else:
                    n_amb += 1
            rng.shuffle(pos)
            pos = pos[: cfg.get("max_pos_per_category", 10 ** 9)]
            neg_pool = rng.sample(neg, min(len(neg), cfg.get("neg_pool_size", 20000)))
            matched = _match(pos, neg_pool, vec, cid_index[cid])
            for iid in pos:
                rows.append(_row(split, imgs[iid], cat["name"], 1, vec[iid]))
            for p_iid, n_iid in matched:
                r = _row(split, imgs[n_iid], cat["name"], 0, vec[n_iid])
                r["matched_to"] = p_iid
                rows.append(r)
            rows.append(dict(source="coco", report=True, split=split, descriptor=cat["name"],
                             n_pos=len(pos), n_neg_pool=len(neg), n_matched=len(matched), n_ambiguous=n_amb))
    return rows


def _row(split: str, im: Dict, cat: str, value: int, v) -> Dict:
    return dict(source="coco", image_id=f"{split}/{im['file_name']}", path=f"{split}2017/{im['file_name']}",
                descriptor=cat, block="presence", value=value, cluster=f"{split}/{im['file_name']}",
                split=split, scene=[int(i) for i in np.nonzero(v)[0]])


def _match(pos, neg, vec, drop_idx, topk: int = 50, chunk: int = 256):
    if not pos or not neg:
        return []
    N = np.stack([vec[i] for i in neg])
    N[:, drop_idx] = 0
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-6)
    used = np.zeros(len(neg), dtype=bool)
    out = []
    for s in range(0, len(pos), chunk):
        P = np.stack([vec[i] for i in pos[s:s + chunk]])
        P[:, drop_idx] = 0
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
        sim = P @ N.T
        k = min(topk, sim.shape[1])
        cand = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        for r, p_iid in enumerate(pos[s:s + chunk]):
            order = cand[r][np.argsort(-sim[r, cand[r]])]
            pick = next((j for j in order if not used[j]), None)
            if pick is None:
                free = np.nonzero(~used)[0]
                if len(free) == 0:
                    return out
                pick = free[np.argmax(sim[r, free])]
            used[pick] = True
            out.append((p_iid, neg[pick]))
    return out
