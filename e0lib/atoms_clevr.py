"""CLEVR atoms from the official scene graphs (perfect labels).

An atom row: {source, image_id, path, descriptor, block, value, ...}; `path` is relative to the
materialised image directory (images are pulled from the zip only for sampled pairs).
Presence descriptors give value 0/1; count descriptors give bin 0/1/2.
"""
from __future__ import annotations

import json
import zipfile
from typing import Dict, List


def matches(obj: Dict, spec: Dict[str, str]) -> bool:
    return all(obj.get(k) == v for k, v in spec.items())


def load_scenes(cfg: Dict) -> List[Dict]:
    if cfg.get("zip"):
        with zipfile.ZipFile(cfg["zip"]) as z:
            return json.load(z.open(cfg["scenes_member"]))["scenes"]
    return json.load(open(cfg["scenes_json"]))["scenes"]


def build(cfg: Dict) -> List[Dict]:
    rows = []
    for sc in load_scenes(cfg):
        fn = sc["image_filename"]
        objs = sc["objects"]
        for d in cfg["presence_descriptors"]:
            n = sum(matches(o, d["spec"]) for o in objs)
            rows.append(dict(source="clevr", image_id=fn, path=fn, descriptor=d["name"], block="presence",
                             value=int(n > 0), raw_count=n, cluster=fn))
        for d in cfg["count_descriptors"]:
            n = sum(matches(o, d["spec"]) for o in objs)
            rows.append(dict(source="clevr", image_id=fn, path=fn, descriptor=d["name"], block="count",
                             value=min(n, 2), raw_count=n, cluster=fn))
    return rows


def prevalence_report(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    rep: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r.get("report"):
            continue
        key = f"{r['block']}:{r['descriptor']}"
        rep.setdefault(key, {})
        rep[key][str(r["value"])] = rep[key].get(str(r["value"]), 0) + 1
    for key, cnt in rep.items():
        tot = sum(cnt.values())
        for k in list(cnt):
            cnt[k] = round(cnt[k] / tot, 4)
        cnt["n"] = tot
    return rep
