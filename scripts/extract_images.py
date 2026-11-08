"""Materialise only the images referenced by data/pairs.jsonl (zip/tar random access).

Usage: python scripts/extract_images.py [--sources clevr coco]
"""
import argparse
import os
import sys
import zipfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib.io import load_yaml, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["clevr", "coco"])
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/sources.yaml"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    pairs = read_jsonl(os.path.join(args.data, "pairs.jsonl"))
    need = defaultdict(dict)
    for p in pairs:
        for img in (p["img1"], p["img2"]):
            need[p["source"]][img["path"]] = img
    for src in args.sources:
        root = cfg["image_root"][src]
        todo = {k: v for k, v in need[src].items() if not os.path.exists(os.path.join(root, k))}
        print(f"{src}: {len(need[src])} images referenced, {len(todo)} to extract")
        if not todo:
            continue
        if src == "clevr":
            with zipfile.ZipFile(cfg["clevr"]["zip"]) as z:
                for rel in todo:
                    _write(os.path.join(root, rel), z.read(cfg["clevr"]["images_prefix"] + rel))
        elif src == "coco":
            by_split = defaultdict(list)
            for rel in todo:
                by_split[rel.split("2017/")[0]].append(rel)
            for split, rels in by_split.items():
                with zipfile.ZipFile(cfg["coco"]["image_zips"][split]) as z:
                    for rel in rels:
                        _write(os.path.join(root, rel), z.read(rel))


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


if __name__ == "__main__":
    main()
