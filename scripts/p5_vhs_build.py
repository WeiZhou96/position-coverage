"""P5b: Visual Haystacks (single needle, 5 images; MIT licence) with the needle rotated through every slot.

Source: $VHS_ROOT/single_needle/VHs_large/visual_haystack_5.json (1,000 questions,
"For the image with <needle>, is there <target>?", 500 yes / 500 no; COCO train2017 images).
- Items whose images appear in any of this project's training sets (data/s1/train_items.jsonl, data/e3/train_*.jsonl,
  data/p1/train_*.jsonl) are dropped, so that trained checkpoints never saw the evaluation images.
- For every kept item the five images stay fixed; the needle image is placed at slot 1..5 and the four other images
  keep their order (5 requests per item). Layout as in E3: "Image k: <image>" lines, the question, "Answer with Yes or
  No only." (s1.student_segments).
- Images are extracted from the COCO 2017 train or val zip (by file name; some VHs paths name the wrong
  split) into
  $VHS_ROOT/images/ (only the needed files).
Output: data/p5vhs/eval_requests.jsonl, data/p5vhs/build_report.json. Usage: python scripts/p5_vhs_build.py
"""
import glob
import hashlib
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib import s1  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402

VHS = os.environ.get("VHS_ROOT", os.path.join(ROOT, "data", "raw", "visual_haystacks"))
COCO = os.environ.get("COCO_ROOT", os.path.join(ROOT, "data", "raw", "coco2017"))
ZIPS = [os.path.join(COCO, "train2017.zip"), os.path.join(COCO, "val2017.zip")]
IMG = os.path.join(VHS, "images")
OUT = os.path.join(ROOT, "data", "p5vhs")


def rid(*parts):
    return hashlib.md5(json.dumps(parts).encode()).hexdigest()[:20]


def main():
    data = json.load(open(os.path.join(VHS, "single_needle", "VHs_large", "visual_haystack_5.json")))
    used = set()
    files = [os.path.join(ROOT, "data", "s1", "train_items.jsonl")] + sorted(
        glob.glob(os.path.join(ROOT, "data", "e3", "train_*.jsonl"))) + sorted(
        glob.glob(os.path.join(ROOT, "data", "p1", "train_*.jsonl")))
    for f in files:
        for it in iter_jsonl(f):
            for i in it.get("image_ids") or []:
                used.add(os.path.basename(i))
    kept, dropped = [], 0
    for d in data:
        imgs = d["pos_image"] + d["neg_image"]
        assert len(d["pos_image"]) == 1 and len(imgs) == 5
        if {os.path.basename(p) for p in imgs} & used:
            dropped += 1
            continue
        kept.append(d)
    need = sorted({p for d in kept for p in d["pos_image"] + d["neg_image"]})
    zips = [zipfile.ZipFile(z) for z in ZIPS]          # some VHs paths say train2017 for val2017 images
    index = {}
    for z in zips:
        for name in z.namelist():
            index.setdefault(os.path.basename(name), (z, name))
    for p in need:
        dst = os.path.join(IMG, p)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            z, name = index[os.path.basename(p)]
            with z.open(name) as src, open(dst, "wb") as out:
                out.write(src.read())
    reqs = []
    for d in kept:
        q = d["conversations"][0]["value"].strip()
        ans = d["conversations"][1]["value"].strip().lower() == "yes"
        needle, negs = d["pos_image"][0], d["neg_image"]
        for slot in range(1, 6):
            order = negs[:slot - 1] + [needle] + negs[slot - 1:]
            paths = [os.path.join(IMG, p) for p in order]
            reqs.append(dict(rid=rid("vhs", d["id"], slot), set="vhs", mode="logit", source="coco",
                             segments=[list(s) for s in s1.student_segments(paths, q)], cands=["Yes", "No"],
                             sems=[1, 0], item_id=d["id"], slot=slot, n=5, answer=ans, needle=d["needle"],
                             target=d["target"], image_ids=[os.path.basename(p) for p in order]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "eval_requests.jsonl"), "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    rep = dict(source_items=len(data), dropped_overlap=dropped, kept=len(kept), requests=len(reqs),
               yes=sum(1 for d in kept if d["conversations"][1]["value"].strip().lower() == "yes"), images=len(need))
    json.dump(rep, open(os.path.join(OUT, "build_report.json"), "w"), indent=1)
    print(rep)
    print(reqs[0]["segments"][-1])


if __name__ == "__main__":
    main()
