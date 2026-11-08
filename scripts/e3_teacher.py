"""Label-free targets for the E3 Dself arms (Dself23, Dselfclevr), same procedure as s1_teacher.py for any n:
the frozen model reads each training image alone, the readings are written as full sentences before the question
(images still shown), and the frozen model's answer at that prompt is stored. No hidden states (C3 is not trained).

Items already scored by the S1 teacher (same item_id, same prompt) are copied; only new items are scored.
Gold labels are used only in report.json. Output: data/e3/teacher_<model>/{readings,teacher}.jsonl, report.json

Usage: python scripts/e3_teacher.py --model qwen3vl_8b --device cuda:0 [--limit N]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib import e3, prompts  # noqa: E402
from e0lib.io import load_yaml, read_jsonl, write_jsonl  # noqa: E402

ITEM_FILES = ("train_23.jsonl", "train_clevr.jsonl")


def model_cfg(name):
    cfg = load_yaml(os.path.join(ROOT, "configs/models.yaml"))
    m = dict(cfg["defaults"])
    m.update(cfg["models"][name])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, help="smoke test on N new items")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "8"
    import torch
    torch.set_num_threads(8)
    from e0lib.scoring import Scorer

    out = os.path.join(ROOT, "data", "e3", f"teacher_{args.model}" + ("_smoke" if args.limit else ""))
    os.makedirs(out, exist_ok=True)
    s1dir = os.path.join(ROOT, "data", "s1", f"teacher_{args.model}")
    s1_rows = {t["item_id"]: t for t in read_jsonl(os.path.join(s1dir, "teacher.jsonl"))}
    s1_read = {(r["source"], r["block"], r["descriptor"], r["image_id"]): r["read"]
               for r in read_jsonl(os.path.join(s1dir, "readings.jsonl"))}
    items = {}
    for fn in ITEM_FILES:
        for it in read_jsonl(os.path.join(ROOT, "data", "e3", fn)):
            items[it["item_id"]] = it
    new = [it for iid, it in sorted(items.items()) if iid not in s1_rows]
    if args.limit:
        new = new[:: max(1, len(new) // args.limit)][: args.limit]
    print(f"{len(items)} items, {len(items) - len(new) if not args.limit else 0} copied from S1, {len(new)} to score",
          flush=True)
    sc = Scorer(model_cfg(args.model), device=args.device)
    bs = args.batch_size
    t0 = time.time()

    # 1. single-image readings for images the S1 teacher has not read (presence only: Yes/No)
    imgs = {}
    for it in new:
        for path, iid, v in zip(it["paths"], it["image_ids"], it["vals"]):
            key = (it["source"], it["block"], it["descriptor"], iid)
            if key not in s1_read:
                assert it["block"] == "presence"
                imgs[key] = (path, it["phrase"], v)
    reqs = []
    for key, (path, phrase, _) in sorted(imgs.items()):
        segs, c, sm = prompts.probe_single_request("presence", phrase, 0, path)
        reqs.append(dict(rid=f"{len(reqs)}", key=key, segments=segs, cands=c, sems=sm))
    reading, rows = dict(s1_read), []
    for i in range(0, len(reqs), bs):
        batch = reqs[i:i + bs]
        for r, res in zip(batch, sc.score_batch(batch)):
            d = defaultdict(float)
            for sem, p in zip(r["sems"], res["probs"]):
                d[int(sem)] += p
            reading[r["key"]] = max(d, key=d.get)
            k = r["key"]
            rows.append(dict(source=k[0], block=k[1], descriptor=k[2], image_id=k[3], read=reading[k],
                             gold=imgs[k][2], dist={a: round(x / sum(d.values()), 4) for a, x in d.items()}))
    write_jsonl(os.path.join(out, "readings.jsonl"), rows)
    print(f"readings: {len(rows)} new images in {(time.time() - t0) / 60:.1f} min", flush=True)

    # 2. teacher answers with the own readings stated before the question
    yes, no = sc.cand_token("Yes"), sc.cand_token("No")
    treqs = []
    for it in new:
        rv = [reading[(it["source"], it["block"], it["descriptor"], iid)] for iid in it["image_ids"]]
        treqs.append(dict(rid=it["item_id"], item_id=it["item_id"], read=rv, cands=["Yes", "No"], sems=["yes", "no"],
                          segments=e3.teacher_segments(it["paths"], it["question"], it["block"], it["phrase"], rv)))
    trow = [] if args.limit else [s1_rows[iid] for iid in sorted(items) if iid in s1_rows]
    for i in range(0, len(treqs), bs):
        batch = treqs[i:i + bs]
        for r, res in zip(batch, sc.score_batch(batch)):
            trow.append(dict(item_id=r["item_id"], read=r["read"], p_yes=res["probs"][0], mass_yes_no=res["mass"],
                             source="e3"))
        if (i // bs) % 100 == 0:
            print(f"teacher {i + len(batch)}/{len(treqs)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    write_jsonl(os.path.join(out, "teacher.jsonl"), trow)

    # 3. report (gold used only here)
    rep = defaultdict(lambda: [0, 0])
    for r in rows:
        rep[f"read|{r['source']}"][0] += int(r["read"] == r["gold"])
        rep[f"read|{r['source']}"][1] += 1
    for t in trow:
        if t.get("source") != "e3":
            continue
        it = items[t["item_id"]]
        tag = f"{it['source']}|{it['predicate']}|n{len(it['paths'])}"
        for k, ok in ((f"teacher|{tag}", (t["p_yes"] > 0.5) == it["answer"]),
                      (f"facts_correct|{tag}", list(t["read"]) == list(it["vals"])),
                      (f"teacher_follows_facts|{tag}", (t["p_yes"] > 0.5) == e3.answer(it["predicate"], t["read"])
                       if it["predicate"] in e3.OPS else None)):
            if ok is not None:
                rep[k][0] += int(ok)
                rep[k][1] += 1
    summary = {k: dict(acc=round(a / n, 4), n=n) for k, (a, n) in sorted(rep.items())}
    summary["n_rows"] = len(trow)
    summary["min_mass_yes_no"] = min((t["mass_yes_no"] for t in trow if t.get("source") == "e3"), default=None)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
