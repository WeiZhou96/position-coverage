"""C3 teacher targets, label-free: the frozen model reads each training image on its own, the readings are written
as full sentences before the question (images still shown), and the frozen model's answer at that prompt is stored.

Gold labels are used only to report reading and teacher accuracy, never in the targets.
Outputs (data/s1/teacher_<model>/): readings.jsonl, teacher.jsonl (top-k log-probs, P(yes)), hidden.npy
[N, 2 positions (before question, answer), 2 layers (L/2, 3L/4), H] float16, report.json.

Usage: python scripts/s1_teacher.py --model qwen3vl_8b --device cuda:0
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib import prompts, s1  # noqa: E402
from e0lib.io import load_yaml, read_jsonl, write_jsonl  # noqa: E402

TOPK = 20


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
    ap.add_argument("--limit", type=int, help="smoke test on the first N items")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "8"
    import numpy as np
    import torch
    torch.set_num_threads(8)
    from e0lib.scoring import Scorer

    out = os.path.join(ROOT, "data", "s1", f"teacher_{args.model}" + ("_smoke" if args.limit else ""))
    os.makedirs(out, exist_ok=True)
    items = read_jsonl(os.path.join(ROOT, "data", "s1", "train_items.jsonl"))
    if args.limit:
        items = items[:: max(1, len(items) // args.limit)][: args.limit]
    sc = Scorer(model_cfg(args.model), device=args.device)
    bs = args.batch_size
    t0 = time.time()

    # 1. single-image readings (presence: Yes/No; count: three bins, averaged over the 3 cyclic option orders)
    imgs = {}
    for it in items:
        for k in (0, 1):
            imgs[(it["source"], it["block"], it["descriptor"], it["image_ids"][k])] = (it["paths"][k], it["phrase"],
                                                                                         (it["v1"], it["v2"])[k])
    reqs = []
    for key, (path, phrase, _) in sorted(imgs.items()):
        for perm in ((0,) if key[1] == "presence" else (0, 1, 2)):
            segs, c, sm = prompts.probe_single_request(key[1], phrase, 0, path, perm)
            reqs.append(dict(rid=f"{len(reqs)}", key=key, segments=segs, cands=c, sems=sm))
    probs = defaultdict(lambda: defaultdict(float))
    for i in range(0, len(reqs), bs):
        batch = reqs[i:i + bs]
        for r, res in zip(batch, sc.score_batch(batch)):
            for sem, p in zip(r["sems"], res["probs"]):
                probs[r["key"]][int(sem)] += p
    reading, rows = {}, []
    for key, d in probs.items():
        v = max(d, key=d.get)
        reading[key] = v
        rows.append(dict(source=key[0], block=key[1], descriptor=key[2], image_id=key[3], read=v,
                         gold=imgs[key][2], dist={k: round(x / sum(d.values()), 4) for k, x in d.items()}))
    write_jsonl(os.path.join(out, "readings.jsonl"), rows)
    print(f"readings: {len(rows)} images in {(time.time() - t0) / 60:.1f} min", flush=True)

    # 2. teacher prompts: own readings as full sentences before the question
    n_layers = sc.model.config.get_text_config().num_hidden_layers
    layers = s1.rep_layers(n_layers)
    yes, no = sc.cand_token("Yes"), sc.cand_token("No")
    treqs = []
    for it in items:
        rv = [reading[(it["source"], it["block"], it["descriptor"], it["image_ids"][k])] for k in (0, 1)]
        q = prompts.fill(prompts.REL_TEMPLATES[it["predicate"]][it["template"]], it["phrase"])
        treqs.append(dict(item_id=it["item_id"], q=q, read=rv,
                          segments=s1.teacher_segments(it["paths"], q, it["block"], it["phrase"], rv)))
    hidden = np.zeros((len(treqs), 2, 2, sc.model.config.get_text_config().hidden_size), dtype=np.float16)
    trow = []
    for i in range(0, len(treqs), bs):
        batch = treqs[i:i + bs]
        enc = sc.encode(batch)
        enc = {k: (v.to(sc.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        with torch.inference_mode():
            o = sc.model(**enc, output_hidden_states=True, logits_to_keep=1)
        lp = torch.log_softmax(o.logits[:, -1, :].float(), dim=-1)
        top = torch.topk(lp, TOPK, dim=-1)
        ids = enc["input_ids"].tolist()
        for b, r in enumerate(batch):
            pq, pa = s1.align_positions(sc.tok, ids[b], r["q"])
            for li, l in enumerate(layers):
                hidden[i + b, 0, li] = o.hidden_states[l][b, pq].float().cpu().numpy()
                hidden[i + b, 1, li] = o.hidden_states[l][b, pa].float().cpu().numpy()
            py = float(torch.softmax(lp[b, [yes, no]], dim=-1)[0])
            trow.append(dict(item_id=r["item_id"], read=r["read"], p_yes=py,
                             top_ids=top.indices[b].tolist(), top_lp=[round(x, 5) for x in top.values[b].tolist()],
                             mass_yes_no=float(lp[b, [yes, no]].exp().sum())))
        if (i // bs) % 100 == 0:
            print(f"teacher {i + len(batch)}/{len(treqs)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    write_jsonl(os.path.join(out, "teacher.jsonl"), trow)
    np.save(os.path.join(out, "hidden.npy"), hidden)

    # 3. report (gold used only here)
    gold = {it["item_id"]: it for it in items}
    rep = defaultdict(lambda: [0, 0])
    for r in rows:
        k = f"read|{r['source']}|{r['block']}"
        rep[k][0] += int(r["read"] == r["gold"])
        rep[k][1] += 1
    for t in trow:
        it = gold[t["item_id"]]
        cell = f"{it['v1']}{it['v2']}"
        for k in (f"teacher|{it['source']}|{it['predicate']}|{cell}", f"teacher|{it['source']}|{it['predicate']}",
                  f"facts_correct|{it['source']}|{it['block']}"):
            if k.startswith("facts_correct"):
                rep[k][0] += int(t["read"] == [it["v1"], it["v2"]])
            else:
                rep[k][0] += int((t["p_yes"] > 0.5) == it["answer"])
            rep[k][1] += 1
    summary = {k: dict(acc=round(a / n, 4), n=n) for k, (a, n) in sorted(rep.items())}
    summary["layers"] = list(layers)
    summary["min_mass_yes_no"] = min(t["mass_yes_no"] for t in trow)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
