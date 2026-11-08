"""Teacher-forced facts probe: does the answer follow the written facts or the images? (post hoc, inference only)

The prompt is the E3 state-then-answer prompt (s1.srt_segments). The assistant turn is pre-filled with per-image
facts in exactly the D / Dself training format (s1.srt_target without the answer), ending in "Answer:"; the next
token is scored over " Yes" / " No". Three conditions differ in one factor each:
  img        images shown, correct facts
  hidden     every image replaced by the hidden-image placeholder, correct facts (language pathway alone)
  conflict   images shown, facts with one decisive slot flipped (facts and images imply different answers);
             the slot is the decisive slot closest to the middle of the row
  selfimg    the run's own E3 state-then-answer facts (verbatim, up to "Answer:"), images shown (reproduces E3)
  selfhidden the run's own facts, images hidden at the answer step (inference-time fix, no training)
Items: E3 arrangements with n = 3 and 6, every pattern, rotation and operator, the 12 even-indexed groups.
Output: outputs/s1/runs/<run>/eval_e3force.jsonl (resumable). Usage:
  python scripts/dx_force_eval.py --run D_s1 --device cuda:0
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from e0lib import e3, prompts, s1  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402

NS = (3, 6)


def decisive_slot(op, vals):
    n = len(vals)
    cand = [k for k in range(n) if e3.answer(op, vals[:k] + [1 - vals[k]] + vals[k + 1:]) != e3.answer(op, vals)]
    if not cand:
        return None
    mid = (n - 1) / 2
    return min(cand, key=lambda k: (abs(k - mid), k))


def build(run):
    gen = {}
    gp = os.path.join(ROOT, "outputs", "s1", "runs", run, "eval_e3_gen.jsonl")
    for o in iter_jsonl(gp):
        gen[o["rid"]] = o["text"]
    items = []
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        if r["set"] != "e3_grid" or r["mode"] != "gen" or r["n"] not in NS or int(r["group"][-2:]) % 2:
            continue
        paths = [v for k, v in r["segments"] if k == "image"]
        q = r["segments"][-1][1].split("\n")[0]
        X = re.search(r"contains? (.+)\?$", q).group(1)
        phrase = {"singular": X, "plural": X}
        tail = [("text", f"{q}\n" + prompts.fill(s1.SRT_INSTR["presence"], phrase))]
        vals = list(r["vals"])
        base = dict(grid_rid=r["rid"], n=r["n"], predicate=r["predicate"], pattern=r["pattern"], rot=r["rot"],
                    group=r["group"], vals=vals, answer=r["answer"])
        facts = "\n".join(s1.fact_lines("presence", phrase, vals)) + "\nAnswer:"
        items.append(dict(base, rid=r["rid"] + "|img", cond="img", segments=s1.image_segments(paths) + tail,
                          prefix=facts, facts_answer=r["answer"]))
        items.append(dict(base, rid=r["rid"] + "|hidden", cond="hidden",
                          segments=s1.image_segments(paths, [False] * len(paths)) + tail, prefix=facts,
                          facts_answer=r["answer"]))
        k = decisive_slot(r["predicate"], vals)
        if k is not None:
            fv = vals[:k] + [1 - vals[k]] + vals[k + 1:]
            ff = "\n".join(s1.fact_lines("presence", phrase, fv)) + "\nAnswer:"
            items.append(dict(base, rid=r["rid"] + "|conflict", cond="conflict", segments=s1.image_segments(paths)
                              + tail, prefix=ff, facts_answer=e3.answer(r["predicate"], fv), flipped_slot=k + 1))
        own = gen.get(r["rid"], "")
        if "Answer:" in own:
            pre = own.split("Answer:")[0].rstrip() + "\nAnswer:"
            for cond, vis in (("selfimg", None), ("selfhidden", [False] * len(paths))):
                items.append(dict(base, rid=r["rid"] + "|" + cond, cond=cond,
                                  segments=s1.image_segments(paths, vis) + tail, prefix=pre, facts_answer=None))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    import torch
    from s1_eval import model_cfg
    from s1_train import extend_with_targets
    from e0lib.scoring import Scorer

    out_dir = os.path.join(ROOT, "outputs", "s1", "runs", args.run + ("_smoke" if args.limit else ""))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "eval_e3force.jsonl")
    sc = Scorer(model_cfg("qwen3vl_8b"), device=args.device)
    if args.run != "base":
        ck = torch.load(os.path.join(ROOT, "outputs", "s1", "runs", args.run, "lora.pt"), map_location="cpu")
        hp = ck["meta"]["hp"]
        s1.inject_lora(sc.model, hp["lora_r"], hp["lora_alpha"], 0.0)
        missing, unexpected = sc.model.load_state_dict(ck["lora"], strict=False)
        assert not unexpected and not [k for k in missing if "lora_" in k], "LoRA weights do not match"
        s1.merge_lora(sc.model)
        sc.model.eval()
    tok = sc.tok
    items = build(args.run)
    done = {o["rid"] for o in iter_jsonl(path)} if os.path.exists(path) else set()
    items = [it for it in items if it["rid"] not in done]
    if args.limit:
        items = items[:: max(1, len(items) // args.limit)][: args.limit]
    items.sort(key=lambda it: (it["cond"] in ("hidden", "selfhidden"), it["n"], it["grid_rid"]))
    print(f"{args.run}: {len(items)} items ({len(done)} done)", flush=True)
    with open(path, "a") as f, torch.inference_mode():
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            enc = sc.encode(batch)
            tg, cands = [], []
            for it in batch:
                pre = tok.encode(it["prefix"], add_special_tokens=False)
                ids = []
                for word in ("Yes", "No"):
                    full = tok.encode(it["prefix"] + " " + word, add_special_tokens=False)
                    assert full[:len(pre)] == pre and len(full) == len(pre) + 1, (it["prefix"][-20:], word)
                    ids.append(full[-1])
                assert ids == [tok.encode(" Yes", add_special_tokens=False)[-1], tok.encode(" No", add_special_tokens=False)[-1]]
                tg.append(pre)
                cands.append(ids)
            enc2, _ = extend_with_targets(enc, tg, tok.pad_token_id)
            enc2 = {k: (v.to(sc.device) if hasattr(v, "to") else v) for k, v in enc2.items()}
            logits = sc.model(**enc2).logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            for b, it in enumerate(batch):
                ly, ln = float(logp[b, cands[b][0]]), float(logp[b, cands[b][1]])
                m = max(ly, ln)
                py = torch.exp(torch.tensor(ly - m)) / (torch.exp(torch.tensor(ly - m)) + torch.exp(torch.tensor(ln - m)))
                row = {k: it[k] for k in it if k not in ("segments", "prefix")}
                row.update(p_yes=float(py), mass=float(torch.exp(torch.tensor(ly)) + torch.exp(torch.tensor(ln))))
                f.write(json.dumps(row) + "\n")
            f.flush()
            if (i // args.batch_size) % 50 == 0:
                print(f"  {i + len(batch)}/{len(items)}", flush=True)
    print(f"{args.run}: finished", flush=True)


if __name__ == "__main__":
    main()
