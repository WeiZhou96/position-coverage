"""Evaluate the base model or a trained S1 run on data/s1/eval_requests.jsonl (resumable).

  logit mode  every run: first-token Yes/No scoring, as in E0
  gen mode    arms D, Dself and the base model (zero-shot state-then-answer): greedy generation, 'Answer: Yes/No' parsed

Usage: python scripts/s1_eval.py --run base|A_s1|... --device cuda:0 [--modes logit gen] [--limit N]
       --data e3 evaluates data/e3/eval_requests.jsonl and writes eval_e3_<mode>.jsonl into the run directory
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib import s1  # noqa: E402
from e0lib.io import iter_jsonl, load_yaml  # noqa: E402


def model_cfg(name):
    cfg = load_yaml(os.path.join(ROOT, "configs/models.yaml"))
    m = dict(cfg["defaults"])
    m.update(cfg["models"][name])
    return m


def parse_answer(text):
    m = re.findall(r"answer\s*[:：]\s*\**\s*(yes|no)", text, flags=re.I)
    if m:
        return m[-1].lower() == "yes", True
    m = re.findall(r"\b(yes|no)\b", text, flags=re.I)
    return ((m[-1].lower() == "yes") if m else None), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--modes", nargs="+")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--limit", type=int, help="smoke test: N requests per mode")
    ap.add_argument("--data", default="s1", help="request set: data/<data>/eval_requests.jsonl")
    ap.add_argument("--base-model", action="store_true",
                    help="evaluate the base model without LoRA; --run only names the output directory (P2)")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "8"
    import torch
    torch.set_num_threads(8)
    from e0lib.scoring import Scorer

    modes = args.modes or (["logit", "gen"] if (args.run == "base" or args.run.startswith("D")) else ["logit"])
    out = os.path.join(ROOT, "outputs", "s1", "runs", args.run + ("_smoke" if args.limit else ""))
    os.makedirs(out, exist_ok=True)
    sc = Scorer(model_cfg(args.model), device=args.device)
    if args.run != "base" and not args.base_model:
        ck = torch.load(os.path.join(ROOT, "outputs", "s1", "runs", args.run, "lora.pt"), map_location="cpu")
        hp = ck["meta"]["hp"]
        s1.inject_lora(sc.model, hp["lora_r"], hp["lora_alpha"], 0.0)
        missing, unexpected = sc.model.load_state_dict(ck["lora"], strict=False)
        assert not unexpected and not [k for k in missing if "lora_" in k], "LoRA weights do not match"
        s1.merge_lora(sc.model)
        sc.model.eval()
    reqs_all = list(iter_jsonl(os.path.join(ROOT, "data", args.data, "eval_requests.jsonl")))
    for mode in modes:
        path = os.path.join(out, f"eval_{mode}.jsonl" if args.data == "s1" else f"eval_{args.data}_{mode}.jsonl")
        done = {r["rid"] for r in iter_jsonl(path)} if os.path.exists(path) else set()
        reqs = [r for r in reqs_all if r["mode"] == mode and r["rid"] not in done]
        if args.limit:
            reqs = reqs[:: max(1, len(reqs) // args.limit)][: args.limit]
        groups = defaultdict(list)
        for r in reqs:
            groups[sum(1 for k, _ in r["segments"] if k == "image")].append(r)
        batches = []
        for _, rs in sorted(groups.items()):
            rs.sort(key=lambda r: (tuple(v for k, v in r["segments"] if k == "image"), r["rid"]))
            batches += [rs[i:i + args.batch_size] for i in range(0, len(rs), args.batch_size)]
        print(f"{args.run} {mode}: {len(reqs)} requests ({len(done)} done)", flush=True)
        pool = ThreadPoolExecutor(max_workers=1)
        pending, it = deque(), iter(batches)

        def submit():
            b = next(it, None)
            if b is not None:
                pending.append((b, pool.submit(sc.encode, b)))

        for _ in range(3):
            submit()
        t0, n = time.time(), 0
        with open(path, "a") as f:
            while pending:
                batch, fut = pending.popleft()
                submit()
                enc = fut.result()
                if mode == "logit":
                    rows = sc.score_batch(batch, enc)
                    rows = [dict(rid=r["rid"], probs=r["probs"], mass=r["mass"]) for r in rows]
                else:
                    enc = {k: (v.to(sc.device) if hasattr(v, "to") else v) for k, v in enc.items()}
                    with torch.inference_mode():
                        g = sc.model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
                    texts = sc.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    rows = []
                    for r, txt in zip(batch, texts):
                        ans, explicit = parse_answer(txt)
                        rows.append(dict(rid=r["rid"], text=txt, pred=ans, explicit=explicit))
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                f.flush()
                n += len(batch)
                if n % (args.batch_size * 100) < args.batch_size:
                    rate = n / (time.time() - t0)
                    print(f"  {n}/{len(reqs)} {rate:.1f}/s eta {(len(reqs) - n) / max(rate, 1e-6) / 60:.0f} min",
                          flush=True)
        pool.shutdown()
        print(f"{args.run} {mode}: finished {n} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
