"""P3 mechanism probe: hidden states and attention at the answer position (the preregistered analysis plan).

Items: E3 grid, operator ALL, patterns F (all present) and B1 (one image missing), n = 4 and 6, every rotation, all
24 groups (480 items). For each item, with the E3 direct-answer prompt (identical to eval_e3_logit):
  hidden[l]   hidden state of the last prompt position (where the Yes/No answer is read) after layer l = 0..L
  attn[l, k]  attention from that position to the tokens of image k (sum over the image's tokens, mean over heads)
  ntok[k]     number of tokens of image k;  p_yes  P(Yes) from the final logits (as in eval_e3_logit)
Eager attention, batch size 1. Output: outputs/p3/<run>.npz and outputs/p3/<run>.meta.jsonl.
Usage: python scripts/p3_extract.py --run A_s1 [--model qwen3vl_8b] --device cuda:0 [--limit N]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from e0lib import s1  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402


def items():
    out = []
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        if (r["set"] == "e3_grid" and r["mode"] == "logit" and r["predicate"] == "ALL" and r["n"] in (4, 6)
                and r["pattern"] in ("F", "B1")):
            out.append(r)
    out.sort(key=lambda r: (r["n"], r["group"], r["pattern"], r["rot"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    import torch
    from s1_eval import model_cfg
    from e0lib.scoring import Scorer

    cfg = model_cfg(args.model)
    cfg["attn"] = "eager"
    sc = Scorer(cfg, device=args.device)
    if args.run != "base":
        ck = torch.load(os.path.join(ROOT, "outputs", "s1", "runs", args.run, "lora.pt"), map_location="cpu")
        hp = ck["meta"]["hp"]
        s1.inject_lora(sc.model, hp["lora_r"], hp["lora_alpha"], 0.0)
        missing, unexpected = sc.model.load_state_dict(ck["lora"], strict=False)
        assert not unexpected and not [k for k in missing if "lora_" in k], "LoRA weights do not match"
        s1.merge_lora(sc.model)
        sc.model.eval()
    its = items()
    if args.limit:
        its = its[:: max(1, len(its) // args.limit)][: args.limit]
    yes, no = sc.cand_token("Yes"), sc.cand_token("No")
    H, A, NT, meta = [], [], [], []
    with torch.inference_mode():
        for i, r in enumerate(its):
            enc = sc.encode([r])
            enc = {k: (v.to(sc.device) if hasattr(v, "to") else v) for k, v in enc.items()}
            out = sc.model(**enc, output_hidden_states=True, output_attentions=True)
            ids = enc["input_ids"][0]
            is_img = (ids == sc.image_token_id).cpu().numpy()
            starts = np.flatnonzero(is_img & ~np.r_[False, is_img[:-1]])
            ends = np.flatnonzero(is_img & ~np.r_[is_img[1:], False]) + 1
            assert len(starts) == r["n"] == len(ends), (len(starts), r["n"])
            H.append(np.stack([h[0, -1].float().cpu().numpy() for h in out.hidden_states]).astype(np.float16))
            att = np.stack([a[0, :, -1, :].float().mean(0).cpu().numpy() for a in out.attentions])   # [L, T]
            row = np.full((att.shape[0], 6), np.nan, dtype=np.float32)
            for k, (s, e) in enumerate(zip(starts, ends)):
                row[:, k] = att[:, s:e].sum(1)
            A.append(row)
            NT.append(np.r_[ends - starts, np.zeros(6 - r["n"], dtype=int)])
            lg = out.logits[0, -1].float()
            p = torch.softmax(lg[[yes, no]], 0)[0].item()
            meta.append(dict(rid=r["rid"], group=r["group"], source=r["source"], n=r["n"], pattern=r["pattern"],
                             rot=r["rot"], vals=r["vals"], answer=r["answer"], p_yes=p))
            if i % 50 == 0:
                print(f"  {i}/{len(its)}", flush=True)
    od = os.path.join(ROOT, "outputs", "p3")
    os.makedirs(od, exist_ok=True)
    name = args.run + ("_smoke" if args.limit else "")
    np.savez_compressed(os.path.join(od, f"{name}.npz"), hidden=np.stack(H), attn=np.stack(A), ntok=np.stack(NT))
    with open(os.path.join(od, f"{name}.meta.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"{name}: {len(meta)} items, hidden {np.stack(H).shape}, attn {np.stack(A).shape}", flush=True)


if __name__ == "__main__":
    main()
