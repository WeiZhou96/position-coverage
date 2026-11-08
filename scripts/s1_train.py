"""S1 LoRA training. All arms share images, questions, templates, item order (per seed) and optimiser steps.

  A      gold labels, full input only (cross-entropy on the first answer token)
  B      blueprint four-view direct distillation: A + KD on {none, only1, only2, both} with exact conditional P(yes)
  Bd     B + the blueprint's signed interaction term (Huber on delta), weight 0.25
  C3     label-free: match the frozen model's answer when its own per-image readings are stated before the question
         (soft cross-entropy on the teacher's top-k) + cosine alignment of hidden states before the question and at
         the answer position (layers L/2, 3L/4)
  C3ans  C3 without the representation term
  D      state-then-answer (multi-image SRT): teacher-forced gold facts followed by 'Answer: Yes/No'
  Dself  label-free D: the facts are the model's own single-image readings and the answer is the frozen model's
         answer when those readings are stated before the question (the C3 teacher); no human labels

Usage: python scripts/s1_train.py --arm A --seed 1 --device cuda:0 [--max-steps N]
E3 arms reuse A and Dself on other item files (any number of images), e.g.
       python scripts/s1_train.py --arm Dself --name Dself23 --items data/e3/train_23.jsonl \
              --teacher-dir data/e3/teacher_qwen3vl_8b --seed 1 --device cuda:0
"""
import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib import e3, logic, prompts, s1  # noqa: E402
from e0lib.io import load_yaml, read_jsonl  # noqa: E402

ARMS = ("A", "B", "Bd", "C3", "C3ans", "D", "Dself")
HP = dict(lora_r=16, lora_alpha=32, lora_dropout=0.05, lr=1e-4, warmup_frac=0.03, items_per_step=16,
          weight_decay=0.0, grad_clip=1.0, epochs=1,
          kd_weight=1.0, interaction_weight=0.25, epsilon=0.01, huber_delta=1.0,   # blueprint starting values
          rep_weight=1.0)
MICRO = {"A": 8, "B": 8, "Bd": 8, "C3": 8, "C3ans": 8, "D": 8, "Dself": 8}


def model_cfg(name):
    cfg = load_yaml(os.path.join(ROOT, "configs/models.yaml"))
    m = dict(cfg["defaults"])
    m.update(cfg["models"][name])
    return m


def paired_evidence_loss(student_logits, target_probs, epsilon, kd_weight, interaction_weight, huber_delta):
    """Blueprint appendix B, unchanged: inputs [N, 4, K], observation order none, only1, only2, both."""
    import torch
    import torch.nn.functional as F
    scores = student_logits.float()
    targets = target_probs.detach().to(device=scores.device, dtype=torch.float32)
    k = scores.shape[-1]
    teacher = (targets + epsilon) / (1.0 + k * epsilon)
    teacher_log = teacher.log()
    student_log = F.log_softmax(scores, dim=-1)
    kd = (teacher * (teacher_log - student_log)).sum(-1).mean(-1)

    def contrast(lp):
        z = lp - lp.mean(dim=-1, keepdim=True)
        return z[:, 3] - z[:, 1] - z[:, 2] + z[:, 0]

    inter = F.huber_loss(contrast(student_log), contrast(teacher_log), delta=huber_delta, reduction="none").mean(-1)
    return (kd_weight * kd + interaction_weight * inter).mean(), kd.mean().detach(), inter.mean().detach()


def extend_with_targets(enc, targets, pad_id):
    """Append target token ids to every left-padded prompt row and re-pad on the left (images untouched)."""
    import torch
    ids, am = enc["input_ids"], enc["attention_mask"]
    seq_keys = [k for k, v in enc.items() if hasattr(v, "shape") and v.shape == ids.shape and k not in
                ("input_ids", "attention_mask")]
    rows = []
    for b in range(ids.shape[0]):
        keep = am[b].bool()
        t = torch.tensor(targets[b], dtype=ids.dtype)
        rows.append(dict(input_ids=torch.cat([ids[b][keep], t]),
                         attention_mask=torch.ones(int(keep.sum()) + len(t), dtype=am.dtype),
                         **{k: torch.cat([enc[k][b][keep], torch.zeros(len(t), dtype=enc[k].dtype)]) for k in seq_keys},
                         n_tgt=len(t)))
    L = max(len(r["input_ids"]) for r in rows)
    new = dict(enc)
    for k, fill in [("input_ids", pad_id), ("attention_mask", 0)] + [(k, 0) for k in seq_keys]:
        new[k] = torch.stack([torch.cat([torch.full((L - len(r[k]),), fill, dtype=r[k].dtype), r[k]]) for r in rows])
    labels = torch.full_like(new["input_ids"], -100)
    for b, r in enumerate(rows):
        labels[b, L - r["n_tgt"]:] = new["input_ids"][b, L - r["n_tgt"]:]
    return new, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, help="smoke test: stop after N optimiser steps")
    ap.add_argument("--micro", type=int, help="override the micro-batch size (items)")
    ap.add_argument("--teacher-dir", help="another teacher directory (E3 arms, smoke tests)")
    ap.add_argument("--items", default=os.path.join("data", "s1", "train_items.jsonl"), help="training item file")
    ap.add_argument("--name", help="run name prefix (default: the arm)")
    ap.add_argument("--answer-from-reads", action="store_true",
                    help="Dself only: answer target computed from the own readings by the operator definition")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "8"
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(8)
    from e0lib.scoring import Scorer

    run = f"{args.name or args.arm}_s{args.seed}" + ("_smoke" if args.max_steps else "")
    out = os.path.join(ROOT, "outputs", "s1", "runs", run)
    os.makedirs(out, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = [it for it in read_jsonl(os.path.join(ROOT, args.items)) if it["split"] == "train"]
    items.sort(key=lambda it: it["item_id"])
    random.Random(args.seed).shuffle(items)          # same order for every arm with this seed
    teach, hid = None, None
    if args.arm in ("C3", "C3ans", "Dself"):
        tdir = args.teacher_dir or os.path.join(ROOT, "data", "s1", f"teacher_{args.model}")
        trows = read_jsonl(os.path.join(tdir, "teacher.jsonl"))
        teach = {t["item_id"]: (j, t) for j, t in enumerate(trows)}
        hid = np.load(os.path.join(tdir, "hidden.npy"), mmap_mode="r") if args.arm == "C3" else None
        if args.max_steps:                           # smoke test against a partial teacher
            items = [it for it in items if it["item_id"] in teach]
        missing = [it["item_id"] for it in items if it["item_id"] not in teach]
        assert not missing, f"{len(missing)} items lack teacher targets"

    sc = Scorer(model_cfg(args.model), device=args.device)
    model, tok = sc.model, sc.tok
    params, n_lora = s1.inject_lora(model, HP["lora_r"], HP["lora_alpha"], HP["lora_dropout"])
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    yes, no = sc.cand_token("Yes"), sc.cand_token("No")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    layers = s1.rep_layers(model.config.get_text_config().num_hidden_layers)

    micro = args.micro or MICRO[args.arm]
    accum = HP["items_per_step"] // micro
    steps = (len(items) * HP["epochs"]) // HP["items_per_step"]
    if args.max_steps:
        steps = min(steps, args.max_steps)
    warm = max(1, int(HP["warmup_frac"] * steps))
    opt = torch.optim.AdamW(params, lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    meta = dict(arm=args.arm, name=args.name or args.arm, items=args.items, seed=args.seed, model=args.model,
                teacher_dir=args.teacher_dir, answer_from_reads=args.answer_from_reads, n_items=len(items),
                steps=steps, micro=micro,
                accum=accum, n_lora_modules=n_lora, n_lora_params=sum(p.numel() for p in params), hp=HP,
                rep_layers=list(layers))
    print(json.dumps(meta), flush=True)
    log = open(os.path.join(out, "train_log.jsonl"), "a")

    def question(it):
        if "question" in it:                         # E3 item files carry the question text
            return it["question"]
        return prompts.fill(prompts.REL_TEMPLATES[it["predicate"]][it["template"]], it["phrase"])

    def micro_loss(batch):
        stats = {}
        if args.arm in ("A", "C3", "C3ans"):
            reqs = [dict(segments=s1.student_segments(it["paths"], question(it))) for it in batch]
            enc = {k: (v.to(sc.device) if hasattr(v, "to") else v) for k, v in sc.encode(reqs).items()}
            o = model(**enc, logits_to_keep=1, output_hidden_states=(args.arm == "C3"))
            lp = F.log_softmax(o.logits[:, -1, :].float(), dim=-1)
            if args.arm == "A":
                gold = torch.tensor([yes if it["answer"] else no for it in batch], device=lp.device)
                loss = F.nll_loss(lp, gold)
                stats["ce"] = float(loss)
                return loss, stats
            ce = []
            for b, it in enumerate(batch):
                j, t = teach[it["item_id"]]
                q = torch.softmax(torch.tensor(t["top_lp"], device=lp.device), dim=-1)
                ce.append(-(q * lp[b, t["top_ids"]]).sum())
            loss = torch.stack(ce).mean()
            stats["soft_ce"] = float(loss)
            if args.arm == "C3":
                ids = enc["input_ids"].tolist()
                cos = []
                for b, it in enumerate(batch):
                    j, _ = teach[it["item_id"]]
                    pq, pa = s1.align_positions(tok, ids[b], question(it))
                    tgt = torch.tensor(np.asarray(hid[j]), device=lp.device, dtype=torch.float32)  # [2 pos, 2 layers, H]
                    for li, l in enumerate(layers):
                        h = o.hidden_states[l][b]
                        cos.append(F.cosine_similarity(h[pq].float(), tgt[0, li], dim=0))
                        cos.append(F.cosine_similarity(h[pa].float(), tgt[1, li], dim=0))
                rep = 1 - torch.stack(cos).mean()
                stats["rep"] = float(rep)
                loss = loss + HP["rep_weight"] * rep
            return loss, stats
        if args.arm in ("B", "Bd"):
            # one forward per view group (0, 1 or 2 images) so short prompts are not padded to two-image length
            views = ((False, False), (True, False), (False, True), (True, True))
            last = [None] * 4
            for group in ((0,), (1, 2), (3,)):
                reqs = [dict(segments=s1.student_segments(it["paths"], question(it), views[v]))
                        for v in group for it in batch]
                enc = {k: (v.to(sc.device) if hasattr(v, "to") else v) for k, v in sc.encode(reqs).items()}
                lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
                for gi, v in enumerate(group):
                    last[v] = lg[gi * len(batch):(gi + 1) * len(batch)]
            logits = torch.stack(last, dim=1)                       # [N, 4 views, V]
            gold = torch.tensor([yes if it["answer"] else no for it in batch], device=logits.device)
            qa = F.nll_loss(F.log_softmax(logits[:, 3], dim=-1), gold)
            cand = logits[:, :, [yes, no]]
            py = torch.tensor([[it["p_yes"][v] for v in ("none", "only1", "only2", "both")] for it in batch],
                              device=logits.device)
            aux, kd, inter = paired_evidence_loss(cand, torch.stack([py, 1 - py], -1), HP["epsilon"], HP["kd_weight"],
                                                  HP["interaction_weight"] if args.arm == "Bd" else 0.0,
                                                  HP["huber_delta"])
            stats.update(ce=float(qa), kd=float(kd), inter=float(inter))
            return qa + aux, stats
        # D: state-then-answer with gold facts; Dself: with the model's own readings and its own facts-stated answer
        reqs = [dict(segments=s1.srt_segments(it["paths"], question(it), it["block"], it["phrase"])) for it in batch]
        enc = sc.encode(reqs)
        if args.arm == "Dself":
            tgt = [(tuple(teach[it["item_id"]][1]["read"]), teach[it["item_id"]][1]["p_yes"] > 0.5) for it in batch]
            if args.answer_from_reads:              # E3 contingency arm Dself23p (PREDICTIONS_E3.md, amendment)
                tgt = [(v, e3.answer(it["predicate"], v) if it["predicate"] in e3.OPS
                        else logic.answer(it["predicate"], v[0], v[1])) for it, (v, _) in zip(batch, tgt)]
        else:
            tgt = [(tuple(it.get("vals") or (it["v1"], it["v2"])), it["answer"]) for it in batch]
        tg = [tok.encode(s1.srt_target(it["block"], it["phrase"], v, a), add_special_tokens=False) + [im_end]
              for it, (v, a) in zip(batch, tgt)]
        enc, labels = extend_with_targets(enc, tg, tok.pad_token_id)
        enc = {k: (v.to(sc.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        labels = labels.to(sc.device)
        n_keep = max(len(t) for t in tg) + 1
        o = model(**enc, logits_to_keep=n_keep)
        logits = o.logits[:, :-1, :].float()
        lab = labels[:, -(n_keep - 1):]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1), ignore_index=-100)
        stats["ce_tok"] = float(loss)
        return loss, stats

    t0 = time.time()
    pos = 0
    for step in range(steps):
        agg = {}
        for _ in range(accum):
            batch = [items[(pos + i) % len(items)] for i in range(micro)]
            pos += micro
            loss, st = micro_loss(batch)
            (loss / accum).backward()
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v / accum
        gn = torch.nn.utils.clip_grad_norm_(params, HP["grad_clip"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == steps - 1:
            row = dict(step=step, lr=sched.get_last_lr()[0], grad_norm=float(gn), min=round((time.time() - t0) / 60, 2),
                       **{k: round(v, 5) for k, v in agg.items()})
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    torch.save(dict(lora=s1.lora_state(model), meta=meta), os.path.join(out, "lora.pt"))
    meta["train_minutes"] = round((time.time() - t0) / 60, 1)
    meta["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(sc.device) / 2 ** 30, 1)
    with open(os.path.join(out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("saved", out, json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
