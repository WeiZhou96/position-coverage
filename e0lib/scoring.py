"""Logit scoring of answer candidates at the start of the assistant reply (no generation).

Each request carries `segments` [(kind, value)], `cands` (answer strings) and `sems`.
For every request we store the full-vocabulary log-probabilities of the first token of each
candidate, their normalised probabilities, the covered probability mass and the number of
visual tokens of each image.
"""
from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from typing import Dict, List, Sequence

import torch
from PIL import Image


def _host_staged_cross_device_copies():
    """Route every CUDA -> other-CUDA tensor move through host memory.

    On some multi-GPU machines the driver reports peer access between GPU pairs, but peer-to-peer copies return
    corrupted data. Models split over several GPUs (device_map) move activations between GPUs
    with Tensor.to / copy_; staging them through the CPU keeps the values exact. Single-GPU runs never call this.
    """
    if getattr(torch.Tensor.to, "_host_staged", False):
        return
    orig_to, orig_copy = torch.Tensor.to, torch.Tensor.copy_

    def to(self, *args, **kwargs):
        if self.is_cuda:
            dev = torch._C._nn._parse_to(*args, **kwargs)[0]
            if dev is not None and dev.type == "cuda":
                idx = dev.index if dev.index is not None else torch.cuda.current_device()
                if idx != self.device.index:
                    return orig_to(orig_to(self, "cpu"), *args, **kwargs)
        return orig_to(self, *args, **kwargs)

    def copy_(self, src, non_blocking=False):
        if self.is_cuda and src.is_cuda and self.device != src.device:
            src = orig_to(src, "cpu")
        return orig_copy(self, src, non_blocking)

    to._host_staged = True
    torch.Tensor.to = to
    torch.Tensor.copy_ = copy_


class Scorer:
    def __init__(self, cfg: Dict, device: str = "cuda:0"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.cfg = cfg
        self.device = device
        pk = dict(cfg.get("processor_kwargs") or {})
        self.processor = AutoProcessor.from_pretrained(cfg["path"], **pk)
        tok = getattr(self.processor, "tokenizer", self.processor)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tok = tok
        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        if cfg.get("device_map"):     # P5: large models split over the visible GPUs (CUDA_VISIBLE_DEVICES)
            _host_staged_cross_device_copies()
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg["path"], dtype=dtype, attn_implementation=cfg.get("attn", "sdpa"),
                device_map=cfg["device_map"]).eval()
            self.device = device = str(self.model.get_input_embeddings().weight.device)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg["path"], dtype=dtype, attn_implementation=cfg.get("attn", "sdpa")).to(device).eval()
        self.template_kwargs = dict(cfg.get("chat_template_kwargs") or {})
        self.image_token_id = self._image_token_id()
        self._cand_cache: Dict[str, int] = {}
        self.max_image_side = cfg.get("max_image_side")
        # decoded + resized images are reused by ~30 requests each; cache them (speed only)
        self._img_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self._img_cache_max = int(cfg.get("image_cache", 6000))
        self._img_lock = threading.Lock()
        self._tok_lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _image_token_id(self):
        for attr in ("image_token_id", "image_token_index"):
            v = getattr(self.processor, attr, None) or getattr(self.model.config, attr, None)
            if isinstance(v, int):
                return v
        tokname = getattr(self.processor, "image_token", None)
        if tokname:
            return self.tok.convert_tokens_to_ids(tokname)
        return None

    def cand_token(self, cand: str) -> int:
        """First token id of a candidate as it would appear at the start of the reply."""
        if cand not in self._cand_cache:
            ids = self.tok.encode(cand, add_special_tokens=False)
            self._cand_cache[cand] = ids[0]
            self._cand_cache[cand + "#ntok"] = len(ids)
        return self._cand_cache[cand]

    def check_candidates(self, cands: Sequence[str]) -> Dict[str, int]:
        return {c: (self.cand_token(c), self._cand_cache[c + "#ntok"]) for c in cands}

    def _load(self, path: str) -> Image.Image:
        with self._img_lock:
            im = self._img_cache.get(path)
            if im is not None:
                self._img_cache.move_to_end(path)
                return im
        im = Image.open(path).convert("RGB")
        if self.max_image_side and max(im.size) > self.max_image_side:
            s = self.max_image_side / max(im.size)
            im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BICUBIC)
        with self._img_lock:
            self._img_cache[path] = im
            if len(self._img_cache) > self._img_cache_max:
                self._img_cache.popitem(last=False)
        return im

    def _messages(self, segments) -> List[Dict]:
        content = []
        for kind, val in segments:
            if kind == "text":
                if content and content[-1]["type"] == "text":
                    content[-1]["text"] += val
                else:
                    content.append({"type": "text", "text": val})
            else:
                content.append({"type": "image", "image": self._load(val)})
        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------ scoring
    def encode(self, reqs: List[Dict]):
        """CPU side (image loading + processor); safe to run in a background thread."""
        msgs = [self._messages(r["segments"]) for r in reqs]
        with self._tok_lock:   # Rust tokenizers are not safe under concurrent padding changes
            return self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                padding=True, **self.template_kwargs)

    @torch.inference_mode()
    def score_batch(self, reqs: List[Dict], enc=None) -> List[Dict]:
        if enc is None:
            enc = self.encode(reqs)
        enc = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        out = self.model(**enc)
        logits = out.logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        res = []
        ids = enc["input_ids"]
        with self._tok_lock:
            tops = [self.tok.decode([int(t)]) for t in torch.argmax(logits, dim=-1).tolist()]
            cids = [[self.cand_token(c) for c in r["cands"]] for r in reqs]
        for b, r in enumerate(reqs):
            cid = cids[b]
            lp = [float(logp[b, i]) for i in cid]
            m = max(lp)
            z = sum(math.exp(x - m) for x in lp)
            probs = [math.exp(x - m) / z for x in lp]
            n_vis = int((ids[b] == self.image_token_id).sum()) if self.image_token_id is not None else -1
            res.append(dict(rid=r["rid"], logp=lp, probs=probs, mass=float(sum(math.exp(x) for x in lp)),
                            top_token=tops[b],
                            n_visual_tokens=n_vis, n_images=sum(1 for k, _ in r["segments"] if k == "image"),
                            input_sha=hashlib.sha256(ids[b][enc["attention_mask"][b].bool()].cpu().numpy().tobytes()).hexdigest()[:16]))
        return res

    def render_text(self, req: Dict) -> str:
        """Prompt text after the chat template, for inspection."""
        msgs = self._messages(req["segments"])
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                  **self.template_kwargs)
