"""Diagnostic probes on the E3 grid (direction re-diagnosis after report 10.13; post hoc, inference only).

Built from data/e3/eval_requests.jsonl (same images, same 'Image k: <image>' layout as the E3 grid):
  dx_ctx   in-context reading: all n images shown, question "Does Image k contain X?" for every slot k.
           Arrangements: patterns A1 and B1 (one odd image), every rotation, n = 2-6. Gives the per-slot reading in
           context, comparable with the isolated readings (e3_single) and with the E3 answers on the same arrangement.
  dx_text  text-only facts: no images; the correct per-image facts as sentences, then the E3 question. Every E3 grid
           item (all n, operators, patterns, rotations). Tests the language-side integration on its own.
Output: data/e3dx/eval_requests.jsonl (evaluated with: python scripts/s1_eval.py --run <run> --data e3dx --modes logit,
which writes eval_e3dx_logit.jsonl into the run directory). Usage: python scripts/dx_e3_build.py
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib import e3, prompts, s1  # noqa: E402
from e0lib.io import iter_jsonl  # noqa: E402

OUT = os.path.join(ROOT, "data", "e3dx")


def rid(*parts):
    return hashlib.md5(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:20]


def main():
    grid = [r for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl"))
            if r["set"] == "e3_grid" and r["mode"] == "logit"]
    reqs = []
    for r in grid:
        paths = [v for k, v in r["segments"] if k == "image"]
        assert len(paths) == r["n"]
        q = r["segments"][-1][1].split("\n")[0]
        X = re.search(r"contains? (.+)\?$", q).group(1)
        assert q == e3.TEMPLATES[r["predicate"]].format(N=e3.NWORD[r["n"]], X=X), q
        phrase = {"singular": X, "plural": X}   # plural unused by the presence templates
        meta = {k: r[k] for k in ("source", "block", "descriptor", "group", "n", "pattern", "rot", "vals",
                                  "image_ids")}
        # text-only facts, every grid item
        facts = "\n".join(s1.fact_lines("presence", phrase, r["vals"]))
        text = f"Facts about the {e3.NWORD[r['n']]} images:\n{facts}\n{q}\n{prompts.INSTR_BIN}"
        reqs.append(dict(rid=rid("dx_text", r["rid"]), set="dx_text", mode="logit", segments=[["text", text]],
                         cands=["Yes", "No"], sems=[1, 0], predicate=r["predicate"], answer=r["answer"],
                         grid_rid=r["rid"], **meta))
        # in-context reading, one arrangement per (group, n, pattern, rot): take the ALL item
        if r["predicate"] == "ALL" and r["pattern"] in ("A1", "B1"):
            for k in range(1, r["n"] + 1):
                segs = s1.image_segments(paths) + [("text", prompts.fill(prompts.PROBE_PRESENCE_CTX[0], phrase, k)
                                                    + f"\n{prompts.INSTR_BIN}")]
                reqs.append(dict(rid=rid("dx_ctx", r["group"], r["n"], r["pattern"], r["rot"], k), set="dx_ctx",
                                 mode="logit", segments=[list(s) for s in segs], cands=["Yes", "No"], sems=[1, 0],
                                 slot=k, value=r["vals"][k - 1], image_id=r["image_ids"][k - 1], **meta))
    assert len({x["rid"] for x in reqs}) == len(reqs)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "eval_requests.jsonl"), "w") as f:
        for x in reqs:
            f.write(json.dumps(x) + "\n")
    print(Counter(x["set"] for x in reqs), Counter((x["set"], x["n"]) for x in reqs if x["set"] == "dx_ctx"))
    print("example ctx:", [x for x in reqs if x["set"] == "dx_ctx"][0]["segments"][-1])
    print("example text:", [x for x in reqs if x["set"] == "dx_text"][100]["segments"][0][1])


if __name__ == "__main__":
    main()
