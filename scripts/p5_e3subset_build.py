"""P5a evaluation subset of the E3 grid (direct answering; rids identical to data/e3/eval_requests.jsonl).

Large models are split over two GPUs and evaluate the full E3 grid too slowly, so P5a uses only what its metrics need:
  ALL, patterns F and B1, n = 4, 5, 6, every rotation            (position metric INT_46 / EDGE_46)
  n = 6, every operator and pattern, rotations 0, 2, 4             (aggregation gap AGG_6; ALL F/B1 at n = 6 keep all)
  all single-image readings (e3_single)                            (ideal-integration reference)
The 8-12B checkpoints are scored on the same subset from their existing full E3 outputs.
Output: data/p5e3/eval_requests.jsonl. Usage: python scripts/p5_e3subset_build.py
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib.io import iter_jsonl  # noqa: E402


def keep(r):
    if r["mode"] != "logit":
        return False
    if r["set"] == "e3_single":
        return True
    if r["predicate"] == "ALL" and r["pattern"] in ("F", "B1") and r["n"] in (4, 5, 6):
        return True
    return r["n"] == 6 and r["rot"] in (0, 2, 4)


def main():
    rows = [r for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")) if keep(r)]
    os.makedirs(os.path.join(ROOT, "data", "p5e3"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "p5e3", "eval_requests.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), Counter((r["set"], r.get("n")) for r in rows))


if __name__ == "__main__":
    main()
