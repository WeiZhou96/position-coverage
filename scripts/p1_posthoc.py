"""P1 post-hoc breakdown (not pre-registered): ALL accuracy on B1 (one image missing) by the missing slot, and on F
(all present), per arm (mean of seeds), n = 3, 4, 5, 6. Output: outputs/p1/posthoc_slots.md"""
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib.io import iter_jsonl  # noqa: E402

ARMS = ["A", "A23", "A3edge", "A24", "A23q"]


def main():
    req = {}
    for r in iter_jsonl(os.path.join(ROOT, "data", "e3", "eval_requests.jsonl")):
        if r["set"] == "e3_grid" and r["mode"] == "logit" and r["predicate"] == "ALL" and r["pattern"] in ("B1", "F"):
            r.pop("segments", None)
            req[r["rid"]] = r
    lines = ["# P1 post hoc: ALL accuracy by missing slot (B1) and all-present (F); arm = mean of seeds", "",
             "| arm | n | B1 by missing slot 1..n | F |", "|---|---|---|---|"]
    for arm in ARMS:
        acc = defaultdict(list)
        for s in (1, 2):
            for o in iter_jsonl(os.path.join(ROOT, "outputs", "s1", "runs", f"{arm}_s{s}", "eval_e3_logit.jsonl")):
                r = req.get(o["rid"])
                if r is None:
                    continue
                c = float((o["probs"][0] > 0.5) == r["answer"])
                key = (r["n"], "F") if r["pattern"] == "F" else (r["n"], r["vals"].index(0) + 1)
                acc[key].append(c)
        for n in (3, 4, 5, 6):
            b1 = " ".join("%.2f" % (sum(acc[(n, k)]) / len(acc[(n, k)])) for k in range(1, n + 1))
            f = sum(acc[(n, "F")]) / len(acc[(n, "F")])
            lines.append(f"| {arm} | {n} | {b1} | {f:.2f} |")
    os.makedirs(os.path.join(ROOT, "outputs", "p1"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "p1", "posthoc_slots.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
