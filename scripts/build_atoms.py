"""Build atom tables. Usage: python scripts/build_atoms.py --source clevr|coco"""
import argparse
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from e0lib.io import load_yaml, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["clevr", "coco"])
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/sources.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    scfg = cfg[args.source]
    if args.source == "clevr":
        from e0lib import atoms_clevr
        rows = atoms_clevr.build(scfg)
        report = atoms_clevr.prevalence_report(rows)
    elif args.source == "coco":
        from e0lib import atoms_coco
        need = [scfg["instances_json"].format(split=s) for s in scfg["splits"]] + \
               [scfg["captions_json"].format(split=s) for s in scfg["splits"]]
        if not all(os.path.exists(p) for p in need):
            with zipfile.ZipFile(scfg["annotations_zip"]) as z:
                z.extractall(os.path.dirname(scfg["annotations_dir"]))
        rows = atoms_coco.build(scfg, seed=cfg["seed"])
        report = [r for r in rows if r.get("report")]
        rows = [r for r in rows if not r.get("report")]
    n = write_jsonl(os.path.join(args.out, f"atoms_{args.source}.jsonl"), rows)
    with open(os.path.join(args.out, f"atoms_{args.source}_report.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"{args.source}: {n} atom rows")
    print(json.dumps(report, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
