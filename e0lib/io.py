from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, Iterator, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_id(*parts: Any, n: int = 16) -> str:
    s = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def in_subsample(key: str, frac: float, salt: str = "e0-v2") -> bool:
    """Deterministic subsample membership from a hash, independent of row order."""
    if frac >= 1.0:
        return True
    h = int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < frac


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml  # local import keeps pure-logic modules dependency free
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
