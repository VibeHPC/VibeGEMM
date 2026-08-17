#!/usr/bin/env python3
"""Validate that every order is a permutation of its source maximal clique."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CLIQUE_ROOT = HERE.parent / "maximal_clique" / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def main():
    total = 0
    for architecture in ARCHITECTURES:
        for level in LEVELS:
            clique_path = CLIQUE_ROOT / architecture / f"{level}.json"
            sequence_path = HERE / "results" / architecture / f"{level}.json"
            cliques = json.loads(clique_path.read_text(encoding="utf-8"))
            result = json.loads(sequence_path.read_text(encoding="utf-8"))
            if result["record_ids"] != cliques["record_ids"] or result["source"]["sha256"] != hashlib.sha256(clique_path.read_bytes()).hexdigest():
                raise ValueError(f"stale sequence source: {architecture}/{level}")
            seen = set()
            for sequence in result["sequences"]:
                clique = cliques["maximal_cliques"][sequence["clique_index"]]
                if sorted(sequence["order"]) != clique or tuple(sequence["order"]) in seen:
                    raise ValueError(f"invalid/duplicate ordering: {architecture}/{level}")
                seen.add(tuple(sequence["order"]))
            total += len(result["sequences"])
    print(f"Validated 15 ordered-sequence sets containing {total} sequences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
