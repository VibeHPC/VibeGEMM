#!/usr/bin/env python3
"""Validate level order, source sequences, hashes, and trajectory uniqueness."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SEQUENCE_ROOT = HERE.parent / "ordered_sequence" / "results"
HIERARCHY_ROOT = ROOT / "optimization_graph" / "hierarchical_optimization_graph" / "results"
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def main():
    total = 0
    for architecture in ("sm80", "sm90", "sm90a"):
        result = json.loads((HERE / "results" / f"{architecture}.json").read_text(encoding="utf-8"))
        hierarchy_path = HIERARCHY_ROOT / f"{architecture}.json"
        if result["source"]["hierarchy"]["sha256"] != hashlib.sha256(hierarchy_path.read_bytes()).hexdigest():
            raise ValueError(f"stale hierarchy source: {architecture}")
        sequence_sets, sequence_maps = {}, {}
        for level in LEVELS:
            path = SEQUENCE_ROOT / architecture / f"{level}.json"
            sequence_sets[level] = json.loads(path.read_text(encoding="utf-8"))
            if result["source"]["ordered_sequences"][level]["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"stale trajectory source: {architecture}/{level}")
            sequence_maps[level] = {item["id"]: item for item in sequence_sets[level]["sequences"]}
        seen = set()
        for trajectory in result["trajectories"]:
            identity = tuple(step["record_id"] for step in trajectory["steps"])
            if identity in seen:
                raise ValueError(f"duplicate trajectory: {architecture}")
            seen.add(identity)
            cursor = 0
            for level in LEVELS:
                sequence_id = trajectory["level_sequences"][level]
                sequence = sequence_maps[level].get(sequence_id)
                if sequence is None:
                    raise ValueError(f"unknown sequence {architecture}/{level}/{sequence_id}")
                expected = [sequence_sets[level]["record_ids"][index] for index in sequence["order"]]
                actual = [step["record_id"] for step in trajectory["steps"][cursor:cursor + len(expected)]]
                if actual != expected or any(step["level"] != level or step["cross_level_compatibility"] != "unverified" for step in trajectory["steps"][cursor:cursor + len(expected)]):
                    raise ValueError(f"invalid trajectory concatenation: {trajectory['id']}")
                cursor += len(expected)
            if cursor != len(trajectory["steps"]):
                raise ValueError(f"extra trajectory steps: {trajectory['id']}")
        total += len(seen)
    print(f"Validated 3 compatible trajectory sets containing {total} trajectories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
