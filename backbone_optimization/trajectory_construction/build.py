#!/usr/bin/env python3
"""Concatenate one ordered sequence from each level into candidate trajectories."""

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SEQUENCE_ROOT = HERE.parent / "ordered_sequence" / "results"
HIERARCHY_ROOT = ROOT / "optimization_graph" / "hierarchical_optimization_graph" / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def coprime_stride(total, budget):
    stride = max(1, total // max(1, budget))
    while math.gcd(stride, total) != 1:
        stride += 1
    return stride


def digits(value, radices):
    output = []
    for radix in reversed(radices):
        output.append(value % radix); value //= radix
    return list(reversed(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-budget", type=int, default=1000)
    args = parser.parse_args()
    summary = {}
    output_root = HERE / "results"
    for architecture in ARCHITECTURES:
        hierarchy_path = HIERARCHY_ROOT / f"{architecture}.json"
        hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
        sets, sources = {}, {}
        for level in LEVELS:
            path = SEQUENCE_ROOT / architecture / f"{level}.json"
            sets[level] = json.loads(path.read_text(encoding="utf-8"))
            sources[level] = {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        radices = [len(sets[level]["sequences"]) for level in LEVELS]
        total_space = math.prod(radices)
        count = min(args.trajectory_budget, total_space)
        stride = coprime_stride(total_space, count)
        trajectories = []
        for rank in range(count):
            indices = digits((rank * stride) % total_space, radices)
            level_sequences, steps = {}, []
            for level, sequence_index in zip(LEVELS, indices):
                sequence = sets[level]["sequences"][sequence_index]
                level_sequences[level] = sequence["id"]
                for position, record_index in enumerate(sequence["order"]):
                    steps.append({"level": level, "record_id": sets[level]["record_ids"][record_index], "sequence_id": sequence["id"], "position_in_level": position, "cross_level_compatibility": "unverified"})
            identity = "|".join(step["record_id"] for step in steps)
            trajectories.append({"id": "trajectory:" + hashlib.sha256(identity.encode()).hexdigest()[:16], "rank": rank + 1, "level_sequences": level_sequences, "steps": steps, "status": "proposed"})
        document = {
            "schema_version": "1.0.0", "architecture": architecture,
            "trajectory_set_id": f"{hierarchy['hierarchy_id']}:compatible-trajectories-v1",
            "source": {"hierarchy": {"path": hierarchy_path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(hierarchy_path.read_bytes()).hexdigest()}, "ordered_sequences": sources},
            "level_order": list(LEVELS), "cross_level_compatibility": "unverified",
            "trajectories": trajectories,
            "statistics": {"sequence_counts_by_level": dict(zip(LEVELS, radices)), "trajectory_combination_space": total_space, "trajectory_budget": args.trajectory_budget, "trajectory_count": len(trajectories), "combination_complete": len(trajectories) == total_space, "selection": "deterministic coprime-stride sampling"},
        }
        output = output_root / f"{architecture}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
        summary[architecture] = document["statistics"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
