#!/usr/bin/env python3
"""Validate hierarchical graphs against all five source level graphs."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
LEVEL_ROOT = HERE.parent / "compatible_level_graph" / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def main():
    total_memberships = total_edges = 0
    for architecture in ARCHITECTURES:
        hierarchy = json.loads((HERE / "results" / f"{architecture}.json").read_text(encoding="utf-8"))
        if hierarchy["architecture"] != architecture or hierarchy["level_order"] != list(LEVELS):
            raise ValueError(f"invalid hierarchy scope/order: {architecture}")
        if set(hierarchy["levels"]) != set(LEVELS) or hierarchy["semantics"]["cross_level_edges"]:
            raise ValueError(f"invalid hierarchy semantics: {architecture}")
        for level in LEVELS:
            source_path = LEVEL_ROOT / architecture / f"{level}.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            expected = {
                "graph_id": source["graph_id"], "record_ids": source["record_ids"],
                "edge_encoding": source["edge_encoding"], "compatible_edges": source["compatible_edges"],
                "statistics": source["statistics"],
            }
            if hierarchy["levels"][level] != expected:
                raise ValueError(f"embedded level differs from source: {architecture}/{level}")
            if hierarchy["sources"][level]["sha256"] != hashlib.sha256(source_path.read_bytes()).hexdigest():
                raise ValueError(f"stale source hash: {architecture}/{level}")
            total_memberships += source["statistics"]["node_count"]
            total_edges += source["statistics"]["compatible_edge_count"]
    print(f"Validated 3 hierarchical optimization graphs: levels=15, node memberships={total_memberships}, compatible edges={total_edges}, cross-level edges=0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
