#!/usr/bin/env python3
"""Validate all materialized compatible level graphs against their sources."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
RESULT_ROOT = HERE / "results"
ANALYSIS_ROOT = HERE.parent / "compatibility_analysis" / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def main():
    total_nodes = total_edges = 0
    for architecture in ARCHITECTURES:
        for level in LEVELS:
            analysis_path = ANALYSIS_ROOT / architecture / f"{level}.json"
            graph_path = RESULT_ROOT / architecture / f"{level}.json"
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            if graph["architecture"] != architecture or graph["level"] != level:
                raise ValueError(f"graph scope mismatch: {graph_path}")
            if graph["record_ids"] != analysis["record_ids"]:
                raise ValueError(f"node set mismatch: {graph_path}")
            if graph["source"]["compatibility_result_sha256"] != hashlib.sha256(analysis_path.read_bytes()).hexdigest():
                raise ValueError(f"stale graph source hash: {graph_path}")
            incompatible = {(pair[0], pair[1]) for pair in analysis["incompatible_pairs"]}
            compatible = {tuple(edge) for edge in graph["compatible_edges"]}
            node_count = len(graph["record_ids"])
            if any(not (0 <= left < right < node_count) for left, right in compatible | incompatible):
                raise ValueError(f"edge index is outside the node set: {graph_path}")
            possible = node_count * (node_count - 1) // 2
            if compatible & incompatible or len(compatible) + len(incompatible) != possible:
                raise ValueError(f"graph is not an exact partition: {graph_path}")
            if len(compatible) != len(graph["compatible_edges"]):
                raise ValueError(f"duplicate compatible edge: {graph_path}")
            total_nodes += node_count
            total_edges += len(compatible)
    print(f"Validated 15 compatible level graphs: node memberships={total_nodes}, compatible edges={total_edges}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
