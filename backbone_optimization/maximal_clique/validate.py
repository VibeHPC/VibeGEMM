#!/usr/bin/env python3
"""Validate clique membership, pair compatibility, maximality, and uniqueness."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
GRAPH_ROOT = ROOT / "optimization_graph" / "compatible_level_graph" / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def main():
    total = 0
    for architecture in ARCHITECTURES:
        for level in LEVELS:
            graph_path = GRAPH_ROOT / architecture / f"{level}.json"
            clique_path = HERE / "results" / architecture / f"{level}.json"
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            result = json.loads(clique_path.read_text(encoding="utf-8"))
            if result["record_ids"] != graph["record_ids"] or result["source"]["sha256"] != hashlib.sha256(graph_path.read_bytes()).hexdigest():
                raise ValueError(f"stale clique source: {architecture}/{level}")
            adjacency = [set() for _ in graph["record_ids"]]
            for left, right in graph["compatible_edges"]:
                adjacency[left].add(right); adjacency[right].add(left)
            seen = set()
            for clique in result["maximal_cliques"]:
                key = tuple(clique)
                if key in seen or clique != sorted(set(clique)):
                    raise ValueError(f"duplicate/invalid clique: {architecture}/{level}")
                seen.add(key)
                members = set(clique)
                if any((members - {node}) - adjacency[node] for node in members):
                    raise ValueError(f"non-clique result: {architecture}/{level}")
                if any(all(candidate in adjacency[node] for node in members) for candidate in set(range(len(adjacency))) - members):
                    raise ValueError(f"non-maximal clique: {architecture}/{level}")
            total += len(seen)
    print(f"Validated 15 maximal-clique sets containing {total} cliques.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
