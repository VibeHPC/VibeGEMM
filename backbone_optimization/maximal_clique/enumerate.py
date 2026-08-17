#!/usr/bin/env python3
"""Budgeted Bron-Kerbosch enumeration over every compatible level graph."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
GRAPH_ROOT = ROOT / "optimization_graph" / "compatible_level_graph" / "results"
OUTPUT_ROOT = HERE / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def enumerate_cliques(graph, budget):
    count = len(graph["record_ids"])
    adjacency = [set() for _ in range(count)]
    for left, right in graph["compatible_edges"]:
        adjacency[left].add(right)
        adjacency[right].add(left)
    cliques = []
    truncated = False

    def visit(current, candidates, excluded):
        nonlocal truncated
        if len(cliques) >= budget:
            truncated = True
            return
        if not candidates and not excluded:
            cliques.append(sorted(current))
            return
        pivot = max(candidates | excluded, key=lambda node: (len(candidates & adjacency[node]), -node), default=None)
        extension = sorted(candidates - (adjacency[pivot] if pivot is not None else set()))
        for node in extension:
            visit(current | {node}, candidates & adjacency[node], excluded & adjacency[node])
            candidates.remove(node)
            excluded.add(node)
            if truncated:
                return

    visit(set(), set(range(count)), set())
    cliques.sort(key=lambda clique: (-len(clique), clique))
    return cliques, truncated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clique-budget", type=int, default=10000)
    args = parser.parse_args()
    if args.clique_budget <= 0:
        raise ValueError("clique budget must be positive")
    summary = {}
    for architecture in ARCHITECTURES:
        for level in LEVELS:
            source = GRAPH_ROOT / architecture / f"{level}.json"
            graph = json.loads(source.read_text(encoding="utf-8"))
            cliques, truncated = enumerate_cliques(graph, args.clique_budget)
            document = {
                "schema_version": "1.0.0", "architecture": architecture, "level": level,
                "clique_set_id": f"{graph['graph_id']}:maximal-cliques-v1",
                "source": {"graph_id": graph["graph_id"], "path": source.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                "record_ids": graph["record_ids"], "clique_encoding": "array of record indices",
                "maximal_cliques": cliques,
                "statistics": {"node_count": len(graph["record_ids"]), "clique_count": len(cliques), "largest_clique_size": max(map(len, cliques), default=0), "clique_budget": args.clique_budget, "enumeration_complete": not truncated},
            }
            output = OUTPUT_ROOT / architecture / f"{level}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
            summary[f"{architecture}/{level}"] = document["statistics"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
