#!/usr/bin/env python3
"""Build one undirected compatibility graph for every architecture and level."""

import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).parent
OPTIMIZATION_GRAPH = HERE.parent
RESULT_ROOT = OPTIMIZATION_GRAPH / "compatibility_analysis" / "results"
OUTPUT_ROOT = HERE / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_graph(result_path):
    analysis = json.loads(result_path.read_text(encoding="utf-8"))
    record_ids = analysis["record_ids"]
    incompatible = {(pair[0], pair[1]) for pair in analysis["incompatible_pairs"]}
    edges = [list(pair) for pair in itertools.combinations(range(len(record_ids)), 2) if pair not in incompatible]
    return {
        "schema_version": "1.0.0",
        "graph_id": f'{analysis["architecture"]}:{analysis["level"]}:compatible-level-graph-v1',
        "architecture": analysis["architecture"],
        "level": analysis["level"],
        "source": {
            "compatibility_analysis_id": analysis["analysis_id"],
            "compatibility_result": result_path.relative_to(OPTIMIZATION_GRAPH.parent).as_posix(),
            "compatibility_result_sha256": sha256(result_path),
            "policy": "Undirected complete graph over record_ids minus incompatible_pairs.",
        },
        "record_ids": record_ids,
        "edge_encoding": "[left_record_index, right_record_index]",
        "compatible_edges": edges,
        "statistics": {
            "node_count": len(record_ids),
            "possible_edge_count": len(record_ids) * (len(record_ids) - 1) // 2,
            "compatible_edge_count": len(edges),
            "incompatible_pair_count": len(incompatible),
        },
    }


def validate(graph):
    node_count = len(graph["record_ids"])
    if len(set(graph["record_ids"])) != node_count:
        raise ValueError(f'duplicate node in {graph["graph_id"]}')
    seen = set()
    for edge in graph["compatible_edges"]:
        key = tuple(edge)
        if len(edge) != 2 or not (0 <= edge[0] < edge[1] < node_count) or key in seen:
            raise ValueError(f'invalid edge {edge} in {graph["graph_id"]}')
        seen.add(key)
    stats = graph["statistics"]
    if stats["compatible_edge_count"] != len(seen):
        raise ValueError(f'edge statistics mismatch in {graph["graph_id"]}')
    if stats["compatible_edge_count"] + stats["incompatible_pair_count"] != stats["possible_edge_count"]:
        raise ValueError(f'graph is not the exact compatibility complement in {graph["graph_id"]}')


def main():
    summary = {}
    for architecture in ARCHITECTURES:
        for level in LEVELS:
            source = RESULT_ROOT / architecture / f"{level}.json"
            graph = build_graph(source)
            validate(graph)
            output = OUTPUT_ROOT / architecture / f"{level}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(graph, separators=(",", ":")) + "\n", encoding="utf-8")
            summary[f"{architecture}/{level}"] = graph["statistics"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
