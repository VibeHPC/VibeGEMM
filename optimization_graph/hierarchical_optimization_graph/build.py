#!/usr/bin/env python3
"""Assemble five compatible level graphs into the paper's hierarchy."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
LEVEL_GRAPH_ROOT = HERE.parent / "compatible_level_graph" / "results"
OUTPUT_ROOT = HERE / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(architecture):
    levels = {}
    sources = {}
    node_memberships = edge_count = 0
    for level in LEVELS:
        path = LEVEL_GRAPH_ROOT / architecture / f"{level}.json"
        graph = load(path)
        if graph["architecture"] != architecture or graph["level"] != level:
            raise ValueError(f"level graph scope mismatch: {path}")
        levels[level] = {
            "graph_id": graph["graph_id"],
            "record_ids": graph["record_ids"],
            "edge_encoding": graph["edge_encoding"],
            "compatible_edges": graph["compatible_edges"],
            "statistics": graph["statistics"],
        }
        sources[level] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
        }
        node_memberships += graph["statistics"]["node_count"]
        edge_count += graph["statistics"]["compatible_edge_count"]
    return {
        "schema_version": "1.0.0",
        "hierarchy_id": f"{architecture}:hierarchical-optimization-graph-v1",
        "architecture": architecture,
        "definition": "G = {G_atom, G_tile, G_collective, G_kernel, G_device}",
        "level_order": list(LEVELS),
        "semantics": {
            "level_graphs_are_independent": True,
            "edges_are_same_level_undirected_compatibility": True,
            "cross_level_edges": False,
            "transitive_compatibility_inference": False,
            "maximal_cliques_materialized": False,
        },
        "sources": sources,
        "levels": levels,
        "statistics": {
            "level_graph_count": len(LEVELS),
            "node_membership_count": node_memberships,
            "compatible_edge_count": edge_count,
            "nodes_by_level": {level: levels[level]["statistics"]["node_count"] for level in LEVELS},
            "edges_by_level": {level: levels[level]["statistics"]["compatible_edge_count"] for level in LEVELS},
        },
    }


def validate(graph):
    if graph["level_order"] != list(LEVELS) or set(graph["levels"]) != set(LEVELS):
        raise ValueError("hierarchy must contain exactly the five ordered levels")
    if not graph["semantics"]["level_graphs_are_independent"] or graph["semantics"]["cross_level_edges"]:
        raise ValueError("invalid hierarchical graph semantics")
    for level in LEVELS:
        level_graph = graph["levels"][level]
        count = len(level_graph["record_ids"])
        seen = set()
        for edge in level_graph["compatible_edges"]:
            key = tuple(edge)
            if len(edge) != 2 or not (0 <= edge[0] < edge[1] < count) or key in seen:
                raise ValueError(f"invalid {level} edge: {edge}")
            seen.add(key)
        if len(seen) != level_graph["statistics"]["compatible_edge_count"]:
            raise ValueError(f"{level} edge count mismatch")


def main():
    summary = {}
    for architecture in ARCHITECTURES:
        graph = build(architecture)
        validate(graph)
        output = OUTPUT_ROOT / f"{architecture}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(graph, separators=(",", ":")) + "\n", encoding="utf-8")
        summary[architecture] = graph["statistics"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
