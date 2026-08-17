#!/usr/bin/env python3
"""Create deterministic semantic orderings for every materialized clique."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CLIQUE_ROOT = HERE.parent / "maximal_clique" / "results"
OUTPUT_ROOT = HERE / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")

PHASE_TAGS = {
    "atom": (("load", "copy", "tma", "cp-async"), ("mma", "wgmma", "tensor-core"), ("store", "commit", "wait")),
    "tile": (("shape", "hierarchy", "count"), ("layout", "swizzle", "iterator"), ("alignment", "resource")),
    "collective": (("mainloop", "pipeline", "stages"), ("schedule", "dispatch"), ("epilogue", "fusion", "visitor")),
    "kernel": (("storage", "composition"), ("schedule", "scheduler", "persistent"), ("raster", "swizzle")),
    "device": (("hardware", "feasibility", "occupancy"), ("workspace", "arguments", "params", "initialization"), ("launch", "update", "error")),
}


def priority(level, record_id):
    text = record_id.lower()
    for index, tags in enumerate(PHASE_TAGS[level]):
        if any(tag in text for tag in tags):
            return index, text
    return len(PHASE_TAGS[level]), text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders-per-clique", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    summary = {}
    for architecture in ARCHITECTURES:
        for level in LEVELS:
            source = CLIQUE_ROOT / architecture / f"{level}.json"
            cliques = json.loads(source.read_text(encoding="utf-8"))
            sequences = []
            for clique_index, clique in enumerate(cliques["maximal_cliques"]):
                forward = sorted(clique, key=lambda node: priority(level, cliques["record_ids"][node]))
                orders = [forward]
                if args.orders_per_clique == 2 and len(forward) > 1 and list(reversed(forward)) != forward:
                    orders.append(list(reversed(forward)))
                for order_index, order in enumerate(orders):
                    sequences.append({"id": f"sequence:{clique_index}:{order_index}", "clique_index": clique_index, "order": order, "ordering": "semantic-forward" if order_index == 0 else "semantic-reverse"})
            document = {
                "schema_version": "1.0.0", "architecture": architecture, "level": level,
                "sequence_set_id": f"{cliques['clique_set_id']}:ordered-v1",
                "source": {"clique_set_id": cliques["clique_set_id"], "path": source.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                "record_ids": cliques["record_ids"], "sequences": sequences,
                "statistics": {"clique_count": len(cliques["maximal_cliques"]), "sequence_count": len(sequences), "orders_per_clique_limit": args.orders_per_clique, "clique_enumeration_complete": cliques["statistics"]["enumeration_complete"]},
            }
            output = OUTPUT_ROOT / architecture / f"{level}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
            summary[f"{architecture}/{level}"] = document["statistics"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
