#!/usr/bin/env python3
"""Validate semantic completeness and dependency consistency of analyses."""

import json
from pathlib import Path

HERE = Path(__file__).parent
ALLOWED_STAGES = {"input_data_movement", "accumulation", "epilogue"}


def main():
    files = sorted((HERE / "results").glob("*.json"))
    if not files:
        raise ValueError("no variant-analysis results")
    groups = operations = 0
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["status"] != "analyzed" or item["next_stage"] != "patch_localization":
            raise ValueError(f"invalid analysis state: {path}")
        op_ids = [op["id"] for op in item["variant_operations"]]
        if len(op_ids) != len(set(op_ids)):
            raise ValueError(f"duplicate operation: {path}")
        assigned = []
        for group in item["patch_groups"]:
            if group["availability_stage"] not in ALLOWED_STAGES or group["localization_status"] != "pending":
                raise ValueError(f"invalid group stage: {path}")
            if not group["operation_ids"] or not set(group["operation_ids"]).issubset(op_ids):
                raise ValueError(f"invalid group membership: {path}")
            assigned.extend(group["operation_ids"])
        if sorted(assigned) != sorted(op_ids):
            raise ValueError(f"variant operations are not partitioned exactly once: {path}")
        indices = [op["topological_index"] for op in item["variant_operations"]]
        if indices != sorted(indices):
            raise ValueError(f"operations are not topologically ordered: {path}")
        groups += len(item["patch_groups"])
        operations += len(op_ids)
    print(f"Validated {len(files)} variant analyses containing {operations} variant operations and {groups} patch groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

