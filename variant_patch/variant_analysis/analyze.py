#!/usr/bin/env python3
"""Normalize a target GEMM variant into semantic, dependency-aware patch groups."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
BACKBONE_KINDS = {"gemm", "dual_gemm", "grouped_gemm"}
INPUT_KINDS = {"dequantize", "input_scale", "input_permute", "unpack"}
ACCUMULATION_KINDS = {"accumulator_transform", "partial_reduction"}


def stable_id(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def topological_order(operations):
    by_id = {item["id"]: item for item in operations}
    if len(by_id) != len(operations):
        raise ValueError("operation IDs must be unique")
    producer = {}
    for operation in operations:
        for value in operation["outputs"]:
            if value in producer:
                raise ValueError(f"multiple producers for value: {value}")
            producer[value] = operation["id"]
    dependencies = {
        item["id"]: {producer[value] for value in item["inputs"] if value in producer}
        for item in operations
    }
    ordered, ready = [], sorted(key for key, deps in dependencies.items() if not deps)
    while ready:
        current = ready.pop(0)
        ordered.append(by_id[current])
        for key in sorted(dependencies):
            if current in dependencies[key]:
                dependencies[key].remove(current)
                if not dependencies[key] and key not in {item["id"] for item in ordered} and key not in ready:
                    ready.append(key)
                    ready.sort()
    if len(ordered) != len(operations):
        raise ValueError("operation graph contains a cycle")
    return ordered, producer


def stage_of(operation, producer, by_id):
    if operation.get("stage_hint"):
        return operation["stage_hint"], "explicit stage hint from target specification"
    if operation["kind"] in INPUT_KINDS:
        return "input_data_movement", "transforms a GEMM operand before it is consumed"
    if operation["kind"] in ACCUMULATION_KINDS:
        return "accumulation", "requires access to partial or live accumulators"
    parent_kinds = {by_id[producer[value]]["kind"] for value in operation["inputs"] if value in producer}
    if parent_kinds & BACKBONE_KINDS:
        return "epilogue", "its operand becomes available after GEMM accumulation"
    return "epilogue", "depends on post-GEMM variant values"


def connected_groups(variant_ops, producer, stage_by_id):
    ids = {item["id"] for item in variant_ops}
    adjacency = {item["id"]: set() for item in variant_ops}
    for item in variant_ops:
        for value in item["inputs"]:
            parent = producer.get(value)
            if parent in ids and stage_by_id[parent] == stage_by_id[item["id"]]:
                adjacency[parent].add(item["id"])
                adjacency[item["id"]].add(parent)
    groups, unseen = [], set(ids)
    while unseen:
        seed, component = min(unseen), set()
        stack = [seed]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        unseen -= component
        groups.append(component)
    return groups


def analyze(target):
    required = {"variant_id", "name", "supported_architectures", "problem", "tensors", "operations", "outputs", "reference"}
    missing = sorted(required - target.keys())
    if missing:
        raise ValueError(f"{target.get('variant_id', '<unknown>')}: missing {missing}")
    ordered, producer = topological_order(target["operations"])
    by_id = {item["id"]: item for item in ordered}
    backbone = [item for item in ordered if item["kind"] in BACKBONE_KINDS]
    if not backbone:
        raise ValueError(f"{target['variant_id']}: no GEMM backbone operation")
    variant_ops = [item for item in ordered if item["kind"] not in BACKBONE_KINDS]
    stages = {item["id"]: stage_of(item, producer, by_id) for item in variant_ops}
    tensor_map = {item["name"]: item for item in target["tensors"]}
    groups = []
    for component in connected_groups(variant_ops, producer, {key: value[0] for key, value in stages.items()}):
        operations = [item for item in ordered if item["id"] in component]
        produced = {value for item in operations for value in item["outputs"]}
        consumed = {value for item in operations for value in item["inputs"]}
        external_inputs = sorted(consumed - produced)
        external_outputs = sorted(value for value in produced if value in target["outputs"] or any(value in item["inputs"] for item in ordered if item["id"] not in component))
        stage = stages[operations[0]["id"]][0]
        identity = {"variant": target["variant_id"], "stage": stage, "operations": [item["id"] for item in operations]}
        groups.append({
            "group_id": f"patch-group:{stable_id(identity)}",
            "operation_ids": [item["id"] for item in operations],
            "computation": " -> ".join(item.get("expression", item["kind"]) for item in operations),
            "inputs": external_inputs,
            "outputs": external_outputs,
            "tensor_requirements": [tensor_map[value] for value in external_inputs + external_outputs if value in tensor_map],
            "availability_stage": stage,
            "stage_rationale": stages[operations[0]["id"]][1],
            "requires_reduction": any(item["kind"] in {"reduce_mean", "reduce_sum", "rmsnorm"} for item in operations),
            "localization_status": "pending",
        })
    reference = dict(target["reference"])
    if reference.get("path"):
        reference_path = ROOT / reference["path"]
        reference["sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest() if reference_path.is_file() else None
        reference["available"] = reference_path.is_file()
    document = {
        "schema_version": "1.0.0",
        "analysis_id": f"variant-analysis:{stable_id(target)}",
        "variant_id": target["variant_id"],
        "name": target["name"],
        "supported_architectures": target["supported_architectures"],
        "problem": target["problem"],
        "reference": reference,
        "backbone_operations": [item["id"] for item in backbone],
        "variant_operations": [{**item, "topological_index": ordered.index(item)} for item in variant_ops],
        "data_dependencies": [{"value": value, "producer": parent, "consumers": [item["id"] for item in ordered if value in item["inputs"]]} for value, parent in sorted(producer.items())],
        "patch_groups": groups,
        "outputs": target["outputs"],
        "status": "analyzed",
        "next_stage": "patch_localization",
    }
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "examples" / "representative_variants.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    source = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = json.loads(source.read_text(encoding="utf-8"))
    targets = payload["variants"] if isinstance(payload, dict) and "variants" in payload else [payload]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for target in targets:
        result = analyze(target)
        output = output_dir / f"{target['variant_id']}.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summary[target["variant_id"]] = {"variant_operations": len(result["variant_operations"]), "patch_groups": len(result["patch_groups"])}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
