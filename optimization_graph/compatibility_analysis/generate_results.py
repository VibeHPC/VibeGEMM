#!/usr/bin/env python3
"""Materialize the current LLM review of same-level incompatible strategies."""

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD_ROOT = ROOT / "knowledge" / "records"
OUTPUT = Path(__file__).parent / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def load_records():
    records = []
    for path in sorted(RECORD_ROOT.rglob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        records.append(item)
    return records


def contains(record, *needles):
    text = " ".join((record["id"], record["name"], record["summary"], *record["tags"])).lower()
    return any(needle in text for needle in needles)


def incompatible_reason(level, left, right):
    pair = (left, right)
    if level == "atom":
        if all(contains(item, "mma-sync", "wgmma") for item in pair):
            return "Alternative arithmetic atoms define different operand, accumulator, sparsity, or residency semantics and are not selected together for one GEMM computation."
        if all(contains(item, "tma-load") for item in pair):
            return "Alternative TMA load forms select different ordinary, multicast, or im2col movement semantics for the same operand transfer."
        if all(contains(item, "tma-store") for item in pair):
            return "Alternative TMA store forms select different ordinary or im2col destination semantics for the same transfer."
        cp_variants = ("cp-async-cp-async", "cp-async-cp-async-nan", "cp-async-cp-async-zfill")
        if all(any(token in item["id"] for token in cp_variants) for item in pair):
            return "Alternative cp.async copy forms disagree on the fill behavior of a predicated-off transfer."

    if level == "tile":
        sm80_policies = ("fast-f32-warp-policy", "mixed-input-warp-policy", "sparse-mma-core", "warp-tensorop-policy")
        if all(any(token in item["id"] for token in sm80_policies) for item in pair):
            return "Alternative warp/tile compute policies define mutually exclusive operand and MMA-core configurations."

    if level == "collective":
        mainloop_tokens = (
            "array-cp-async-mainloop", "cp-async-multistage", "pipelined-mainloop",
            "reduction-mainloop", "softmax-mainloop", "sparse-multistage",
            "array-tma-wgmma", "mixed-input-transform", "non-tma-wgmma",
            "fp8-blockwise-scaling", "sparse-metadata-flow", "tma-wgmma-rs", "tma-wgmma-ss",
        )
        if all(any(token in item["id"] for token in mainloop_tokens) for item in pair):
            return "Alternative collective mainloops define different dataflow, operand transformation, sparsity, or WGMMA residency for the same GEMM mainloop."
        dispatch = ("auto-kernel-schedule", "cooperative-dispatch", "mixed-input-dispatch", "pingpong-dispatch", "fp8-fast-accum")
        if all(any(token in item["id"] for token in dispatch) for item in pair):
            return "Alternative dispatch policies assign incompatible consumer roles or kernel schedules."
        epilogue_tokens = (
            "fusion", "linear-combination", "epilogue", "visitor", "direct-store",
            "per-channel-scaling", "scaling-absmax", "broadcast", "relu", "gelu", "silu", "hardswish",
        )
        if all(any(token in item["id"] for token in epilogue_tokens) for item in pair):
            return "The two records describe alternative top-level epilogue/output strategies; composition must instead be represented by a dedicated fused or visitor strategy."

    if level == "kernel":
        warp_schedules = ("basic-warp-specialized", "cooperative-schedule", "pingpong-schedule", "non-tma-warp-specialized")
        if all(any(token in item["id"] for token in warp_schedules) for item in pair):
            return "Alternative warp-specialized schedules assign incompatible producer/consumer execution roles."
        array_schedules = ("array-cooperative", "array-pingpong")
        if all(any(token in item["id"] for token in array_schedules) for item in pair):
            return "Alternative array-GEMM kernel schedules cannot both control the same kernel."
        schedulers = ("grouped-scheduler", "static-persistent-scheduler", "stream-k-scheduler")
        if all(any(token in item["id"] for token in schedulers) for item in pair):
            return "Alternative persistent tile schedulers define mutually exclusive work-distribution policies."
        sm80_modes = ("serial-split-k", "split-k-parallel", "stream-k", "sparse-universal", "universal-gemm-modes")
        if all(any(token in item["id"] for token in sm80_modes) for item in pair):
            return "Alternative kernel decomposition or universal execution modes cannot simultaneously own the GEMM work schedule."

    if level == "device":
        interfaces = (
            "batched-strided", "blockwise-gemm", "grouped", "pointer-array", "sparse-universal",
            "universal-adapter", "streamk-broadcast-fusion", "sparse-absmax-specialized",
        )
        if all(any(item["id"].endswith(token) for token in interfaces) for item in pair):
            return "Alternative device-facing GEMM interfaces represent different workload or operator contracts."
        if {left["id"], right["id"]} == {"cutlass.device.serial-split-k", "cutlass.device.split-k-reduction"}:
            return "Serial split-K and parallel split-K reduction are alternative orchestration protocols."
    return None


def compact(record):
    return {
        "id": record["id"], "name": record["name"], "strategy": record["strategy"],
        "primitives": record["primitives"], "parameters": record["parameters"],
        "constraints": record["constraints"], "tags": record["tags"],
    }


def build_result(architecture, level, records):
    records = sorted(records, key=lambda item: item["id"])
    prompt_input = {"architecture": architecture, "level": level, "records": [compact(item) for item in records]}
    incompatible = []
    reason_codes = {}
    reason_catalog = {}
    for (left_index, left), (right_index, right) in itertools.combinations(enumerate(records), 2):
        reason = incompatible_reason(level, left, right)
        if reason:
            if reason not in reason_codes:
                code = f"R{len(reason_codes) + 1}"
                reason_codes[reason] = code
                reason_catalog[code] = reason
            incompatible.append([left_index, right_index, reason_codes[reason]])
    total = len(records) * (len(records) - 1) // 2
    return {
        "schema_version": "1.0.0",
        "analysis_id": f"{architecture}:{level}:same-level-compatibility-v1",
        "architecture": architecture,
        "level": level,
        "inference": {
            "kind": "llm-semantic-judgment",
            "reviewer": "Codex",
            "prompt_version": "same-level-incompatible-pairs-v1",
            "input_sha256": hashlib.sha256(json.dumps(prompt_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "policy": "Only returned pairs are incompatible; every omitted unordered pair is compatible by default.",
        },
        "record_ids": [item["id"] for item in records],
        "pair_encoding": "[left_record_index, right_record_index, reason_code]",
        "reason_catalog": reason_catalog,
        "incompatible_pairs": incompatible,
        "statistics": {
            "record_count": len(records), "total_unordered_pairs": total,
            "incompatible_pair_count": len(incompatible), "default_compatible_pair_count": total - len(incompatible),
        },
    }


def validate(result):
    record_count = len(result["record_ids"])
    seen = set()
    for pair in result["incompatible_pairs"]:
        left, right, reason_code = pair
        key = (left, right)
        if not (0 <= left < right < record_count) or reason_code not in result["reason_catalog"]:
            raise ValueError(f"invalid incompatible pair: {key}")
        if key in seen:
            raise ValueError(f"duplicate incompatible pair: {key}")
        seen.add(key)


def main():
    groups = defaultdict(list)
    for record in load_records():
        for architecture in record["architectures"]:
            groups[(architecture, record["level"])].append(record)
    summary = {}
    for architecture in ARCHITECTURES:
        for level in LEVELS:
            result = build_result(architecture, level, groups[(architecture, level)])
            validate(result)
            destination = OUTPUT / architecture / f"{level}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Pair-heavy Atom results use compact JSON to avoid repeating
            # indentation across millions of LLM-returned index pairs.
            destination.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
            summary[f"{architecture}/{level}"] = result["statistics"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
