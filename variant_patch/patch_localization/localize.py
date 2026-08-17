#!/usr/bin/env python3
"""Map semantic patch groups to architecture-aware backbone source regions."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ANALYSIS_ROOT = HERE.parent / "variant_analysis" / "results"
REFINEMENT_ROOT = ROOT / "backbone_optimization" / "progressive_refinement" / "results"
ARCHITECTURES = ("sm80", "sm90", "sm90a")

REGIONS = {
    "input_data_movement": {
        "region": "mainloop_input_movement",
        "candidates": {
            "sm80": ["after_global_load_before_cp_async_commit", "after_shared_memory_load_before_mma"],
            "sm90": ["after_tma_load_before_wgmma", "after_shared_memory_load_before_wgmma"],
            "sm90a": ["after_tma_load_before_wgmma", "after_shared_memory_load_before_wgmma"],
        },
        "anchors": ["cp.async", "TMA", "tma", "global_load", "shared"],
    },
    "accumulation": {
        "region": "mma_accumulation_loop",
        "candidates": {arch: ["after_mma_accumulate_before_pipeline_commit"] for arch in ARCHITECTURES},
        "anchors": ["mma_sync", "wgmma", "accumulator", "gemm"],
    },
    "epilogue": {
        "region": "epilogue",
        "candidates": {arch: ["after_final_accumulator_before_output_conversion", "before_global_store"] for arch in ARCHITECTURES},
        "anchors": ["epilogue", "accumulator", "global_store", "output"],
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_id(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def locate_group(group, architecture, source_text, source_is_placeholder):
    template = REGIONS[group["availability_stage"]]
    matches = []
    lines = source_text.splitlines()
    if not source_is_placeholder:
        for line_number, line in enumerate(lines, 1):
            for anchor in template["anchors"]:
                if anchor.lower() in line.lower():
                    matches.append({"anchor": anchor, "line": line_number, "excerpt": line.strip()[:240]})
    candidates = list(template["candidates"][architecture])
    if group["requires_reduction"] and template["region"] == "epilogue":
        candidates.insert(0, "epilogue_cross_thread_reduction_before_global_store")
    return {
        "localization_id": f"localization:{stable_id({'group': group['group_id'], 'architecture': architecture})}",
        "group_id": group["group_id"],
        "semantic_stage": group["availability_stage"],
        "backbone_region": template["region"],
        "candidate_insertion_points": candidates,
        "required_live_values": group["inputs"],
        "produced_values": group["outputs"],
        "tensor_requirements": group["tensor_requirements"],
        "computation": group["computation"],
        "requires_reduction": group["requires_reduction"],
        "source_anchor_matches": matches,
        "source_resolution": "anchor_candidates" if matches else "symbolic_only",
        "status": "symbolic_localized",
    }


def main():
    summaries = {}
    for analysis_path in sorted(ANALYSIS_ROOT.glob("*.json")):
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        for architecture in ARCHITECTURES:
            if architecture not in analysis["supported_architectures"]:
                continue
            refinement_path = REFINEMENT_ROOT / f"{architecture}.json"
            refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
            backbone = refinement["best_validated_backbone"]
            backbone_path = ROOT / backbone["kernel_path"]
            if digest(backbone_path) != backbone["kernel_sha256"]:
                raise ValueError(f"stale best backbone: {architecture}")
            source_text = backbone_path.read_text(encoding="utf-8")
            source_is_placeholder = refinement["mode"] == "placeholder_simulation"
            localizations = [locate_group(group, architecture, source_text, source_is_placeholder) for group in analysis["patch_groups"]]
            document = {
                "schema_version": "1.0.0",
                "localization_set_id": f"patch-localization:{stable_id({'analysis': analysis['analysis_id'], 'architecture': architecture, 'backbone': backbone['kernel_sha256']})}",
                "architecture": architecture,
                "variant_id": analysis["variant_id"],
                "source": {
                    "analysis_path": analysis_path.relative_to(ROOT).as_posix(),
                    "analysis_sha256": digest(analysis_path),
                    "refinement_path": refinement_path.relative_to(ROOT).as_posix(),
                    "refinement_sha256": digest(refinement_path),
                    "backbone_path": backbone_path.relative_to(ROOT).as_posix(),
                    "backbone_sha256": digest(backbone_path),
                    "backbone_is_placeholder": source_is_placeholder,
                },
                "backbone_invariants": [
                    "preserve GEMM tensor-core instruction family",
                    "preserve tile shapes and thread/warp mapping",
                    "preserve mainloop pipeline and synchronization unless required by the patch",
                    "preserve existing shared-memory and register lifetimes",
                    "modify only the selected localized region",
                ],
                "localizations": localizations,
                "status": "symbolic_localized",
                "next_stage": "patch_integration",
            }
            output = HERE / "results" / architecture / f"{analysis['variant_id']}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            summaries[f"{architecture}/{analysis['variant_id']}"] = len(localizations)
    print(json.dumps({"documents": len(summaries), "localized_patch_groups": sum(summaries.values()), "by_document": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
