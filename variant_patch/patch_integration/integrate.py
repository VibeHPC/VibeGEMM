#!/usr/bin/env python3
"""Transactionally create full-kernel integration candidates via a provider."""

import hashlib
import json
from pathlib import Path

from provider import PlaceholderIntegrationProvider

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
LOCALIZATION_ROOT = HERE.parent / "patch_localization" / "results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    provider = PlaceholderIntegrationProvider()
    summary = {}
    for localization_path in sorted(LOCALIZATION_ROOT.glob("*/*.json")):
        localization_set = json.loads(localization_path.read_text(encoding="utf-8"))
        architecture = localization_set["architecture"]
        variant_id = localization_set["variant_id"]
        backbone_path = ROOT / localization_set["source"]["backbone_path"]
        backbone_sha256 = localization_set["source"]["backbone_sha256"]
        if digest(backbone_path) != backbone_sha256:
            raise ValueError(f"stale backbone: {architecture}/{variant_id}")
        current_source = backbone_path.read_bytes().decode("utf-8")
        current_sha256 = hashlib.sha256(current_source.encode()).hexdigest()
        steps = []
        context = {
            "architecture": architecture,
            "variant_id": variant_id,
            "original_backbone": {"path": backbone_path.relative_to(ROOT).as_posix(), "sha256": backbone_sha256, "is_placeholder": localization_set["source"]["backbone_is_placeholder"]},
            "backbone_invariants": localization_set["backbone_invariants"],
        }
        # The transaction stays in memory. No candidate is committed until every
        # localized transformation returns a complete, hashable source snapshot.
        for index, localization in enumerate(localization_set["localizations"]):
            response = provider.integrate(current_source, localization, context)
            if not response["integrated_source"] or hashlib.sha256(response["integrated_source"].encode()).hexdigest() != response["integrated_sha256"]:
                raise ValueError(f"provider returned an invalid snapshot: {localization['localization_id']}")
            steps.append({
                "index": index,
                "localization_id": localization["localization_id"],
                "provider": response["provider"],
                "placeholder": True,
                "prompt": response["prompt"],
                "input_source_sha256": current_sha256,
                "output_source_sha256": response["integrated_sha256"],
                "status": "placeholder_transform_applied",
            })
            current_source = response["integrated_source"]
            current_sha256 = response["integrated_sha256"]
        artifact = HERE / "artifacts" / architecture / variant_id / "kernel.variant.placeholder.cu"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(current_source.encode("utf-8"))
        document = {
            "schema_version": "1.0.0",
            "integration_id": localization_set["localization_set_id"].replace("patch-localization", "patch-integration"),
            "architecture": architecture,
            "variant_id": variant_id,
            "mode": "placeholder_integration",
            "real_llm_called": False,
            "real_cuda_integration_performed": False,
            "source": {
                "localization_path": localization_path.relative_to(ROOT).as_posix(),
                "localization_sha256": digest(localization_path),
                "original_backbone_path": backbone_path.relative_to(ROOT).as_posix(),
                "original_backbone_sha256": backbone_sha256,
            },
            "transaction": {
                "atomic": True,
                "partial_outputs_committed": False,
                "steps": steps,
                "candidate_path": artifact.relative_to(ROOT).as_posix(),
                "candidate_sha256": digest(artifact),
                "status": "placeholder_candidate_committed",
            },
            "rollback": {
                "action": "discard_candidate_and_restore_original_backbone",
                "target_path": backbone_path.relative_to(ROOT).as_posix(),
                "target_sha256": backbone_sha256,
            },
            "ready_for_real_validation": False,
            "status": "integration_placeholder_created",
            "next_stage": "replace_placeholder_or_final_validation",
        }
        output = HERE / "results" / architecture / f"{variant_id}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        summary[f"{architecture}/{variant_id}"] = len(steps)
    print(json.dumps({"documents": len(summary), "placeholder_transforms": sum(summary.values()), "by_document": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
