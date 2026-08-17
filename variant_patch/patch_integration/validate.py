#!/usr/bin/env python3
"""Audit transactional chaining, provenance, rollback, and placeholder claims."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = sorted((HERE / "results").glob("*/*.json"))
    transforms = 0
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["mode"] != "placeholder_integration" or item["real_llm_called"] or item["real_cuda_integration_performed"] or item["ready_for_real_validation"]:
            raise ValueError(f"integration overclaims real work: {path}")
        source = item["source"]
        localization_path = ROOT / source["localization_path"]
        backbone_path = ROOT / source["original_backbone_path"]
        if digest(localization_path) != source["localization_sha256"] or digest(backbone_path) != source["original_backbone_sha256"]:
            raise ValueError(f"stale source: {path}")
        localization_ids = [entry["localization_id"] for entry in json.loads(localization_path.read_text(encoding="utf-8"))["localizations"]]
        steps = item["transaction"]["steps"]
        if [entry["localization_id"] for entry in steps] != localization_ids:
            raise ValueError(f"localization coverage/order mismatch: {path}")
        expected_input = source["original_backbone_sha256"]
        for index, step in enumerate(steps):
            if step["index"] != index or step["input_source_sha256"] != expected_input or not step["placeholder"]:
                raise ValueError(f"invalid transaction chain: {path}")
            expected_input = step["output_source_sha256"]
        candidate = ROOT / item["transaction"]["candidate_path"]
        if digest(candidate) != item["transaction"]["candidate_sha256"] or digest(candidate) != expected_input:
            raise ValueError(f"candidate hash mismatch: {path}")
        marker_count = candidate.read_text(encoding="utf-8").count("VIBEGEMM_INTEGRATION_PLACEHOLDER_BEGIN")
        if marker_count != len(steps):
            raise ValueError(f"placeholder marker mismatch: {path}")
        if item["rollback"]["target_path"] != source["original_backbone_path"] or item["rollback"]["target_sha256"] != source["original_backbone_sha256"]:
            raise ValueError(f"invalid rollback: {path}")
        if not item["transaction"]["atomic"] or item["transaction"]["partial_outputs_committed"]:
            raise ValueError(f"non-atomic transaction: {path}")
        transforms += len(steps)
    if len(files) != 18:
        raise ValueError(f"expected 18 integration documents, found {len(files)}")
    print(f"Validated {len(files)} transactional integration placeholders containing {transforms} localized transforms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

