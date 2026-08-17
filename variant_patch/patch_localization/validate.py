#!/usr/bin/env python3
"""Validate localization provenance, partitioning, and source resolution claims."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
ALLOWED_REGIONS = {"mainloop_input_movement", "mma_accumulation_loop", "epilogue"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = sorted((HERE / "results").glob("*/*.json"))
    total = 0
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["status"] != "symbolic_localized" or item["next_stage"] != "patch_integration":
            raise ValueError(f"invalid state: {path}")
        for key in ("analysis", "refinement", "backbone"):
            source = ROOT / item["source"][f"{key}_path"]
            if digest(source) != item["source"][f"{key}_sha256"]:
                raise ValueError(f"stale {key}: {path}")
        ids = set()
        for localization in item["localizations"]:
            if localization["localization_id"] in ids or localization["backbone_region"] not in ALLOWED_REGIONS:
                raise ValueError(f"invalid localization: {path}")
            ids.add(localization["localization_id"])
            if not localization["candidate_insertion_points"] or localization["status"] != "symbolic_localized":
                raise ValueError(f"empty localization: {path}")
            if localization["source_resolution"] == "symbolic_only" and localization["source_anchor_matches"]:
                raise ValueError(f"inconsistent source resolution: {path}")
        total += len(ids)
    if len(files) != 18:
        raise ValueError(f"expected 18 localization documents, found {len(files)}")
    print(f"Validated {len(files)} localization documents containing {total} localized patch groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
