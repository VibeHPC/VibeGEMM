#!/usr/bin/env python3
"""Validate the closed-world same-level compatibility result set."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
ARCHITECTURES = ("sm80", "sm90", "sm90a")
LEVELS = ("atom", "tile", "collective", "kernel", "device")


def main():
    records = {}
    for path in (ROOT / "knowledge" / "records").rglob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        records[item["id"]] = item
    checked_pairs = 0
    for architecture in ARCHITECTURES:
        for level in LEVELS:
            path = HERE / "results" / architecture / f"{level}.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            expected = sorted(item["id"] for item in records.values() if architecture in item["architectures"] and item["level"] == level)
            if result["architecture"] != architecture or result["level"] != level or result["record_ids"] != expected:
                raise ValueError(f"record scope mismatch: {path}")
            seen = set()
            for pair in result["incompatible_pairs"]:
                left, right, reason_code = pair
                key = left, right
                if not (0 <= left < right < len(expected)) or reason_code not in result["reason_catalog"] or key in seen:
                    raise ValueError(f"invalid pair {key} in {path}")
                seen.add(key)
            total = len(expected) * (len(expected) - 1) // 2
            stats = result["statistics"]
            if stats != {"record_count": len(expected), "total_unordered_pairs": total, "incompatible_pair_count": len(seen), "default_compatible_pair_count": total - len(seen)}:
                raise ValueError(f"statistics mismatch: {path}")
            checked_pairs += total
    print(f"Validated 15 same-level compatibility results covering {checked_pairs} unordered pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
