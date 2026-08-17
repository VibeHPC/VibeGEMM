#!/usr/bin/env python3
"""Validate placeholder refinement provenance and state-machine invariants."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    counts = {"runs": 0, "traces": 0, "events": 0}
    for architecture in ("sm80", "sm90", "sm90a"):
        result = json.loads((HERE / "results" / f"{architecture}.json").read_text(encoding="utf-8"))
        if result["mode"] != "placeholder_simulation" or result["real_llm_called"] or result["real_cuda_validation_performed"]:
            raise ValueError(f"invalid placeholder declaration: {architecture}")
        source = ROOT / result["source"]["trajectory_path"]
        if digest(source) != result["source"]["trajectory_sha256"]:
            raise ValueError(f"stale trajectory source: {architecture}")
        best_runtime = result["initial_valid_state"]["runtime_us"]
        for trace in result["trajectory_traces"]:
            latest_sha = trace["start_state"]["kernel_sha256"]
            latest_runtime = trace["start_state"]["runtime_us"]
            for event in trace["events"]:
                if event["input_valid_state"]["kernel_sha256"] != latest_sha or event["input_valid_state"]["runtime_us"] != latest_runtime:
                    raise ValueError(f"invalid event input state: {trace['trajectory_id']}")
                if not event["attempts"] or len(event["attempts"]) > result["configuration"]["max_repairs"] + 1:
                    raise ValueError(f"invalid repair count: {trace['trajectory_id']}")
                for attempt in event["attempts"]:
                    candidate = ROOT / attempt["candidate_path"]
                    if digest(candidate) != attempt["candidate_sha256"]:
                        raise ValueError(f"candidate hash mismatch: {candidate}")
                terminal = event["attempts"][-1]
                if event["status"] == "accepted":
                    if terminal["decision"] != "accept":
                        raise ValueError("accepted event lacks accept decision")
                    latest_sha = terminal["candidate_sha256"]
                    latest_runtime = terminal["validation"]["performance"]["runtime_us"]
                    best_runtime = min(best_runtime, latest_runtime)
                elif terminal["decision"] not in ("repair", "terminate"):
                    raise ValueError("rejected event has invalid decision")
            if trace["terminal_valid_state"]["kernel_sha256"] != latest_sha or trace["terminal_valid_state"]["runtime_us"] != latest_runtime:
                raise ValueError(f"invalid terminal state: {trace['trajectory_id']}")
            if trace["status"] == "terminated_and_rolled_back" and trace["rollback_state"] != trace["terminal_valid_state"]:
                raise ValueError(f"invalid rollback: {trace['trajectory_id']}")
            counts["traces"] += 1
            counts["events"] += len(trace["events"])
        if result["best_validated_backbone"]["runtime_us"] != best_runtime:
            raise ValueError(f"invalid global best: {architecture}")
        counts["runs"] += 1
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
