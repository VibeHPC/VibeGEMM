#!/usr/bin/env python3
"""Run paper-faithful progressive refinement using replaceable placeholders."""

import argparse
import hashlib
import json
from pathlib import Path

from provider import PlaceholderLLMProvider, write_candidate
from validation import PlaceholderValidator, decision

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
TRAJECTORY_ROOT = HERE.parent / "trajectory_construction" / "results"
KNOWLEDGE_ROOT = ROOT / "knowledge" / "records"
ARCHITECTURES = ("sm80", "sm90", "sm90a")
INITIAL_CODE = """// VibeGEMM Stage 2 initial-kernel placeholder\n// Replace with a valid architecture-specific GEMM kernel.\nextern \"C\" __global__ void vibegemm_initial_placeholder() {}\n"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_record_index():
    records = {}
    for path in KNOWLEDGE_ROOT.rglob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("level"):
            item["_source_path"] = path.relative_to(ROOT).as_posix()
            records[item["id"]] = item
    return records


def diagnostics(validation):
    return {
        "compilation_execution": validation["compilation_execution"]["diagnostics"],
        "numerical_correctness": validation["numerical_correctness"]["diagnostics"],
    }


def run_architecture(architecture, args, records, provider, validator):
    trajectory_path = TRAJECTORY_ROOT / f"{architecture}.json"
    trajectory_set = json.loads(trajectory_path.read_text(encoding="utf-8"))
    run_id = f"{architecture}-placeholder-refinement-v1"
    artifact_root = HERE / "artifacts" / run_id
    initial_path = artifact_root / "initial.cu"
    initial_path.parent.mkdir(parents=True, exist_ok=True)
    initial_path.write_bytes(INITIAL_CODE.encode("utf-8"))
    global_best = {"kernel_path": initial_path.relative_to(ROOT).as_posix(), "kernel_sha256": sha256(initial_path), "runtime_us": args.initial_runtime_us, "trajectory_id": None, "step_index": None}
    traces = []
    for trajectory in trajectory_set["trajectories"][:args.trajectory_limit]:
        latest_code = INITIAL_CODE
        latest = dict(global_best) if args.start_from_global_best else {"kernel_path": initial_path.relative_to(ROOT).as_posix(), "kernel_sha256": sha256(initial_path), "runtime_us": args.initial_runtime_us, "trajectory_id": None, "step_index": None}
        if args.start_from_global_best and global_best["kernel_path"] != initial_path.relative_to(ROOT).as_posix():
            latest_code = (ROOT / global_best["kernel_path"]).read_text(encoding="utf-8")
        trace = {"trajectory_id": trajectory["id"], "rank": trajectory["rank"], "start_state": dict(latest), "events": [], "status": "completed"}
        for step_index, step in enumerate(trajectory["steps"][:args.step_limit]):
            input_valid_state = dict(latest)
            record = records.get(step["record_id"])
            if record is None:
                raise ValueError(f"missing knowledge record: {step['record_id']}")
            context = {"architecture": architecture, "trajectory_id": trajectory["id"], "trajectory_rank": trajectory["rank"], "step_index": step_index}
            response = provider.generate(latest_code, record, context)
            attempts = []
            accepted = False
            for repair_attempt in range(args.max_repairs + 1):
                candidate_path = artifact_root / trajectory["id"].replace(":", "-") / f"step-{step_index:04d}-attempt-{repair_attempt}.cu"
                write_candidate(candidate_path, response)
                validation = validator.validate(context, repair_attempt, latest["runtime_us"])
                action, reason = decision(validation, args.slowdown_threshold)
                attempts.append({"repair_attempt": repair_attempt, "candidate_path": candidate_path.relative_to(ROOT).as_posix(), "candidate_sha256": response["code_sha256"], "generation_placeholder": True, "validation": validation, "decision": action, "reason": reason})
                if action == "accept":
                    latest_code = response["code"]
                    latest = {"kernel_path": candidate_path.relative_to(ROOT).as_posix(), "kernel_sha256": response["code_sha256"], "runtime_us": validation["performance"]["runtime_us"], "trajectory_id": trajectory["id"], "step_index": step_index}
                    if latest["runtime_us"] < global_best["runtime_us"]:
                        global_best = dict(latest)
                    accepted = True
                    break
                if action == "terminate":
                    break
                if repair_attempt < args.max_repairs:
                    response = provider.repair(latest_code, record, context, diagnostics(validation), repair_attempt + 1)
            trace["events"].append({"step_index": step_index, "level": step["level"], "record_id": step["record_id"], "record_source": record["_source_path"], "input_valid_state": input_valid_state, "attempts": attempts, "status": "accepted" if accepted else "rejected_and_rolled_back"})
            if not accepted:
                trace["status"] = "terminated_and_rolled_back"
                trace["rollback_state"] = dict(latest)
                break
        trace["terminal_valid_state"] = dict(latest)
        traces.append(trace)
    result = {
        "schema_version": "1.0.0", "run_id": run_id, "architecture": architecture,
        "mode": "placeholder_simulation", "real_llm_called": False, "real_cuda_validation_performed": False,
        "source": {"trajectory_path": trajectory_path.relative_to(ROOT).as_posix(), "trajectory_sha256": sha256(trajectory_path)},
        "configuration": {"trajectory_limit": args.trajectory_limit, "step_limit": args.step_limit, "max_repairs": args.max_repairs, "slowdown_threshold": args.slowdown_threshold, "initial_runtime_us": args.initial_runtime_us, "scenario": args.scenario, "start_from_global_best": args.start_from_global_best},
        "provider": provider.provider_id, "validator": validator.validator_id,
        "initial_valid_state": {"kernel_path": initial_path.relative_to(ROOT).as_posix(), "kernel_sha256": sha256(initial_path), "runtime_us": args.initial_runtime_us},
        "trajectory_traces": traces, "best_validated_backbone": global_best,
        "statistics": {"trajectory_count": len(traces), "accepted_steps": sum(e["status"] == "accepted" for t in traces for e in t["events"]), "rolled_back_trajectories": sum(t["status"] == "terminated_and_rolled_back" for t in traces), "placeholder_attempts": sum(len(e["attempts"]) for t in traces for e in t["events"])},
    }
    output = HERE / "results" / f"{architecture}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result["statistics"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=(*ARCHITECTURES, "all"), default="all")
    parser.add_argument("--trajectory-limit", type=int, default=3)
    parser.add_argument("--step-limit", type=int, default=3)
    parser.add_argument("--max-repairs", type=int, default=5)
    parser.add_argument("--slowdown-threshold", type=float, default=0.30)
    parser.add_argument("--initial-runtime-us", type=float, default=100.0)
    parser.add_argument("--scenario", choices=("all-pass", "exercise-recovery"), default="exercise-recovery")
    parser.add_argument("--start-from-global-best", action="store_true")
    args = parser.parse_args()
    if args.trajectory_limit < 1 or args.step_limit < 1 or args.max_repairs < 0:
        parser.error("limits must be positive and max-repairs must be non-negative")
    records = load_record_index()
    provider, validator = PlaceholderLLMProvider(), PlaceholderValidator(args.scenario)
    architectures = ARCHITECTURES if args.architecture == "all" else (args.architecture,)
    print(json.dumps({arch: run_architecture(arch, args, records, provider, validator) for arch in architectures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
