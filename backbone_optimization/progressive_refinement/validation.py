#!/usr/bin/env python3
"""Validation boundary plus a deterministic placeholder implementation."""


class PlaceholderValidator:
    """Exercises Stage 2 control flow; it never claims real GPU validation."""

    validator_id = "placeholder-validator-v1"
    is_placeholder = True

    def __init__(self, scenario="all-pass"):
        self.scenario = scenario

    def validate(self, context, repair_attempt, previous_runtime_us):
        rank, step = context["trajectory_rank"], context["step_index"]
        compile_ok = not (self.scenario == "exercise-recovery" and rank == 1 and step == 0 and repair_attempt == 0)
        numerical_ok = not (self.scenario == "exercise-recovery" and rank == 2 and step == 0)
        ratio = 1.35 if self.scenario == "exercise-recovery" and rank == 3 and step == 0 else 0.99
        runtime = previous_runtime_us * ratio
        if not compile_ok:
            numerical_ok, runtime = None, None
        elif not numerical_ok:
            runtime = None
        return {
            "validator": self.validator_id,
            "placeholder": True,
            "scenario": self.scenario,
            "compilation_execution": {
                "status": "simulated_pass" if compile_ok else "simulated_fail",
                "diagnostics": None if compile_ok else "placeholder compile failure",
            },
            "numerical_correctness": {
                "status": "not_run" if numerical_ok is None else ("simulated_pass" if numerical_ok else "simulated_fail"),
                "diagnostics": "placeholder tile mismatch" if numerical_ok is False else None,
            },
            "performance": {
                "status": "not_run" if runtime is None else "simulated_profile",
                "runtime_us": runtime,
                "relative_to_previous": None if runtime is None else ratio,
            },
        }


def decision(validation, slowdown_threshold):
    compile_status = validation["compilation_execution"]["status"]
    numerical_status = validation["numerical_correctness"]["status"]
    ratio = validation["performance"]["relative_to_previous"]
    if compile_status != "simulated_pass":
        return "repair", "compilation_or_execution_failure"
    if numerical_status != "simulated_pass":
        return "repair", "numerical_mismatch"
    if ratio > 1.0 + slowdown_threshold:
        return "terminate", "severe_performance_regression"
    return "accept", "validated_within_slowdown_threshold"

