#!/usr/bin/env python3
"""LLM provider boundary and a deterministic, offline placeholder provider."""

import hashlib
from pathlib import Path


class PlaceholderLLMProvider:
    """Produces auditable CUDA-shaped placeholders without calling an LLM."""

    provider_id = "placeholder-llm-v1"
    is_placeholder = True

    def generate(self, current_code, record, context):
        prompt = self._prompt(current_code, record, context, None)
        return self._response(current_code, record, context, prompt, 0)

    def repair(self, current_code, record, context, diagnostics, attempt):
        prompt = self._prompt(current_code, record, context, diagnostics)
        return self._response(current_code, record, context, prompt, attempt)

    @staticmethod
    def _prompt(current_code, record, context, diagnostics):
        return {
            "task": "transform_latest_valid_cuda_kernel",
            "architecture": context["architecture"],
            "trajectory_id": context["trajectory_id"],
            "step_index": context["step_index"],
            "strategy": record,
            "latest_valid_kernel_sha256": hashlib.sha256(current_code.encode()).hexdigest(),
            "validation_diagnostics": diagnostics,
            "requirements": [
                "preserve existing GEMM semantics",
                "apply only the selected strategy",
                "return a complete CUDA source file",
            ],
        }

    def _response(self, current_code, record, context, prompt, attempt):
        marker = (
            f"\n// VIBEGEMM_PLACEHOLDER_BEGIN\n"
            f"// provider: {self.provider_id}\n"
            f"// strategy: {record['id']}\n"
            f"// level: {record['level']}\n"
            f"// repair_attempt: {attempt}\n"
            f"// intent: {record.get('strategy', {}).get('intent', '')}\n"
            f"// TODO(real LLM): implement this strategy in CUDA.\n"
            f"// VIBEGEMM_PLACEHOLDER_END\n"
        )
        code = current_code + marker
        return {
            "provider": self.provider_id,
            "placeholder": True,
            "prompt": prompt,
            "code": code,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        }


def write_candidate(path: Path, response):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response["code"].encode("utf-8"))
