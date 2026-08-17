#!/usr/bin/env python3
"""Replaceable localized-integration provider and its offline placeholder."""

import hashlib


class PlaceholderIntegrationProvider:
    """Builds an auditable candidate without making an external LLM call."""

    provider_id = "placeholder-integration-llm-v1"
    is_placeholder = True

    def integrate(self, current_source, localization, context):
        prompt = {
            "task": "generate_and_integrate_one_localized_cuda_patch",
            "architecture": context["architecture"],
            "variant_id": context["variant_id"],
            "latest_candidate_sha256": hashlib.sha256(current_source.encode()).hexdigest(),
            "original_backbone": context["original_backbone"],
            "target_region": localization["backbone_region"],
            "candidate_insertion_points": localization["candidate_insertion_points"],
            "computation": localization["computation"],
            "required_live_values": localization["required_live_values"],
            "produced_values": localization["produced_values"],
            "tensor_requirements": localization["tensor_requirements"],
            "backbone_invariants": context["backbone_invariants"],
            "requirements": [
                "modify only the localized region",
                "preserve all unrelated backbone source",
                "return the complete integrated CUDA source",
                "preserve variant numerical semantics",
            ],
        }
        marker = (
            "\n// VIBEGEMM_INTEGRATION_PLACEHOLDER_BEGIN\n"
            f"// provider: {self.provider_id}\n"
            f"// localization: {localization['localization_id']}\n"
            f"// target_region: {localization['backbone_region']}\n"
            f"// insertion_point: {localization['candidate_insertion_points'][0]}\n"
            f"// computation: {localization['computation']}\n"
            "// TODO(real LLM): generate and integrate the localized CUDA code.\n"
            "// VIBEGEMM_INTEGRATION_PLACEHOLDER_END\n"
        )
        integrated_source = current_source + marker
        return {
            "provider": self.provider_id,
            "placeholder": True,
            "prompt": prompt,
            "integrated_source": integrated_source,
            "integrated_sha256": hashlib.sha256(integrated_source.encode()).hexdigest(),
        }

