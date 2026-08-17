# Variant-specific patch pipeline

After backbone optimization returns the best validated GEMM backbone, this stage
identifies and integrates only the computation specific to the target variant.
The paper-aligned pipeline is:

```text
target specification + reference implementation
  -> variant_analysis
  -> patch_localization
  -> patch_integration
  -> final validation
```

## Current components

`variant_analysis/` is implemented. It normalizes the target computation,
separates standard GEMM from variant-specific operations, builds explicit data
dependencies, records dtype/layout requirements, and groups connected operations
that should be localized together.

The analysis is semantic: it recommends the earliest operand-availability stage
but does not choose a concrete CUDA source location or modify a backbone. Its
results end with `next_stage: patch_localization`.

`patch_localization/` maps each semantic group to architecture-aware candidate
regions in the best backbone while preserving source provenance and invariants.
`patch_integration/` constructs the localized LLM request and transactionally
creates a complete kernel candidate. Its current LLM integration provider is an
explicit placeholder and makes no external model call.

Run the paper-derived examples:

```bash
python variant_patch/variant_analysis/analyze.py
python variant_patch/variant_analysis/validate.py
python variant_patch/patch_localization/localize.py
python variant_patch/patch_localization/validate.py
python variant_patch/patch_integration/integrate.py
python variant_patch/patch_integration/validate.py
```
