# Variant analysis

This is the first component of variant-patch construction. It consumes a
structured target specification and optional reference provenance, then identifies
all operations beyond standard GEMM and their data dependencies.

The analyzer produces:

- standard GEMM backbone operations;
- topologically ordered variant-specific operations;
- produced-value dependency and consumer relationships;
- dtype and layout requirements from named tensors;
- dependency-connected semantic patch groups;
- an availability-stage recommendation: input/data movement, accumulation, or
  epilogue.

It does not select concrete source locations or generate CUDA. Those responsibilities
belong to `patch_localization/` and `patch_integration/`.

Run the six paper-derived examples and validate them:

```bash
python variant_patch/variant_analysis/analyze.py
python variant_patch/variant_analysis/validate.py
```

For a custom target, pass either one variant object or `{ "variants": [...] }` to
`--input`. A reference may be symbolic or contain a workspace-relative `path`; an
available file receives a SHA-256 provenance digest but is not executed.
