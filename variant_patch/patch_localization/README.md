# Patch localization

This component maps semantic patch groups to architecture-aware regions of the
best validated backbone. It consumes `variant_analysis/results/` and the Stage 2
best-backbone provenance for SM80, SM90, and SM90a.

Every localization records the region, ordered candidate insertion points,
required live values, produced values, tensor requirements, reduction needs, and
backbone invariants. Source anchors are advisory. When a placeholder backbone has
no concrete CUDA anchor, the result remains `symbolic_only` and never invents a
line number.

```bash
python variant_patch/patch_localization/localize.py
python variant_patch/patch_localization/validate.py
```

