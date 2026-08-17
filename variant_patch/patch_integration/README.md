# Patch integration

This component combines localized generation and integration in one provider call
per patch group. It keeps the original best backbone immutable, applies every
transformation to an in-memory candidate, and commits only the complete candidate
after all localized steps return consistent source snapshots.

The current `PlaceholderIntegrationProvider` does not call an LLM or implement
CUDA. It appends explicit integration markers to demonstrate the transaction,
prompt contract, source-hash chain, and rollback path. Results therefore state
`real_cuda_integration_performed: false` and are not ready for real validation.

```bash
python variant_patch/patch_integration/integrate.py
python variant_patch/patch_integration/validate.py
```

