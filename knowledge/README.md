# VibeGEMM GPU Knowledge Repository

This directory implements the first offline component described in the
VibeGEMM paper. Knowledge is stored as small, reviewable records rather than as
copied source code. Every record captures:

`<primitives, strategy, parameters, constraints, architecture, source>`

Records are organized into five optimization levels: `atom`, `tile`,
`collective`, `kernel`, and `device`. A record is useful only when its strategy
is actionable, its constraints are explicit, and its evidence resolves to the
vendored source tree.

## Layout

```text
knowledge/
  record_structure/knowledge-record.schema.json
  record_source/cutlass-4.7.0-gemm-scope.json
  record-manifest.json
  records/<architecture>/<level>/*.json
tools/knowledge/
  validate.py
  query.py
```

## Record policy

- IDs are stable and retain library/version provenance inside each record.
- `records/` is the only knowledge data root. It has no library or version
  directory layer; records are grouped by primary architecture and level.
- The manifest classifies records as `concrete_candidate`,
  `parameterized_method`, or `curated_seed`. A record may declare additional
  compatible architectures even though it has one canonical file location.
- `summary` describes what the strategy does, not merely the source symbol.
- Parameters distinguish tunable values from fixed hardware properties.
- Constraints are machine-readable predicates where possible.
- Source references use repository-relative paths and precise line ranges.
- Extracted records remain `candidate` until source validation and human review
  succeed.
- CUTLASS code remains under its BSD-3-Clause license. Records reference source
  locations and do not embed substantial source excerpts.

## Commands

```bash
python tools/knowledge/validate.py
python tools/knowledge/query.py --architecture sm90a --level atom
python tools/knowledge/query.py --tag warp-specialized --json
python tools/knowledge/inventory.py
python tools/knowledge/extract_atoms.py
python tools/knowledge/extract_atoms.py --write
python tools/knowledge/query.py --kind concrete_candidate --architecture sm80
python tools/knowledge/query.py --kind parameterized_method --architecture sm90a
python tools/knowledge/normalize_atoms.py
python tools/knowledge/normalize_atoms.py --write
python tools/knowledge/validate.py
python tools/knowledge/extract_tiles.py
python tools/knowledge/extract_tiles.py --write
python tools/knowledge/extract_collectives.py
python tools/knowledge/extract_collectives.py --write
python tools/knowledge/extract_kernels.py
python tools/knowledge/extract_kernels.py --write
python tools/knowledge/extract_devices.py --write
```

The canonical repository contains 2,422 records: 2,177 concrete candidates,
235 parameterized methods, and 10 curated seed records. Optimization-graph
construction consumes every record under `knowledge/records/`; the manifest
retains each record's class and provenance rather than filtering graph input.

The extraction scope is maintained in
`knowledge/record_source/cutlass-4.7.0-gemm-scope.json`. It covers SM80 and
SM90/SM90a GEMM atoms, copy primitives, tiling builders, mainloops, epilogues,
kernel schedules, and device interfaces. A source inventory is broader than the
curated record set: it defines what must be examined, while records capture the
reusable optimization methods found there.
