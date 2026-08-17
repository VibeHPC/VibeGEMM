# Same-level compatibility analysis

This directory contains the first half of the paper's hierarchical optimization
graph stage. For every architecture and optimization level, all parameterized
method records are reviewed together and the LLM returns only unordered pairs
that it judges unable to coexist. Every record under `knowledge/records/`
participates, including concrete candidates, parameterized methods, and curated
seeds.

An omitted pair is compatible by default. Results contain no cross-level pairs,
do not infer compatibility transitively, and do not claim source-backed proof.
To keep large Atom results compact, each pair is encoded as
`[left_record_index, right_record_index, reason_code]`; indices resolve through
`record_ids` and reasons through `reason_catalog`.

```console
python optimization_graph/compatibility_analysis/generate_results.py
```

Graph construction and maximal-clique enumeration are intentionally outside
this directory. The latter belongs to backbone trajectory construction.
