# Hierarchical Optimization Graph

This offline component implements the paper's three-step graph construction
from every record under `knowledge/records/`.

## 1. Same-level compatibility analysis

`compatibility_analysis/` groups all records by declared GPU architecture and
optimization level. It records only same-level unordered pairs judged
incompatible; every omitted pair is compatible by default.

```console
python optimization_graph/compatibility_analysis/generate_results.py
python optimization_graph/compatibility_analysis/validate_results.py
```

## 2. Compatible level graphs

`compatible_level_graph/` builds one undirected graph per architecture and
level. Nodes are all records in that analysis. Edges are the complete set of
unordered pairs minus the incompatible pairs.

```console
python optimization_graph/compatible_level_graph/build.py
python optimization_graph/compatible_level_graph/validate.py
```

## 3. Hierarchical assembly

`hierarchical_optimization_graph/` combines the five independent level graphs
for each architecture as:

`G = {G_atom, G_tile, G_collective, G_kernel, G_device}`

```console
python optimization_graph/hierarchical_optimization_graph/build.py
python optimization_graph/hierarchical_optimization_graph/validate.py
```

Final graphs are stored in
`hierarchical_optimization_graph/results/{sm80,sm90,sm90a}.json`.

This component creates no cross-level edges and performs no transitive
compatibility inference. Maximal-clique enumeration, strategy ordering, and
trajectory construction belong to `backbone_optimization/`.
