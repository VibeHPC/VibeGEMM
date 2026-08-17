# Compatible level graphs

Each result is one undirected graph for one GPU architecture and one
optimization level. Nodes are every record listed by the corresponding
compatibility analysis. Edges are all unordered record pairs except those that
the analysis marked incompatible.

```console
python optimization_graph/compatible_level_graph/build.py
python optimization_graph/compatible_level_graph/validate.py
```

Edges use compact integer indices into `record_ids`. These files contain no
cross-level edges, no transitive inference, no maximal cliques, and no ordered
optimization trajectories.
