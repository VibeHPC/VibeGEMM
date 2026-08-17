# Hierarchical Optimization Graph

For each architecture, this component assembles the five independently built
same-level compatibility graphs as:

`G = {G_atom, G_tile, G_collective, G_kernel, G_device}`

The hierarchy embeds the exact level graphs and binds each source by SHA-256.
It adds no cross-level edges and performs no transitive compatibility inference.
Maximal-clique enumeration and ordered trajectory construction remain downstream
backbone-optimization responsibilities.

```console
python optimization_graph/hierarchical_optimization_graph/build.py
python optimization_graph/hierarchical_optimization_graph/validate.py
```
