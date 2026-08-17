# Final hierarchical optimization graphs

`sm80.json`, `sm90.json`, and `sm90a.json` are the final offline graphs defined
as `G = {G_atom, G_tile, G_collective, G_kernel, G_device}`. Each `levels`
entry embeds the exact corresponding compatible level graph, while `sources`
binds it by path and SHA-256. `semantics` explicitly records that there are no
cross-level edges, transitive inferences, or materialized maximal cliques.

Generate with `../build.py` and validate with `../validate.py`.
