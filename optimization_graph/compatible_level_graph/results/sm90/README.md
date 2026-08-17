# SM90 compatible level graphs

Each JSON file is one independent SM90 level graph. Nodes are the listed
records and `compatible_edges` uses compact `[left_index, right_index]`
undirected edges. Statistics partition all possible pairs into compatible and
incompatible sets; no transitive edges are added.
