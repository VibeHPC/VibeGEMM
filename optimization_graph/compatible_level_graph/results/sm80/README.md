# SM80 compatible level graphs

Each JSON file is one independent SM80 level graph. `record_ids` defines the
node table and every `compatible_edges` entry is `[left_index, right_index]`.
Edges are undirected, indices are strictly ordered, and there are no cross-level
edges. `source` binds the compatibility-analysis input by SHA-256.
