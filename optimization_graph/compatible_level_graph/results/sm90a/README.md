# SM90a compatible level graphs

Each JSON file is one independent SM90a level graph. `record_ids` is the node
table and `compatible_edges` stores undirected index pairs. The Atom file is
compact because it contains thousands of nodes and tens of thousands of
compatible edges. No maximal cliques are materialized here.
