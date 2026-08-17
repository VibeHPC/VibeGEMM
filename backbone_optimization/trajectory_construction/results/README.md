# Compatible trajectory results

Each architecture JSON selects one ordered sequence per level and concatenates
its records in `Atom -> Tile -> Collective -> Kernel -> Device` order. A trajectory
stores the selected sequence IDs and flattened steps. `cross_level_compatibility`
is intentionally `unverified`; progressive refinement is responsible for checking
it. Statistics distinguish the full Cartesian combination space from the stored
deterministic budgeted sample.

