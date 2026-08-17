# Compatible trajectory construction

One ordered sequence is selected from each level and concatenated in the fixed
Atom-to-Device order. The full Cartesian space is recorded, while a deterministic
coprime-stride sampler materializes at most the requested trajectory budget.
Every step is marked `cross_level_compatibility=unverified`; such conflicts are
checked only when progressive refinement instantiates and validates a kernel.
