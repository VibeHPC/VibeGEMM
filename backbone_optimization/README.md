# Backbone optimization

This directory implements the paper's trajectory-style backbone optimization.
Stage 1 constructs candidate compatible trajectories; later stages will progressively
apply, compile, validate, benchmark, and roll back individual steps.

## Stage 1: compatible trajectory construction

The data flow is:

```text
compatible level graph
  -> maximal_clique
  -> ordered_sequence
  -> trajectory_construction
```

For every architecture and level, `maximal_clique/` enumerates maximal compatible
record sets with pivoted Bron-Kerbosch. `ordered_sequence/` turns every set into
deterministic semantic forward/reverse orders. `trajectory_construction/` chooses
one sequence from each level and concatenates them in this fixed order:

```text
Atom -> Tile -> Collective -> Kernel -> Device
```

Run and validate the complete stage:

```bash
python backbone_optimization/maximal_clique/enumerate.py
python backbone_optimization/maximal_clique/validate.py
python backbone_optimization/ordered_sequence/construct.py
python backbone_optimization/ordered_sequence/validate.py
python backbone_optimization/trajectory_construction/build.py
python backbone_optimization/trajectory_construction/validate.py
```

Enumeration is deliberately budgeted because the SM90a Atom graph has a very
large maximal-clique space. Every result states whether enumeration or Cartesian
combination was complete. Cross-level compatibility remains `unverified` in Stage
1 and is checked during progressive refinement rather than invented as graph edges.

## Stage 2: progressive trajectory refinement

`progressive_refinement/` implements sequential knowledge retrieval, a replaceable
LLM code-generation boundary, bounded repair, three-stage validation decisions,
trajectory pruning, rollback, and global-best tracking. Its current provider and
validator are deliberately marked placeholders: they exercise the complete state
machine without calling an LLM, compiling CUDA, or reporting simulated outcomes
as real measurements.
