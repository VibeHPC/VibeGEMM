# Maximal cliques

This component runs pivoted Bron-Kerbosch independently on each of the 15
compatible level graphs. Every emitted clique is pairwise compatible and
maximal. `maximal_cliques` stores arrays of indices into `record_ids`.

Enumeration has an explicit budget because the SM90a Atom graph has more than
100,000 maximal cliques. `statistics.enumeration_complete` records whether the
result is exhaustive.
