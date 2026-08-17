# Ordered-sequence results

Results are grouped by architecture and level. Each sequence references a source
maximal clique and stores `order`, an array of indexes into the document's
`record_ids`. The `record_sequence` can therefore be recovered as
`record_ids[order[i]]`. Source paths and hashes make stale results detectable.

