# SM90 compatibility results

The five JSON files are independent same-level pair-wise analyses over every
SM90-applicable record in `knowledge/records`. `record_ids` is the stable node
table. Each `incompatible_pairs` entry is
`[left_record_index, right_record_index, reason_code]`; `reason_catalog`
resolves the code. Any omitted unordered pair is compatible by default.
