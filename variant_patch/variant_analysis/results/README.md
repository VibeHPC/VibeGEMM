# Variant-analysis results

Each JSON file is one normalized target variant. It separates the GEMM backbone
operations from variant-specific operations, records the value dependency graph,
and partitions every variant operation into exactly one dependency-aware patch
group. These files are the immutable inputs to `patch_localization/`.

