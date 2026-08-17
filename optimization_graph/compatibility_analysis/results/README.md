# Compatibility-analysis results

This directory contains one subdirectory per GPU architecture. Each
architecture has five JSON files (`atom`, `tile`, `collective`, `kernel`, and
`device`). Every file lists all same-level records and only the pairs judged
incompatible; omitted pairs are compatible by default. Generate with
`../generate_results.py` and validate with `../validate_results.py`.
