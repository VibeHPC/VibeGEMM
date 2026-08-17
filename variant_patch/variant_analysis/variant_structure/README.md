# Variant-analysis result structure

`variant-analysis.schema.json` describes normalized analysis output. The central
objects are `variant_operations`, explicit produced-value dependencies, and
`patch_groups`. A patch group is a semantic unit only: `availability_stage` is an
analysis recommendation, while `localization_status` stays `pending` until the
next component maps it to a concrete backbone source region.

