# Shareable data products (derived; no Inside Airbnb records republished)

Inside Airbnb asks users not to republish its data. This package therefore contains **no listing-level
records, no listing or host identifiers, no coordinates and no review text**. It contains:

| File | Content | Source licence |
|---|---|---|
| ta_transition_aggregates.csv | Listings, observed next-snapshot disappearance rate and mean predicted risk per TA x transition; cells with < 10 listings suppressed | derived aggregates (cite Inside Airbnb) |
| ta_transition_context_layer.csv | Release-aware building-consent and rental-bond features per TA x transition, exactly as used by the model | Stats NZ (CC BY 4.0), MBIE (CC BY 3.0 NZ) |
| dev_metrics_by_window.csv | ROC-AUC/PR-AUC per specification and development window | derived |
| R1-R4 *.csv | Paired family comparisons, confirmed-label robustness, construction strata, TA-level association | derived |
| SYNTHETIC_listing_origin_sample.csv | 5,000 synthetic rows with the panel schema; every column resampled independently, so no row is a real listing. For running the code only; results on it are meaningless | synthetic |
| CHECKSUMS.csv | SHA-256 of every file | |

To reproduce listing-level results, obtain the Inside Airbnb New Zealand snapshots listed in the snapshot
manifest (current files from insideairbnb.com/get-the-data; older ones by archived-data request), verify their
SHA-256 against the manifest, and run the published code. Attribution: Inside Airbnb (insideairbnb.com).
