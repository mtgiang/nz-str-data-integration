"""Build DATA_DICTIONARY.csv (analytical panel variables used in the study) and DATA_SOURCE_INVENTORY.csv (source level).

Usage: python -I audit_09_dictionary_inventory.py <PROJECT_ROOT> <REPRO_DIR>
"""
import json, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1]).resolve(); rep = Path(sys.argv[2]).resolve()
spec = json.loads((root / "05_results/development_modelling_IB/freeze/FINAL_PRIMARY_MODEL_SPEC_FROZEN.json").read_text(encoding="utf-8"))
FEAT = spec["final_primary_model"]["predictors"]
KEYS = ["transition", "listing_id", "snapshot_date", "last_scraped", "anchor_date", "future_date", "evaluation_role",
        "next_period_persistence", "secondary_price_available", "price_resilience_delta", "price"]
P = pd.read_parquet(root / "02_intermediate/modelling/modelling_panel_T1_T10.parquet", columns=KEYS + FEAT)
reg = pd.read_csv(rep / "FEATURE_PROVENANCE_REGISTER.csv").set_index("feature")
desc = {
    "transition": "Rolling-origin transition T1-T10 (origin snapshot -> next snapshot)", "listing_id": "Inside Airbnb listing identifier",
    "snapshot_date": "Nominal Inside Airbnb snapshot label of the origin", "last_scraped": "Date the origin listing record was scraped",
    "anchor_date": "Origin nominal date (same as snapshot_date)", "future_date": "Nominal date of the next snapshot (label only; not a predictor)",
    "evaluation_role": "DEVELOPMENT (T1-T9) or LOCKED_T10", "next_period_persistence": "Primary label: 1 = listing present in next snapshot; disappearance = 1 - value",
    "secondary_price_available": "1 if secondary price-percentile change is defined", "price_resilience_delta": "Secondary outcome: change in within-peer advertised-price percentile",
    "price": "Origin advertised nightly price (outcome construction only; not a predictor)",
}
rows = []
for c in KEYS + FEAT:
    s = P[c]
    rows.append(dict(variable=c, role="predictor (A5)" if c in FEAT else ("outcome" if c in ("next_period_persistence", "price_resilience_delta", "secondary_price_available") else "key/metadata"),
                     family=reg.loc[c, "family"] if c in reg.index else "", dtype=str(s.dtype), n=len(s), missing_n=int(s.isna().sum()),
                     missing_pct=round(100 * s.isna().mean(), 3), n_unique=int(s.nunique(dropna=True)),
                     min=s.min() if s.dtype.kind in "ifM" else "", max=s.max() if s.dtype.kind in "ifM" else "",
                     description=desc.get(c, reg.loc[c, "transformation"] if c in reg.index else "")))
pd.DataFrame(rows).to_csv(rep / "DATA_DICTIONARY.csv", index=False)

inv = [
    dict(source_id="S1", source="Inside Airbnb - New Zealand listings, calendar and reviews", provider="Inside Airbnb", files="01_raw/airbnb/<11 dates>/{listings,calendar,reviews}.csv.gz (33 files)",
         version="snapshots 2025-10-05 ... 2026-08-13 (see SNAPSHOT_MANIFEST.csv)", url="https://insideairbnb.com/get-the-data/", licence="CC BY 4.0 (stated on download page, accessed 2026-10-12)",
         redistribution="Licence permits reuse with attribution, but review files contain reviewer first names and free text, and snapshots older than 12 months are distributed by request only; do not redistribute raw review text; share listing-origin derived tables with identifiers only after author/legal review",
         role="listings, review history, calendar configuration, primary and secondary labels", temporal_note="nominal label precedes scrape by 0-17 days (98.48% of origins)"),
    dict(source_id="S2", source="Building consents issued, by territorial authority, monthly (new, number)", provider="Stats NZ", files="01_raw/construction/statsnz_building_consents_ta_monthly_2020M01_2026M07.csv",
         version="extract 2020M01-2026M07, downloaded 2026-10-01", url="https://www.stats.govt.nz/information-releases/ (Building consents issued releases)", licence="CC BY 4.0 International (Stats NZ standard licence; not separately re-checked for this extract)",
         redistribution="Redistributable with attribution", role="4 construction predictors", temporal_note="release dates verified for 2025M07-2026M07 (10:45 NZT); revisions between first release and Oct-2026 extract not quantified"),
    dict(source_id="S3", source="Rental bond data by territorial authority, monthly", provider="MBIE Tenancy Services", files="01_raw/property_market/detailed-monthly-tla-tenancy-september.csv",
         version="file released 10 Sep 2026; data Feb 1993-Jul 2026; provisional (bond-system migration)", url="https://www.tenancy.govt.nz/about-tenancy-services/data-and-statistics/rental-bond-data/",
         licence="CC BY 3.0 NZ (verified on page 2026-10-12)", redistribution="Redistributable with attribution to MBIE", role="2 rental predictors (origin month - 2)",
         temporal_note="month-M file holds data to end of M-2; release day 10th-13th where observed; study used single post-study vintage"),
    dict(source_id="S4", source="Accommodation Data Programme - all measures", provider="MBIE / Fresh Information Ltd", files="01_raw/seasonality/ADP_All_Measures.csv",
         version="downloaded 2026-10-04; monthly from Jun 2020", url="https://www.mbie.govt.nz/immigration-and-tourism/tourism-research-and-data/tourism-data-releases/accommodation-data-programme",
         licence="NOT VERIFIED (no ADP-specific licence statement found)", redistribution="Do not redistribute until licence confirmed", role="historical seasonal profile 2023-2024 (1 predictor)",
         temporal_note="reference period ends Dec 2024 (< T1); post-study vintage"),
    dict(source_id="S5", source="Territorial authority 2025 boundaries (generalised/clipped as supplied)", provider="Stats NZ", files="01_raw/geography/statsnz-territorial-authority-2025-SHP(.zip)",
         version="TA 2025", url="https://datafinder.stats.govt.nz/", licence="CC BY 4.0 International (Stats NZ Datafinder standard; not separately re-checked)", redistribution="Redistributable with attribution",
         role="TA assignment and contextual joins", temporal_note="static classification"),
    dict(source_id="S6", source="MCERT Property and Sales data download (HPI)", provider="Ministry/Centre (MCERT) - see 07_provenance/mcert_hpi_methodological_decision.csv", files="01_raw/property_market/Property-and-Sales-data-download-June-2026.xlsx",
         version="June 2026 workbook", url="(not recorded in project)", licence="CC BY 4.0 (per author handoff; not re-checked)", redistribution="n/a - not used in any reported model",
         role="EXCLUDED from primary model (no vintages; published after T10)", temporal_note="not available at T1-T10"),
]
pd.DataFrame(inv).to_csv(rep / "DATA_SOURCE_INVENTORY.csv", index=False)
print("ok", len(rows))
