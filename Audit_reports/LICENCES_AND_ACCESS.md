# Licences, access and redistribution

Checked 12 October 2026. "Verified" = licence text read on the provider page on that date.

| Source | Licence | Status | What may be shared in a public repository |
|---|---|---|---|
| Inside Airbnb NZ listings/calendar/reviews | CC BY 4.0 ("This data is licensed under a Creative Commons Attribution 4.0 International License", insideairbnb.com/get-the-data) | Verified | Attribution required. Raw review files contain reviewer first names and free text; **do not redistribute review text or reviewer identifiers**. Snapshots older than 12 months are distributed by Inside Airbnb on request only, so readers cannot download the 2025 snapshots from the public page; share the snapshot manifest and checksums, and consider depositing derived listing-origin tables (without text/names) after author and institutional review. |
| Stats NZ building consents | CC BY 4.0 (Stats NZ standard) | Not re-checked for the specific extract | Raw extract and derived TA features may be shared with attribution. |
| Stats NZ TA 2025 boundaries | CC BY 4.0 (Datafinder standard) | Not re-checked | Shareable with attribution (or cite and link). |
| MBIE rental bond data (Tenancy Services) | CC BY 3.0 NZ | Verified | Shareable with attribution "Ministry of Business, Innovation and Employment"; note provisional status of recent data. The two Internet-Archive copies in `audit/external_evidence/` are earlier vintages of the same CC BY 3.0 NZ files. |
| MBIE Accommodation Data Programme | Not found | **UNVERIFIED** | Do not redistribute the CSV until the licence is confirmed with MBIE; share only the derived 12-month TA profile if permitted. |
| MCERT Property & Sales workbook | CC BY 4.0 per author note | Not re-checked | Not used in reported models; no need to share. |
| Code, audit scripts, frozen model artefacts | Authors' choice | Author decision | Recommend an OSI licence (e.g. MIT) for code and CC BY 4.0 for documentation. |

## What the authors must upload themselves (no repository was created and no DOI is claimed)
1. `FINAL_MDPI_DATA/reproducibility/` (scripts, manifests, register, dictionary, checksums, requirements).
2. `04_code/` (original pipeline scripts and notebooks; strip notebook outputs that print review text).
3. `05_results/development_modelling_IB/freeze/`, `05_results/final_locked_T10_primary/` (models, provenance, predictions without free text), `05_results/secondary_price_resilience_K/`, `05_results/manuscript_final_L_to_O/tables/`.
4. Derived contextual tables: `02_intermediate/construction|property_market|seasonality/*_frozen.parquet` (ADP-derived table only if licence allows).
5. Optionally the modelling panel without free text and with listing IDs, subject to Inside Airbnb attribution and institutional advice.
Then insert the repository name and DOI into the Data Availability Statement.
