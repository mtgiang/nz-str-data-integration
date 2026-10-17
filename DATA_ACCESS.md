# Data access: how to obtain the source data and reproduce the results

The Inside Airbnb source files are **not redistributed** in this repository, because Inside Airbnb's data policies ask users not to republish the data. This file explains how to obtain the same files and confirm that they are identical to those used in the study.

## 1. Inside Airbnb, New Zealand (11 snapshots)

Inside Airbnb publishes all New Zealand snapshots on one page; there is no separate public link per snapshot to cite.
Each snapshot consists of three files: `listings.csv.gz` (detailed listings), `calendar.csv.gz` (detailed calendar) and `reviews.csv.gz` (detailed reviews).

**How to download a snapshot**
1. Open <https://insideairbnb.com/get-the-data/>.
2. Press Ctrl+F and search for **New Zealand**.
3. In the New Zealand section, click **show archived data** to list earlier snapshots.
4. Find the snapshot date in the table below and download `listings.csv.gz`, `calendar.csv.gz` and `reviews.csv.gz` listed under that date.
5. If a date is no longer listed (Inside Airbnb keeps about the last twelve months online), request it through the **archived data request** form (<https://insideairbnb.com/data-requests/>), quoting the snapshot date and the three file names.

| # | Snapshot date (as listed by Inside Airbnb) | Role in the study | Files |
|---|---|---|---|
| 1 | 5 October 2025 (2025-10-05) | Origin T1 | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |
| 2 | 7 November 2025 (2025-11-07) | T1 label / origin T2 | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |
| 3 | 11 December 2025 (2025-12-11) | T2 label / origin T3 | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |
| 4 | 16 January 2026 (2026-01-16) | T3 label / origin T4 | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |
| 5 | 13 February 2026 (2026-02-13) | T4 label / origin T5 | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |
| 6 | 17 March 2026 (2026-03-17) | T5 label / origin T6 | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |
| 7 | 16 April 2026 (2026-04-16) | T6 label / origin T7 | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |
| 8 | 23 May 2026 (2026-05-23) | T7 label / origin T8 | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |
| 9 | 19 June 2026 (2026-06-19) | T8 label / origin T9 | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |
| 10 | 12 July 2026 (2026-07-12) | T9 label / origin T10 (locked test) | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |
| 11 | 13 August 2026 (2026-08-13) | T10 label only | listings.csv.gz, calendar.csv.gz, reviews.csv.gz |

Place the downloaded files as:
```
01_raw/airbnb/2025-10-05/listings.csv.gz
01_raw/airbnb/2025-10-05/calendar.csv.gz
01_raw/airbnb/2025-10-05/reviews.csv.gz
...
01_raw/airbnb/2026-08-13/reviews.csv.gz
```
Browsers may rename repeated downloads (for example `listings (1).csv.gz`); rename them back to the names above.

**Verify the files.** Running `python verify_snapshots.py Table_S3_snapshot_manifest.csv` compares every file with the SHA-256 checksums of the study files. Only files reported as `OK` will reproduce the published numbers exactly.

## 2. Contextual data (openly licensed)

| Data | Provider, licence and where to obtain it | File used (date of extract) | Place in |
|---|---|---|---|
| Building consents issued by Territorial Authority, monthly, new dwellings (number), Jan 2020 to Jul 2026 | Stats NZ, Infoshare, *Building Consents Issued* (CC BY 4.0): <https://infoshare.stats.govt.nz/> | `statsnz_building_consents_ta_monthly_2020M01_2026M07.csv` (extract of 1 Oct 2026) | `01_raw/construction/` |
| Official release dates of the consent statistics | Stats NZ information releases *Building consents issued: <month year>*: <https://www.stats.govt.nz/information-releases/> | release calendar (13 releases, Jul 2025 to Jul 2026) | `07_provenance/consents_release_calendar.csv` |
| Territorial Authority 2025 boundaries | Stats NZ Geographic Data Service, *Territorial Authority 2025 (generalised)* (CC BY 4.0): <https://datafinder.stats.govt.nz/> | TA 2025 shapefile (accessed 28 Sep 2026) | `01_raw/geography/` |
| Rental bond data by Territorial Authority | MBIE Tenancy Services, *Rental bond data* (CC BY 3.0 NZ): <https://www.tenancy.govt.nz/about-tenancy-services/data-and-statistics/rental-bond-data/> | `detailed-monthly-tla-tenancy-september.csv` (file of 10 Sep 2026) | `01_raw/property_market/` |
| Accommodation Data Programme (seasonality) | MBIE tourism data releases, *Accommodation Data Programme* (<https://www.mbie.govt.nz/>, search "Accommodation Data Programme") | `ADP_All_Measures.csv` (extract of 4 Oct 2026) | `01_raw/seasonality/` |

The processed release-aware construction and rental features are already provided in `Table_S6_shareable_data/ta_transition_context_layer.csv`, and the recomputed consent windows in `Table_S6_shareable_data/R6_ta_transition_construction_cells.csv`. These official series are revised over time, so a later download may differ slightly; the checksums in Table S3 identify the vintages used.

## 3. Reproduce

```
pip install -r File_S1_code/requirements.txt
python verify_snapshots.py Table_S3_snapshot_manifest.csv
# then run the processing and modelling scripts in File_S1_code (order given in File_S1_code/README)
```

To test the code without the source data, use `Table_S6_shareable_data/SYNTHETIC_listing_origin_sample.csv`. It has the panel schema but contains no real listings, so results obtained on it are meaningless.
