# ============================================================
# 02_clean_listings.py
# STEP 11.2 — TEST CLEANING ON ONE SNAPSHOT
# Snapshot: 2025-10-05
# ============================================================

from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. LOAD PROJECT CONFIG
# ------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import *


# ------------------------------------------------------------
# 2. TEST SNAPSHOT
# ------------------------------------------------------------

SNAPSHOT_DATE = "2025-10-05"

INPUT_FILE = (
    RAW_AIRBNB /
    SNAPSHOT_DATE /
    "listings.csv.gz"
)

SCHEMA_FILE = (
    INTERMEDIATE /
    "airbnb_listings_final_input_schema.csv"
)

OUTPUT_FILE = (
    INT_LISTINGS /
    f"listings_{SNAPSHOT_DATE}.parquet"
)


# ------------------------------------------------------------
# 3. READ FROZEN INPUT SCHEMA
# ------------------------------------------------------------

schema = pd.read_csv(SCHEMA_FILE)

usecols = schema["column_name"].tolist()

print("=" * 65)
print("LISTINGS CLEANING TEST")
print("=" * 65)
print("Snapshot:", SNAPSHOT_DATE)
print("Frozen raw columns:", len(usecols))

assert len(usecols) == 35, (
    f"Expected 35 frozen input columns, found {len(usecols)}"
)


# ------------------------------------------------------------
# 4. READ RAW LISTINGS
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    usecols=usecols,
    low_memory=False
)

raw_rows = len(df)

print("Raw rows:", raw_rows)
print("Columns read:", len(df.columns))


# ------------------------------------------------------------
# 5. BASIC IDENTIFIER PROCESSING
# ------------------------------------------------------------

df = df.rename(
    columns={
        "id": "listing_id"
    }
)

df["snapshot_date"] = pd.to_datetime(
    SNAPSHOT_DATE
)


# ------------------------------------------------------------
# 6. PARSE DATES
# ------------------------------------------------------------

date_cols = [
    "last_scraped",
    "host_since",
    "first_review",
    "last_review"
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )


# ------------------------------------------------------------
# 7. CLEAN PERCENTAGE VARIABLES
# ------------------------------------------------------------

percentage_cols = [
    "host_response_rate",
    "host_acceptance_rate"
]

for col in percentage_cols:
    if col in df.columns:

        df[col] = (
            df[col]
            .astype("string")
            .str.replace("%", "", regex=False)
        )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ) / 100.0


# ------------------------------------------------------------
# 8. CLEAN BOOLEAN VARIABLES
# ------------------------------------------------------------

boolean_cols = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "has_availability",
    "instant_bookable"
]

boolean_map = {
    "t": 1,
    "f": 0,
    "true": 1,
    "false": 0
}

for col in boolean_cols:
    if col in df.columns:

        df[col] = (
            df[col]
            .astype("string")
            .str.lower()
            .map(boolean_map)
            .astype("Int8")
        )


# ------------------------------------------------------------
# 9. CLEAN PRICE
# Processing-only field.
# NOT automatically used as a predictor.
# ------------------------------------------------------------

if "price" in df.columns:

    df["price_raw"] = df["price"]

    df["price"] = (
        df["price"]
        .astype("string")
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )


# ------------------------------------------------------------
# 10. DERIVE AMENITIES COUNT
# ------------------------------------------------------------

if "amenities" in df.columns:

    def count_amenities(value):

        if pd.isna(value):
            return np.nan

        text = str(value).strip()

        if text in ["", "[]"]:
            return 0

        # Count comma-separated entries inside Airbnb list string
        inner = text.strip("[]").strip()

        if inner == "":
            return 0

        return len(
            [
                item
                for item in inner.split(",")
                if item.strip() != ""
            ]
        )

    df["amenities_count"] = (
        df["amenities"]
        .apply(count_amenities)
        .astype("Int32")
    )



# ------------------------------------------------------------
# 11. DERIVE REVIEW-HISTORY FEATURES
# ------------------------------------------------------------
# IMPORTANT:
# snapshot_date remains the fixed rolling-origin snapshot label.
# last_scraped is used only as the listing-specific observation
# reference date for review-age/recency calculations.
# This avoids artificial negative recency caused by the archive
# date preceding the actual listing scrape date.
# ------------------------------------------------------------

if "first_review" in df.columns:

    df["listing_review_age_days"] = (
        df["last_scraped"] -
        df["first_review"]
    ).dt.days


if "last_review" in df.columns:

    df["days_since_last_review"] = (
        df["last_scraped"] -
        df["last_review"]
    ).dt.days

# ------------------------------------------------------------
# 12. QC — IDENTIFIERS
# ------------------------------------------------------------

missing_listing_id = (
    df["listing_id"]
    .isna()
    .sum()
)

duplicate_listing_id = (
    df["listing_id"]
    .duplicated()
    .sum()
)

unique_listing_id = (
    df["listing_id"]
    .nunique()
)


# ------------------------------------------------------------
# 13. QC — TEMPORAL LOGIC
# ------------------------------------------------------------

negative_review_age = (
    df["listing_review_age_days"] < 0
).sum()

negative_days_since_review = (
    df["days_since_last_review"] < 0
).sum()


# ------------------------------------------------------------
# 14. QC — PRICE
# ------------------------------------------------------------

price_nonmissing = (
    df["price"]
    .notna()
    .sum()
)

price_missing_pct = (
    df["price"]
    .isna()
    .mean() * 100
)


# ------------------------------------------------------------
# 15. FINAL TEST SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("CLEANING QC")
print("=" * 65)

print("Rows before cleaning :", raw_rows)
print("Rows after cleaning  :", len(df))
print("Unique listing IDs   :", unique_listing_id)
print("Missing listing IDs  :", missing_listing_id)
print("Duplicate listing IDs:", duplicate_listing_id)

print("\nDerived:")
print(
    "amenities_count non-missing:",
    df["amenities_count"].notna().sum()
)

print(
    "listing_review_age_days non-missing:",
    df["listing_review_age_days"].notna().sum()
)

print(
    "days_since_last_review non-missing:",
    df["days_since_last_review"].notna().sum()
)

print("\nTemporal QC:")
print(
    "Negative listing_review_age_days:",
    negative_review_age
)

print(
    "Negative days_since_last_review:",
    negative_days_since_review
)

print("\nPrice QC:")
print("Price non-missing:", price_nonmissing)
print(
    f"Price missing: {price_missing_pct:.2f}%"
)

print("\nCurrent columns:", len(df.columns))


# ------------------------------------------------------------
# 16. HARD QC
# ------------------------------------------------------------

assert len(df) == raw_rows, (
    "ERROR: Row count changed during cleaning."
)

assert missing_listing_id == 0, (
    "ERROR: Missing listing_id detected."
)

assert duplicate_listing_id == 0, (
    "ERROR: Duplicate listing_id detected."
)


# ------------------------------------------------------------
# 17. SAVE TEST PARQUET
# ------------------------------------------------------------

INT_LISTINGS.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("\nTEST STATUS: PASS")
print("Saved:")
print(OUTPUT_FILE)