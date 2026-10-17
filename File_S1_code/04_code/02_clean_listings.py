# ============================================================
# 02_clean_listings.py
# PRODUCTION — CLEAN ALL 11 AIRBNB LISTINGS SNAPSHOTS
# ============================================================

from pathlib import Path
import sys
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
# 2. INPUT / OUTPUT
# ------------------------------------------------------------

SCHEMA_FILE = (
    INTERMEDIATE /
    "airbnb_listings_final_input_schema.csv"
)

INT_LISTINGS.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FILE = (
    LOGS /
    "02_clean_listings.log"
)

QC_FILE = (
    INTERMEDIATE /
    "airbnb_listings_cleaning_qc.csv"
)


# ------------------------------------------------------------
# 3. READ FROZEN INPUT SCHEMA
# ------------------------------------------------------------

schema = pd.read_csv(SCHEMA_FILE)

usecols = schema["column_name"].tolist()

assert len(usecols) == 35, (
    f"Expected 35 frozen raw input columns, "
    f"found {len(usecols)}"
)


# ------------------------------------------------------------
# 4. HELPER — AMENITIES COUNT
# ------------------------------------------------------------

def count_amenities(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text in ["", "[]"]:
        return 0

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


# ------------------------------------------------------------
# 5. CLEAN ONE SNAPSHOT
# ------------------------------------------------------------

def clean_snapshot(snapshot_date):

    input_file = (
        RAW_AIRBNB /
        snapshot_date /
        "listings.csv.gz"
    )

    output_file = (
        INT_LISTINGS /
        f"listings_{snapshot_date}.parquet"
    )

    print("\n" + "=" * 70)
    print("SNAPSHOT:", snapshot_date)
    print("=" * 70)

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    df = pd.read_csv(
        input_file,
        usecols=usecols,
        low_memory=False
    )

    raw_rows = len(df)

    print("Raw rows:", raw_rows)
    print("Raw columns read:", len(df.columns))


    # --------------------------------------------------------
    # IDENTIFIER
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "id": "listing_id"
        }
    )

    df["snapshot_date"] = pd.to_datetime(
        snapshot_date
    )


    # --------------------------------------------------------
    # DATE VARIABLES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PERCENTAGE VARIABLES
    # --------------------------------------------------------

    percentage_cols = [
        "host_response_rate",
        "host_acceptance_rate"
    ]

    for col in percentage_cols:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype("string")
                .str.replace(
                    "%",
                    "",
                    regex=False
                )
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ) / 100.0


    # --------------------------------------------------------
    # BOOLEAN VARIABLES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PRICE
    # PROCESSING-ONLY FIELD
    # --------------------------------------------------------

    if "price" in df.columns:

        df["price_raw"] = df["price"]

        df["price"] = (
            df["price"]
            .astype("string")
            .str.replace(
                "$",
                "",
                regex=False
            )
            .str.replace(
                ",",
                "",
                regex=False
            )
        )

        df["price"] = pd.to_numeric(
            df["price"],
            errors="coerce"
        )


    # --------------------------------------------------------
    # AMENITIES COUNT
    # --------------------------------------------------------

    if "amenities" in df.columns:

        df["amenities_count"] = (
            df["amenities"]
            .apply(count_amenities)
            .astype("Int32")
        )


    # --------------------------------------------------------
    # REVIEW-HISTORY FEATURES
    #
    # IMPORTANT:
    # snapshot_date remains the rolling-origin snapshot label.
    #
    # last_scraped is the listing-specific observation-time
    # reference for review recency.
    # --------------------------------------------------------

    if "first_review" in df.columns:

        df["listing_review_age_days"] = (
            df["last_scraped"]
            -
            df["first_review"]
        ).dt.days


    if "last_review" in df.columns:

        df["days_since_last_review"] = (
            df["last_scraped"]
            -
            df["last_review"]
        ).dt.days


    # --------------------------------------------------------
    # QC — IDENTIFIER
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # QC — TEMPORAL
    # --------------------------------------------------------

    negative_review_age = (
        df["listing_review_age_days"] < 0
    ).sum()

    negative_days_since_review = (
        df["days_since_last_review"] < 0
    ).sum()


    # --------------------------------------------------------
    # QC — PRICE
    # --------------------------------------------------------

    price_nonmissing = (
        df["price"]
        .notna()
        .sum()
    )

    price_missing_pct = (
        df["price"]
        .isna()
        .mean()
        * 100
    )


    # --------------------------------------------------------
    # HARD QC
    # --------------------------------------------------------

    assert len(df) == raw_rows, (
        f"{snapshot_date}: "
        "Row count changed during cleaning."
    )

    assert missing_listing_id == 0, (
        f"{snapshot_date}: "
        "Missing listing_id detected."
    )

    assert duplicate_listing_id == 0, (
        f"{snapshot_date}: "
        "Duplicate listing_id detected."
    )

    assert negative_review_age == 0, (
        f"{snapshot_date}: "
        "Negative listing_review_age_days detected."
    )

    assert negative_days_since_review == 0, (
        f"{snapshot_date}: "
        "Negative days_since_last_review detected."
    )

    assert len(df.columns) == 40, (
        f"{snapshot_date}: "
        f"Expected 40 cleaned columns, "
        f"found {len(df.columns)}."
    )


    # --------------------------------------------------------
    # SAVE PARQUET
    # --------------------------------------------------------

    df.to_parquet(
        output_file,
        index=False
    )


    # --------------------------------------------------------
    # PRINT SNAPSHOT QC
    # --------------------------------------------------------

    print("Rows after cleaning:", len(df))
    print("Unique listing IDs:", unique_listing_id)
    print("Missing listing IDs:", missing_listing_id)
    print("Duplicate listing IDs:", duplicate_listing_id)

    print(
        "Negative listing_review_age_days:",
        negative_review_age
    )

    print(
        "Negative days_since_last_review:",
        negative_days_since_review
    )

    print(
        f"Price missing: "
        f"{price_missing_pct:.2f}%"
    )

    print("Cleaned columns:", len(df.columns))
    print("STATUS: PASS")

    print("Saved:")
    print(output_file)


    # --------------------------------------------------------
    # RETURN QC RECORD
    # --------------------------------------------------------

    return {
        "snapshot_date":
            snapshot_date,

        "raw_rows":
            raw_rows,

        "cleaned_rows":
            len(df),

        "unique_listing_ids":
            unique_listing_id,

        "missing_listing_ids":
            missing_listing_id,

        "duplicate_listing_ids":
            duplicate_listing_id,

        "negative_listing_review_age_days":
            negative_review_age,

        "negative_days_since_last_review":
            negative_days_since_review,

        "price_nonmissing_n":
            price_nonmissing,

        "price_missing_pct":
            round(
                price_missing_pct,
                2
            ),

        "cleaned_columns":
            len(df.columns),

        "output_file":
            str(output_file)
    }


# ------------------------------------------------------------
# 6. RUN ALL 11 SNAPSHOTS
# ------------------------------------------------------------

print("=" * 70)
print("AIRBNB LISTINGS — PRODUCTION CLEANING")
print("=" * 70)

print("Snapshots:", len(SNAPSHOT_DATES))
print("Frozen raw input columns:", len(usecols))

qc_records = []

for snapshot_date in SNAPSHOT_DATES:

    result = clean_snapshot(
        snapshot_date
    )

    qc_records.append(
        result
    )


# ------------------------------------------------------------
# 7. QC SUMMARY
# ------------------------------------------------------------

qc_summary = pd.DataFrame(
    qc_records
)

qc_summary.to_csv(
    QC_FILE,
    index=False
)


# ------------------------------------------------------------
# 8. FINAL PRODUCTION QC
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL PRODUCTION QC")
print("=" * 70)

print(
    "Snapshots processed:",
    len(qc_summary)
)

print(
    "Total cleaned listing rows:",
    qc_summary[
        "cleaned_rows"
    ].sum()
)

print(
    "Snapshots with missing listing IDs:",
    (
        qc_summary[
            "missing_listing_ids"
        ] > 0
    ).sum()
)

print(
    "Snapshots with duplicate listing IDs:",
    (
        qc_summary[
            "duplicate_listing_ids"
        ] > 0
    ).sum()
)

print(
    "Snapshots with negative review age:",
    (
        qc_summary[
            "negative_listing_review_age_days"
        ] > 0
    ).sum()
)

print(
    "Snapshots with negative review recency:",
    (
        qc_summary[
            "negative_days_since_last_review"
        ] > 0
    ).sum()
)

print(
    "Snapshots with wrong output column count:",
    (
        qc_summary[
            "cleaned_columns"
        ] != 40
    ).sum()
)


# ------------------------------------------------------------
# 9. FINAL ASSERTIONS
# ------------------------------------------------------------

assert len(qc_summary) == 11

assert (
    qc_summary[
        "missing_listing_ids"
    ] == 0
).all()

assert (
    qc_summary[
        "duplicate_listing_ids"
    ] == 0
).all()

assert (
    qc_summary[
        "negative_listing_review_age_days"
    ] == 0
).all()

assert (
    qc_summary[
        "negative_days_since_last_review"
    ] == 0
).all()

assert (
    qc_summary[
        "cleaned_columns"
    ] == 40
).all()


# ------------------------------------------------------------
# 10. WRITE LOG
# ------------------------------------------------------------

with open(
    LOG_FILE,
    "w",
    encoding="utf-8"
) as log:

    log.write(
        "AIRBNB LISTINGS PRODUCTION CLEANING\n"
    )

    log.write(
        "=" * 70 + "\n\n"
    )

    log.write(
        qc_summary.to_string(
            index=False
        )
    )

    log.write(
        "\n\nFINAL STATUS: PASS\n"
    )


# ------------------------------------------------------------
# 11. COMPLETE
# ------------------------------------------------------------

print("\nSaved QC summary:")
print(QC_FILE)

print("\nSaved log:")
print(LOG_FILE)

print("\n" + "=" * 70)
print("PRODUCTION CLEANING STATUS: PASS")
print("=" * 70)