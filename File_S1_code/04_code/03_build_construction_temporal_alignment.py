from pathlib import Path
import pandas as pd

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\yxt2919\7th_NZ_STR_Construction_AI"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "07_provenance"
    / "construction_temporal_alignment.csv"
)

# ============================================================
# AIRBNB ANCHOR DATES
# T1-T10 = predictor dates
# ============================================================

snapshots = pd.DataFrame({
    "transition": [
        "T1", "T2", "T3", "T4", "T5",
        "T6", "T7", "T8", "T9", "T10"
    ],

    "airbnb_anchor_date": pd.to_datetime([
        "2025-10-05",
        "2025-11-07",
        "2025-12-11",
        "2026-01-16",
        "2026-02-13",
        "2026-03-17",
        "2026-04-16",
        "2026-05-23",
        "2026-06-19",
        "2026-07-12"
    ])
})

# ============================================================
# VERIFIED STATS NZ RELEASE CALENDAR
#
# These are actual publication dates.
# They are NOT estimated 1-month or 2-month lags.
# ============================================================

release_calendar = pd.DataFrame({

    "consent_month": [
        "2025M07",
        "2025M08",
        "2025M09",
        "2025M10",
        "2025M11",
        "2025M12",
        "2026M01",
        "2026M02",
        "2026M03",
        "2026M04",
        "2026M05",
        "2026M06",
        "2026M07"
    ],

    "release_date": pd.to_datetime([
        "2025-09-01",  # July 2025
        "2025-10-01",  # August 2025
        "2025-11-03",  # September 2025
        "2025-12-01",  # October 2025
        "2026-01-14",  # November 2025
        "2026-02-03",  # December 2025
        "2026-03-03",  # January 2026
        "2026-04-01",  # February 2026
        "2026-05-01",  # March 2026
        "2026-06-03",  # April 2026
        "2026-07-02",  # May 2026
        "2026-08-03",  # June 2026
        "2026-09-02"   # July 2026
    ])
})

# ============================================================
# QC RELEASE CALENDAR
# ============================================================
# Convert Stats NZ format YYYYMMM, e.g. 2025M07,
# explicitly to a monthly pandas Period.
# Do not let pandas guess the date format.

release_calendar["consent_period"] = pd.PeriodIndex(
    pd.to_datetime(
        release_calendar["consent_month"],
        format="%YM%m"
    ),
    freq="M"
)
# ============================================================
# POINT-IN-TIME MATCHING
# ============================================================

results = []

for _, row in snapshots.iterrows():

    transition = row["transition"]
    anchor_date = row["airbnb_anchor_date"]

    # Only information actually published by anchor date
    available = release_calendar[
        release_calendar["release_date"] <= anchor_date
    ].copy()

    if available.empty:
        raise RuntimeError(
            f"No consent release available for {transition}"
        )

    # Latest consent reference month available at that time
    latest_period = available["consent_period"].max()

    selected = available[
        available["consent_period"] == latest_period
    ].sort_values("release_date").iloc[0]

    release_date = selected["release_date"]

    # --------------------------------------------------------
    # HARD LEAKAGE CHECK
    # --------------------------------------------------------

    if release_date > anchor_date:
        raise RuntimeError(
            f"TEMPORAL LEAKAGE: {transition}"
        )

    lag_days = (anchor_date - release_date).days

    results.append({
        "transition": transition,
        "airbnb_anchor_date": anchor_date.date(),
        "latest_available_consent_month":
            selected["consent_month"],
        "official_release_date":
            release_date.date(),
        "information_lag_days":
            lag_days,
        "temporal_leakage":
            False
    })

# ============================================================
# OUTPUT
# ============================================================

alignment = pd.DataFrame(results)

# Final QC
assert len(alignment) == 10
assert alignment["temporal_leakage"].eq(False).all()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

alignment.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nCONSTRUCTION TEMPORAL ALIGNMENT")
print("=" * 95)

print(
    alignment.to_string(index=False)
)

print("=" * 95)

print("\nQC")
print("Transitions:", len(alignment))
print(
    "Temporal leakage:",
    alignment["temporal_leakage"].sum()
)

print(
    "\nSTATUS: PASS"
    if alignment["temporal_leakage"].sum() == 0
    else "\nSTATUS: FAIL"
)

print(
    "\nSaved to:",
    OUTPUT_FILE
)