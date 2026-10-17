from config import *

import csv
import gzip
from datetime import datetime


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_FILES = [
    "listings.csv.gz",
    "calendar.csv.gz",
    "reviews.csv.gz",
]

INVENTORY_OUT = INTERMEDIATE / "airbnb_inventory.csv"
COLUMNS_OUT = INTERMEDIATE / "airbnb_columns_by_snapshot.csv"
LOG_OUT = LOGS / "01_inventory_airbnb.log"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def count_rows_gzip(file_path):
    """
    Count actual CSV records in a gzip CSV without loading
    the whole file into RAM.

    Uses csv.reader so embedded line breaks inside quoted text
    are not incorrectly counted as separate records.
    """
    with gzip.open(
        file_path,
        "rt",
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as f:
        reader = csv.reader(f)

        # Skip header
        next(reader, None)

        # Count actual CSV records
        row_count = sum(1 for _ in reader)

    return row_count


def read_header_gzip(file_path):
    """
    Read only the CSV header.
    """
    with gzip.open(file_path, "rt", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        return next(reader)


# ============================================================
# MAIN INVENTORY
# ============================================================

inventory_rows = []
column_rows = []
log_lines = []

start_time = datetime.now()

log_lines.append("=" * 70)
log_lines.append("AIRBNB RAW DATA INVENTORY")
log_lines.append(f"Started: {start_time}")
log_lines.append(f"Project root: {PROJECT_ROOT}")
log_lines.append("=" * 70)

print("=" * 70)
print("AIRBNB RAW DATA INVENTORY")
print("=" * 70)

for snapshot_date in SNAPSHOT_DATES:

    snapshot_folder = RAW_AIRBNB / snapshot_date

    print(f"\nSnapshot: {snapshot_date}")
    log_lines.append(f"\nSnapshot: {snapshot_date}")

    for file_name in EXPECTED_FILES:

        file_path = snapshot_folder / file_name
        exists = file_path.exists()

        if not exists:
            print(f"  MISSING: {file_name}")
            log_lines.append(f"  MISSING: {file_name}")

            inventory_rows.append({
                "snapshot_date": snapshot_date,
                "file_name": file_name,
                "exists": False,
                "size_mb": None,
                "row_count": None,
                "column_count": None,
            })

            continue

        size_mb = file_path.stat().st_size / (1024 ** 2)

        try:
            columns = read_header_gzip(file_path)
            column_count = len(columns)

            print(
                f"  FOUND: {file_name} | "
                f"{size_mb:.2f} MB | "
                f"{column_count} columns | counting rows..."
            )

            row_count = count_rows_gzip(file_path)

            print(f"         rows = {row_count:,}")

            log_lines.append(
                f"  FOUND: {file_name} | "
                f"{size_mb:.2f} MB | "
                f"{column_count} columns | "
                f"{row_count:,} rows"
            )

            inventory_rows.append({
                "snapshot_date": snapshot_date,
                "file_name": file_name,
                "exists": True,
                "size_mb": round(size_mb, 2),
                "row_count": row_count,
                "column_count": column_count,
            })

            for position, column_name in enumerate(columns, start=1):
                column_rows.append({
                    "snapshot_date": snapshot_date,
                    "file_name": file_name,
                    "column_position": position,
                    "column_name": column_name,
                })

        except Exception as e:

            print(f"  ERROR reading {file_name}: {e}")
            log_lines.append(f"  ERROR reading {file_name}: {e}")

            inventory_rows.append({
                "snapshot_date": snapshot_date,
                "file_name": file_name,
                "exists": True,
                "size_mb": round(size_mb, 2),
                "row_count": None,
                "column_count": None,
            })


# ============================================================
# WRITE OUTPUT FILES
# ============================================================

with open(INVENTORY_OUT, "w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "snapshot_date",
        "file_name",
        "exists",
        "size_mb",
        "row_count",
        "column_count",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(inventory_rows)


with open(COLUMNS_OUT, "w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "snapshot_date",
        "file_name",
        "column_position",
        "column_name",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(column_rows)


# ============================================================
# QC SUMMARY
# ============================================================

expected_total = len(SNAPSHOT_DATES) * len(EXPECTED_FILES)
found_total = sum(row["exists"] for row in inventory_rows)
missing_total = expected_total - found_total

log_lines.append("\n" + "=" * 70)
log_lines.append("QC SUMMARY")
log_lines.append("=" * 70)
log_lines.append(f"Expected files : {expected_total}")
log_lines.append(f"Found files    : {found_total}")
log_lines.append(f"Missing files  : {missing_total}")

print("\n" + "=" * 70)
print("QC SUMMARY")
print("=" * 70)
print(f"Expected files : {expected_total}")
print(f"Found files    : {found_total}")
print(f"Missing files  : {missing_total}")

if missing_total == 0:
    print("FILE CHECK     : PASS")
    log_lines.append("FILE CHECK     : PASS")
else:
    print("FILE CHECK     : FAIL")
    log_lines.append("FILE CHECK     : FAIL")


end_time = datetime.now()
elapsed = end_time - start_time

print(f"Runtime        : {elapsed}")
print("=" * 70)

log_lines.append(f"Finished: {end_time}")
log_lines.append(f"Runtime: {elapsed}")
log_lines.append("=" * 70)

with open(LOG_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))


print("\nSaved:")
print(INVENTORY_OUT)
print(COLUMNS_OUT)
print(LOG_OUT)