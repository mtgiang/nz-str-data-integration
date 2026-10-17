from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# MAIN DIRECTORIES
# ============================================================

PROTOCOL = PROJECT_ROOT / "00_protocol"
RAW = PROJECT_ROOT / "01_raw"
INTERMEDIATE = PROJECT_ROOT / "02_intermediate"
PROCESSED = PROJECT_ROOT / "03_processed"
CODE = PROJECT_ROOT / "04_code"
RESULTS = PROJECT_ROOT / "05_results"
MANUSCRIPT = PROJECT_ROOT / "06_manuscript"
PROVENANCE = PROJECT_ROOT / "07_provenance"
LOGS = PROJECT_ROOT / "08_logs"

# ============================================================
# RAW DATA
# ============================================================

RAW_AIRBNB = RAW / "airbnb"
RAW_CONSTRUCTION = RAW / "construction"
RAW_PROPERTY_MARKET = RAW / "property_market"
RAW_SEASONALITY = RAW / "seasonality"
RAW_GEOGRAPHY = RAW / "geography"

# ============================================================
# INTERMEDIATE DATA
# ============================================================

INT_LISTINGS = INTERMEDIATE / "listings"
INT_REVIEWS = INTERMEDIATE / "reviews"
INT_CALENDAR = INTERMEDIATE / "calendar"
INT_GEOGRAPHY = INTERMEDIATE / "geography"
INT_CONSTRUCTION = INTERMEDIATE / "construction"
INT_PROPERTY_MARKET = INTERMEDIATE / "property_market"
INT_SEASONALITY = INTERMEDIATE / "seasonality"

# ============================================================
# RESULTS
# ============================================================

TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"

# ============================================================
# PROVENANCE
# ============================================================

SCREENSHOTS = PROVENANCE / "screenshots"
LICENCES = PROVENANCE / "licences"

# ============================================================
# AIRBNB SNAPSHOTS
# ============================================================

SNAPSHOT_DATES = [
    "2025-10-05",
    "2025-11-07",
    "2025-12-11",
    "2026-01-16",
    "2026-02-13",
    "2026-03-17",
    "2026-04-16",
    "2026-05-23",
    "2026-06-19",
    "2026-07-12",
    "2026-08-13",
]