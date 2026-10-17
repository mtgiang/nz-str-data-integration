#!/usr/bin/env python
"""
Assemble the MDPI Data Supplementary Materials (Tables S1-S6, File S1) and a Zenodo-ready copy,
with an Inside-Airbnb policy check. Nothing is modified in place; outputs go to 12_supplementary/.

Run from the project root (folder containing 01_raw/):
    python 11_final/build_supplementary.py
Prerequisites: FINAL_MDPI_DATA/ (from the audit) and 10_revision_out/shareable_data/ (from
revision_addons.py). Missing items are reported, never invented.
"""
import hashlib, json, re, shutil, sys, zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(".").resolve()
OUT = ROOT / "12_supplementary"
SUPP = OUT / "Supplementary_Materials"

# Each table: list of filename patterns searched (in this order) under these folders.
SEARCH_DIRS = ["FINAL_MDPI_DATA", "10_revision_out", "07_provenance", "05_results"]
TABLES = {
    "Table_S1_feature_provenance_register": ["FEATURE_PROVENANCE_REGISTER.csv"],
    "Table_S2_point_in_time_audit": ["POINT_IN_TIME_AUDIT.csv"],
    "Table_S3_snapshot_manifest": ["SNAPSHOT_MANIFEST.csv"],
    "Table_S4_data_dictionary": ["DATA_DICTIONARY.csv"],
    "Table_S5_data_source_inventory": ["DATA_SOURCE_INVENTORY.csv"],
}
EXTRA_DOCS = ["LICENCES_AND_ACCESS.md", "LEAKAGE_RISK_REPORT.md", "RESULTS_VERIFICATION_REPORT.md", "DATA_LINEAGE_AUDIT.md"]
S6_DIR = ROOT / "10_revision_out" / "shareable_data"
CODE_DIRS = ["FINAL_MDPI_DATA/reproducibility/scripts", "04_code", "09_pipeline", "10_revision", "11_final"]
CODE_EXT = {".py", ".ipynb", ".R", ".sql", ".yaml", ".yml", ".txt", ".md", ".toml", ".cfg"}

# Column names that would indicate listing-level / personal data. Metadata tables (S1-S5) legitimately have
# columns such as "name" or "description" for variables, so those generic words are not blocked here.
FORBIDDEN_COL = re.compile(r"(^id$|listing_id|host_id|host_name|host_url|reviewer|^comments?$|picture|thumbnail|"
                           r"listing_url|latitude|longitude|^lat$|^lon$|^lng$|neighborhood_overview|host_about)", re.I)
FORBIDDEN_TEXT = re.compile(r"airbnb\.[a-z.]+/rooms/\d+|muscache\.com", re.I)
COUNT_NAME = re.compile(r"^(n_|num_|count_|unique_|duplicate_|n_unique|n_duplicate)", re.I)
MAX_CODE_FILE_MB = 2.0
# personal / submission documents that must not be published with the code
EXCLUDE_NAME = re.compile(r"(cover_letter|huong_dan|guide_packaging|checklist|changes|readme_run_private|main\.tex)", re.I)


def find(pattern):
    for d in SEARCH_DIRS:
        hits = sorted((ROOT / d).rglob(pattern)) if (ROOT / d).exists() else []
        if hits:
            return hits[0]
    return None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def strip_notebook(src, dst):
    """Copy a notebook without outputs (outputs may print listing-level rows)."""
    nb = json.loads(Path(src).read_text(encoding="utf-8"))
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            c["outputs"], c["execution_count"] = [], None
    Path(dst).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def policy_scan(folder):
    problems = []
    for f in folder.rglob("*"):
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        try:
            if suf in (".csv", ".parquet", ".xlsx"):
                cols = (pd.read_csv(f, nrows=0).columns if suf == ".csv" else
                        pd.read_parquet(f).columns if suf == ".parquet" else pd.read_excel(f, nrows=0).columns)
                bad = [c for c in cols if FORBIDDEN_COL.search(str(c)) and str(c) != "synthetic_id"]
                if bad and suf == ".csv":
                    # columns such as "unique_listing_id" in the snapshot manifest hold COUNTS, not identifiers:
                    # allowed only when the name says it is a count AND every value is a small number
                    df_ = pd.read_csv(f, usecols=bad)
                    def is_count(c):
                        # empty cells are allowed (e.g. rows for Stats NZ / MBIE files have no listing counts);
                        # every non-empty value must be a number far smaller than an Airbnb listing ID
                        raw = df_[c].dropna().astype(str).str.strip()
                        raw = raw[raw != ""]
                        v = pd.to_numeric(raw, errors="coerce")
                        return (COUNT_NAME.search(str(c)) is not None and len(df_) <= 1000
                                and v.notna().all() and (len(v) == 0 or v.abs().max() < 1e8))
                    allowed = [c for c in bad if is_count(c)]
                    if allowed:
                        print(f"  allowed count columns in {f.name}: {allowed}")
                    bad = [c for c in bad if c not in allowed]
                if bad:
                    problems.append(f"{f.relative_to(folder)}: forbidden columns {bad}")
            if suf in (".csv", ".md", ".txt", ".py", ".ipynb", ".json", ".yaml", ".yml"):
                txt = f.read_text(encoding="utf-8", errors="ignore")
                if FORBIDDEN_TEXT.search(txt):
                    problems.append(f"{f.relative_to(folder)}: contains Airbnb listing/photo URLs")
            if suf in (".gz", ".zip") and "Table_S6" not in str(f):
                problems.append(f"{f.relative_to(folder)}: archive inside package (possible raw data)")
        except Exception as e:
            problems.append(f"{f.relative_to(folder)}: could not scan ({e})")
    return problems


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    SUPP.mkdir(parents=True)
    missing, rows = [], []

    # Tables S1-S5 (as CSV + one XLSX workbook for reviewers)
    xlsx = pd.ExcelWriter(SUPP / "Tables_S1-S5.xlsx", engine="openpyxl")
    for name, pats in TABLES.items():
        src = next((find(p) for p in pats if find(p)), None)
        if src is None:
            missing.append(name); continue
        df = pd.read_csv(src)
        df.to_csv(SUPP / f"{name}.csv", index=False)
        df.to_excel(xlsx, sheet_name=name.split("_")[1], index=False)
        rows.append((name, src))
    xlsx.close()
    docs = SUPP / "Audit_reports"; docs.mkdir()
    for d in EXTRA_DOCS:
        src = find(d)
        if src:
            shutil.copy2(src, docs / d)

    # Table S6: shareable derived data products
    if S6_DIR.exists():
        shutil.copytree(S6_DIR, SUPP / "Table_S6_shareable_data")
    else:
        missing.append("Table_S6 (run 10_revision/revision_addons.py first)")

    # File S1: code only (notebook outputs stripped, data files excluded)
    code = SUPP / "File_S1_code"
    for d in CODE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file() or f.suffix not in CODE_EXT or "__pycache__" in f.parts or ".ipynb_checkpoints" in f.parts:
                continue
            if EXCLUDE_NAME.search(f.name) or (d == "11_final" and f.suffix != ".py"):
                if f.name not in ("DATA_ACCESS.md",):
                    print(f"not published (submission document): {f}")
                continue
            if f.suffix != ".ipynb" and f.stat().st_size > MAX_CODE_FILE_MB * 1e6:
                print(f"skip large file {f}"); continue
            dst = code / d / f.relative_to(base); dst.parent.mkdir(parents=True, exist_ok=True)
            if f.suffix == ".ipynb":
                strip_notebook(f, dst)
                if dst.stat().st_size > MAX_CODE_FILE_MB * 1e6:
                    dst.unlink(); print(f"skip large notebook (even without outputs) {f}")
            else:
                shutil.copy2(f, dst)
    req = find("requirements.txt") or (ROOT / "09_pipeline" / "requirements.txt")
    if req and Path(req).exists():
        shutil.copy2(req, code / "requirements.txt")

    for extra in ("DATA_ACCESS.md", "verify_snapshots.py"):
        src = ROOT / "11_final" / extra
        if src.exists():
            shutil.copy2(src, SUPP / extra)
        else:
            missing.append(extra)
    (SUPP / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (SUPP / "README.md").write_text(README, encoding="utf-8")
    (SUPP / "CITATION.cff").write_text(CITATION, encoding="utf-8")

    problems = policy_scan(SUPP)
    if problems:
        print("POLICY CHECK FAILED -- fix before sharing:\n  " + "\n  ".join(problems)); sys.exit(1)

    man = [{"file": str(f.relative_to(SUPP)).replace("\\", "/"), "bytes": f.stat().st_size, "sha256": sha(f)}
           for f in sorted(SUPP.rglob("*")) if f.is_file()]
    pd.DataFrame(man).to_csv(SUPP / "CHECKSUMS.csv", index=False)

    z = OUT / "Supplementary_Materials.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in SUPP.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(OUT))
    print(f"\nSources used:")
    for n, s in rows:
        print(f"  {n} <- {s.relative_to(ROOT)}")
    print(f"\nPolicy check passed. {len(man)} files, {z.stat().st_size/1e6:.1f} MB -> {z}")
    if missing:
        print("MISSING (not invented; manuscript must match):", missing)


README = """# Supplementary Materials

Article: A Multi-Source Data Integration and Temporal Evaluation Framework for Short-Term Rental Market
Screening: Evidence from New Zealand (MDPI *Data*).

| Item | Content |
|---|---|
| Table S1 | Feature provenance register (53 predictors) |
| Table S2 | Point-in-time audit (53 predictors x 10 transitions x 2 cutoffs) |
| Table S3 | Snapshot manifest with SHA-256 checksums of all source files |
| Table S4 | Data dictionary |
| Table S5 | Data-source inventory and licences |
| Table S6 | Shareable derived data products: TA x transition aggregates (cells < 10 listings suppressed), release-aware construction and rental context layer, metric and ablation tables, synthetic schema sample |
| File S1 | Analysis, audit, verification and revision code (notebook outputs removed) |
| Audit_reports | Lineage, leakage, results-verification and licence reports |
| DATA_ACCESS.md | Where to obtain each source file (Inside Airbnb snapshot URLs, archived-data request, Stats NZ/MBIE) |
| verify_snapshots.py | Checks locally obtained snapshots against the SHA-256 in Table S3 |

## Data policy
Inside Airbnb asks users not to republish its data. This package contains **no listing-level records,
listing or host identifiers, coordinates, URLs or review text**; an automated check enforces this.
The synthetic sample is generated by resampling each column independently and contains no real listing.

## Reproducing listing-level results
1. Obtain the eleven Inside Airbnb New Zealand snapshots listed in Table S3 (insideairbnb.com/get-the-data;
   earlier snapshots via an archived-data request).
2. Verify each file against the SHA-256 in Table S3.
3. Place them as `01_raw/airbnb/<YYYY-MM-DD>/{listings,calendar,reviews}.csv.gz`, add the Stats NZ and MBIE
   files listed in Table S5, install `File_S1_code/requirements.txt`, and run the scripts in File S1.

## Licences
Derived tables and documentation: CC BY 4.0. Code: MIT. Source data remain under their providers' terms
(Inside Airbnb; Stats NZ CC BY 4.0; MBIE CC BY 3.0 NZ). Please cite the article and Inside Airbnb.
"""

GITIGNORE = """# Never commit source data (Inside Airbnb asks users not to republish it)
01_raw/
*.csv.gz
*.parquet
02_intermediate/
03_processed/
__pycache__/
.ipynb_checkpoints/
"""

CITATION = """cff-version: 1.2.0
message: "If you use these materials, please cite the article."
title: "Supplementary materials: Multi-source data integration and temporal evaluation for short-term rental screening (New Zealand)"
authors:
  - family-names: Mai
    given-names: Thu Giang
  - family-names: Nguyen
    given-names: Minh
  - family-names: Ghobakhlou
    given-names: Akbar
  - family-names: Chhun
    given-names: Bunleng
  - family-names: Nguyen
    given-names: Thanh
license: CC-BY-4.0
"""

if __name__ == "__main__":
    main()
