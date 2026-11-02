"""
check_R3_excluded.py
Explain why Table A1 (R3, construction strata) covers 292,318 rows while T4-T9 has 293,769 origins.

It reuses revision_addons.py (the script that produced R3_construction_strata.csv) with the SAME
CONFIG you used, so the check follows exactly the same data and the same tercile rule.

Run from the project root (the folder in which you ran revision_addons.py), e.g. in Jupyter:
    import os; os.chdir(r"D:\\...\\FINAL_MDPI_DATA")      # your project root
    %run 10_revision/check_R3_excluded.py 10_revision/revision_addons.py
"""
import importlib.util
import sys

import pandas as pd

ADDONS = sys.argv[1] if len(sys.argv) > 1 else "10_revision/revision_addons.py"
spec = importlib.util.spec_from_file_location("ra", ADDONS)
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)          # loads functions and CONFIG; main() is not executed

C = ra.CONFIG
lev = C["construction_level_col"]
print(f"addons file: {ADDONS}\ntercile column: {lev}")

dev = ra.load_dev()
pan = ra.load_panel()

# --- same tercile construction as construction_strata() in revision_addons.py
tl = pan.groupby(["t", "ta"])[lev].first().reset_index()
tl["stratum"] = tl.groupby("t")[lev].transform(
    lambda x: pd.qcut(x.rank(method="first"), 3, labels=["Low", "Middle", "High"]))

d = (dev[dev.spec == "A1"]
     .merge(pan[["id", "t", "ta"]], on=["id", "t"], how="left")
     .merge(tl[["t", "ta", lev, "stratum"]], on=["t", "ta"], how="left"))

print(f"\nA1 development rows T4-T9: {len(d):,}")
print(f"rows with a tercile:        {d.stratum.notna().sum():,}")
print(f"rows WITHOUT a tercile:     {d.stratum.isna().sum():,}")

miss = d[d.stratum.isna()].copy()
miss["reason"] = "construction value missing for this TA"
miss.loc[miss.ta.isna(), "reason"] = "no TA assigned"
print("\nExcluded rows by reason:")
print(miss.reason.value_counts().to_string())
print("\nExcluded rows by TA code (top 10):")
print(miss.ta.value_counts(dropna=False).head(10).to_string())
print("\nExcluded rows by transition:")
print(miss.groupby("t").size().to_string())

only_999 = set(miss.ta.dropna().astype(str).unique()) <= {"999", "999.0"} and miss.ta.notna().all()
print("\nCONCLUSION:",
      "all excluded rows are TA code 999 (Area Outside Territorial Authority) -> "
      "add the sentence to the Table A1 note." if only_999 else
      "excluded rows are NOT only TA 999 -> send this output before changing the manuscript.")
