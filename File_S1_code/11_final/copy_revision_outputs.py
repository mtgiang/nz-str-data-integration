"""
Copy the aggregate revision outputs into the shareable data package (Table S6).
Run from the project root, e.g. in Jupyter:
    import os; os.chdir(r"C:\\Users\\yxt2919\\7th_NZ_STR_Construction_AI")
    %run 11_final/copy_revision_outputs.py
Only top-level *.csv files of the two reviewer output folders are copied.
RV_cache and temporary sub-folders (listing-level) are never copied.
"""
import re, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(".").resolve()
SRC = {"reviewer1": ROOT / "10_revision_out" / "RV_reviewer1",
       "reviewer2": ROOT / "10_revision_out" / "RV_reviewer2"}
DST = ROOT / "10_revision_out" / "shareable_data" / "revision_R1"
BAD = re.compile(r"(^id$|listing_id|host_id|host_name|reviewer_id|latitude|longitude|^lat$|^lon$|url|comments)", re.I)

if DST.exists():
    shutil.rmtree(DST)
copied, blocked = [], []
for name, src in SRC.items():
    if not src.exists():
        print(f"MISSING folder: {src}"); continue
    out = DST / name; out.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.csv")):          # top level only, no sub-folders
        df = pd.read_csv(f)
        bad = [c for c in df.columns if BAD.search(str(c))]
        if bad or len(df) > 5000:
            blocked.append(f"{f.name}: columns {bad}, rows {len(df)}"); continue
        shutil.copy2(f, out / f.name); copied.append(f"{name}/{f.name} ({len(df)} rows)")

print("COPIED:"); [print("  ", c) for c in copied]
print("BLOCKED (not copied, check manually):"); [print("  ", b) for b in blocked] if blocked else print("   none")
print("Destination:", DST)
