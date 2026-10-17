#!/usr/bin/env python
"""
Verify that locally obtained Inside Airbnb snapshots are byte-identical to those used in the study.

The original Inside Airbnb files are NOT redistributed. Obtain them yourself (see DATA_ACCESS.md), place them as
    01_raw/airbnb/<YYYY-MM-DD>/{listings,calendar,reviews}.csv.gz
and run, from the project root:
    python verify_snapshots.py Table_S3_snapshot_manifest.csv

The manifest columns are detected from their CONTENT, so different column names are accepted:
  - checksum column: values of 64 hex characters (SHA-256), 40 (SHA-1) or 32 (MD5)
  - file column:     values containing listings / calendar / reviews (a path or a file name)
  - date:            a YYYY-MM-DD value in any column of the row (or in the path)
"""
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

HEX = re.compile(r"^[0-9a-fA-F]{32}$|^[0-9a-fA-F]{40}$|^[0-9a-fA-F]{64}$")
DATE = re.compile(r"(20\d\d-\d\d-\d\d)")
KIND = re.compile(r"(listings|calendar|reviews)", re.I)
KIND_NC = re.compile(r"(?:listings|calendar|reviews)", re.I)
ALGO = {32: "md5", 40: "sha1", 64: "sha256"}


def file_hash(p, algo, chunk=1 << 20):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def detect(m):
    def share(col, rx):
        v = m[col].dropna().astype(str).str.strip()
        return v.str.contains(rx).mean() if len(v) else 0.0
    hcol = max(m.columns, key=lambda c: share(c, HEX))
    if share(hcol, HEX) < 0.5:
        return None, None
    others = [c for c in m.columns if c != hcol]
    fcol = max(others, key=lambda c: share(c, KIND_NC)) if others else None
    if fcol is not None and share(fcol, KIND_NC) == 0:
        fcol = None
    return hcol, fcol


def main(manifest, root="01_raw/airbnb"):
    m = pd.read_csv(manifest, dtype=str)
    hcol, fcol = detect(m)
    if hcol is None or fcol is None:
        print("Could not find a checksum column and a file column in the manifest.")
        print("Columns found:", list(m.columns))
        print(m.head(5).to_string())
        sys.exit(1)
    print(f"checksum column: '{hcol}' | file column: '{fcol}'")
    rows = []
    for _, r in m.iterrows():
        exp = str(r[hcol]).strip().lower()
        ftxt = str(r[fcol])
        kind = KIND.search(ftxt)
        if not HEX.match(exp) or not kind:
            continue                                   # contextual files (Stats NZ, MBIE...) are skipped
        rowtxt = " ".join(str(x) for x in r.values)
        date = DATE.search(ftxt) or DATE.search(rowtxt)
        if not date:
            continue
        snap = Path(root, date.group(1))
        name = Path(ftxt.replace("\\", "/")).name
        hits = [snap / name] if (snap / name).is_file() else sorted(snap.glob(f"{kind.group(1).lower()}.csv*"))
        if not hits:
            rows.append((date.group(1), kind.group(1).lower(), "MISSING")); continue
        got = file_hash(hits[0], ALGO[len(exp)])
        rows.append((date.group(1), kind.group(1).lower(), "OK" if got == exp else "DIFFERENT"))
    res = pd.DataFrame(rows, columns=["snapshot", "file", "status"]).drop_duplicates()
    print(res.to_string(index=False))
    ok = int((res.status == "OK").sum())
    print(f"\n{ok}/{len(res)} files identical")
    if ok != len(res):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Table_S3_snapshot_manifest.csv")
