"""Phase 1a - file inventory and SHA-256 checksums for the whole project (read-only).

Usage: python -I audit_01_inventory_checksums.py <PROJECT_ROOT> <OUT_DIR>
Writes: DATA_SOURCE_INVENTORY.csv (source-level, curated + hashed) and FILE_CHECKSUMS_SHA256.csv (every file).
No file under PROJECT_ROOT outside FINAL_MDPI_DATA is modified.
"""
import csv, hashlib, sys, datetime as dt
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve(); out.mkdir(parents=True, exist_ok=True)
SKIP = {".ipynb_checkpoints", "__pycache__", "FINAL_MDPI_DATA"}


def sha256(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


rows = []
for p in sorted(root.rglob("*")):
    if not p.is_file() or any(s in p.parts for s in SKIP):
        continue
    rel = p.relative_to(root).as_posix()
    st = p.stat()
    rows.append({"relative_path": rel, "top_folder": rel.split("/")[0], "bytes": st.st_size,
                 "modified_local": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                 "sha256": sha256(p)})
    print(rel, flush=True)

with open(out / "FILE_CHECKSUMS_SHA256.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("files hashed:", len(rows))
