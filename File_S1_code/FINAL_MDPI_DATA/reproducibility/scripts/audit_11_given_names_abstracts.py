"""Phase 5 - detect given-name discrepancies between the original bib and Crossref; export Crossref abstracts
for the claim-citation review. Usage: python -I audit_11_given_names_abstracts.py <AUDIT_REF_DIR>"""
import json, re, sys, unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
d = Path(sys.argv[1])
n = lambda s: unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().strip()
orig = (d / "references_ORIGINAL_input.bib").read_text(encoding="utf-8")
cr = json.load(open(d / "crossref_records_bib.json", encoding="utf-8"))["bib_crossref"]
res, abstracts = {}, {}
for chunk in re.split(r"\n\s*(?=@\w+\s*\{)", "\n" + orig):
    m = re.match(r"\s*@(\w+)\s*\{\s*([^,]+),(.*)", chunk, re.S)
    if not m:
        continue
    k = m.group(2).strip(); a = re.search(r"author\s*=\s*\{(.*?)\}\s*,", m.group(3), re.S)
    if k not in cr or not isinstance(cr[k], dict):
        continue
    if cr[k].get("abstract"):
        abstracts[k] = re.sub(r"<[^>]+>", " ", cr[k]["abstract"]).strip()[:1500]
    if not a:
        continue
    bib = [x.strip() for x in a.group(1).split(" and ")]
    diffs = []
    for b, c in zip(bib, cr[k].get("author", [])):
        bg = b.split(",", 1)[1].strip() if "," in b else " ".join(b.split()[:-1])
        cg = c.get("given", "")
        nb, nc = n(bg).lower(), n(cg).lower()
        if nb[:1] != nc[:1] or (len(nb) > 2 and len(nc) > 2 and nb.split()[0].rstrip(".") != nc.split()[0].rstrip(".")):
            diffs.append(f"{b} -> {c.get('family')}, {cg}")
    if diffs:
        res[k] = diffs
json.dump(res, open(d / "given_name_differences.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(abstracts, open(d / "crossref_abstracts.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k, v in res.items():
    print(k, v)
print("abstracts available:", len(abstracts), sorted(abstracts))
