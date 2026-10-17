"""Reconstruct MBIE rental-bond file update dates from Internet Archive captures of the Tenancy Services page."""
import re, json, sys, time, urllib.request
UA = {"User-Agent": "Mozilla/5.0 research-audit"}
pages = ["www.tenancy.govt.nz/about-tenancy-services/data-and-statistics/rental-bond-data/"]
out = []
for p in pages:
    cdx = f"https://web.archive.org/cdx/search/cdx?url={p}&from=20250701&to=20261012&output=json&filter=statuscode:200&collapse=digest"
    rows = json.loads(urllib.request.urlopen(urllib.request.Request(cdx, headers=UA), timeout=60).read())
    print("captures:", len(rows) - 1, file=sys.stderr)
    for r in rows[1:]:
        ts = r[1]
        url = f"https://web.archive.org/web/{ts}id_/https://{p}"
        try:
            html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "ignore")
            txt = re.sub(r"<[^>]+>", " ", html); txt = re.sub(r"\s+", " ", txt)
            lu = re.search(r"Last updated\s*([0-9]{1,2} [A-Za-z]+ 20[0-9]{2})", txt)
            cov = re.search(r"Rental bond data, ([A-Za-z]+ [0-9]{4}) to ([A-Za-z]+ [0-9]{4})", txt)
            rec = {"capture_ts": ts, "last_updated": lu.group(1) if lu else None,
                   "coverage_end": cov.group(2) if cov else None}
        except Exception as e:
            rec = {"capture_ts": ts, "error": str(e)}
        out.append(rec); print(json.dumps(rec))
        time.sleep(1)
json.dump(out, open(sys.argv[1], "w"), indent=1)
