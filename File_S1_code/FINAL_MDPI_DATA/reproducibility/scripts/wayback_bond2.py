import re, sys, time, urllib.request
UA = {"User-Agent": "Mozilla/5.0 research-audit"}
p = "www.tenancy.govt.nz/about-tenancy-services/data-and-statistics/rental-bond-data/"
for ts in sys.argv[1:]:
    for attempt in range(3):
        try:
            html = urllib.request.urlopen(urllib.request.Request(f"https://web.archive.org/web/{ts}id_/https://{p}", headers=UA), timeout=90).read().decode("utf-8", "ignore")
            break
        except Exception as e:
            html = ""; err = e; time.sleep(5)
    txt = re.sub(r"<script[\s\S]*?</script>", " ", html); txt = re.sub(r"<[^>]+>", " ", txt); txt = re.sub(r"\s+", " ", txt)
    i = txt.find("Rental bond data,")
    if i < 0: i = txt.find("February 1993")
    print(ts, "|", txt[max(0, i - 50): i + 400] if i >= 0 else ("NO MATCH len=%d" % len(txt)))
    time.sleep(2)
