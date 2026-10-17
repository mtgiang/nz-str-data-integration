"""Fetch official Stats NZ 'Building consents issued: <Month YYYY>' release pages and extract the publication date."""
import re, sys, json, urllib.request, datetime as dt
months = ["July 2025","August 2025","September 2025","October 2025","November 2025","December 2025",
          "January 2026","February 2026","March 2026","April 2026","May 2026","June 2026","July 2026"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36"}
out = []
for m in months:
    slug = "building-consents-issued-" + m.lower().replace(" ", "-")
    url = f"https://www.stats.govt.nz/information-releases/{slug}/"
    rec = {"reference_month": m, "url": url, "accessed_utc": dt.datetime.utcnow().isoformat(timespec="seconds")}
    try:
        html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read().decode("utf-8", "ignore")
        rec["http"] = 200
        cands = set()
        for pat in [r'"datePublished"\s*:\s*"([^"]+)"', r'article:published_time"\s+content="([^"]+)"',
                    r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d\d)',
                    r'"ReleaseDate"\s*:\s*"([^"]+)"', r'release-date[^>]*>([^<]+)<']:
            for x in re.findall(pat, html):
                cands.add(x.strip())
        rec["date_candidates"] = sorted(cands)[:12]
        pub = re.search(r"PublicationDate&quot;:&quot;([0-9: -]+)&quot;", html)
        rec["publication_datetime_nzt"] = pub.group(1) if pub else None
        nxt = re.search(r"will be released on ([0-9]{1,2} [A-Za-z]+ 20[0-9]{2})", html)
        rec["next_release_stated"] = nxt.group(1) if nxt else None
        t = re.search(r"<title>(.*?)</title>", html, re.S)
        rec["title"] = t.group(1).strip() if t else None
    except Exception as e:
        rec["http"] = str(e)
    out.append(rec)
    print(json.dumps(rec, ensure_ascii=False))
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
