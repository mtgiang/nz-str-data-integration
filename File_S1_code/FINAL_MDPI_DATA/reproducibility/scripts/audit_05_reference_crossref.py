"""Phase 5 - verify every BibTeX entry against Crossref (DOI) or the official URL.

Usage: python -I audit_05_reference_crossref.py <references.bib> <OUT_CSV> <OUT_JSON> [extra DOIs...]
Network access to api.crossref.org is required. Writes a row per entry with the Crossref record and field-level checks.
"""
import json, re, sys, time, difflib, unicodedata, urllib.request, urllib.parse
import pandas as pd

UA = {"User-Agent": "MDPI-Data-reference-audit/1.0 (mailto:audit@example.invalid)"}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def parse_bib(text):
    out = []
    for chunk in re.split(r"\n\s*(?=@\w+\s*\{)", "\n" + text):
        m = re.match(r"\s*@(\w+)\s*\{\s*([^,]+),(.*)", chunk, re.S)
        if not m:
            continue
        typ, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body):
            fields[fm.group(1).lower()] = re.sub(r"[{}]", "", fm.group(2)).strip()
        out.append({"type": typ, "key": key, **fields})
    return out


def crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    for i in range(3):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read())["message"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return "ERROR"


def year_of(m):
    for k in ["published-print", "published-online", "issued"]:
        if m.get(k, {}).get("date-parts"):
            return m[k]["date-parts"][0][0]


def url_ok(u):
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=40)
        html = r.read().decode("utf-8", "ignore"); t = re.search(r"<title>(.*?)</title>", html, re.S)
        return r.status, (t.group(1).strip() if t else "")
    except Exception as e:
        return str(e), ""


bib = parse_bib(open(sys.argv[1], encoding="utf-8").read())
rows, raw = [], {}
for e in bib:
    r = {"key": e["key"], "bib_type": e["type"], "bib_authors": e.get("author", ""), "bib_year": e.get("year", ""), "bib_title": e.get("title", ""),
         "bib_journal": e.get("journal", e.get("booktitle", "")), "bib_volume": e.get("volume", ""), "bib_number": e.get("number", ""),
         "bib_pages": e.get("pages", ""), "bib_doi": e.get("doi", ""), "bib_url": e.get("url", "")}
    if r["bib_doi"]:
        m = crossref(r["bib_doi"]); raw[e["key"]] = m
        if m in (None, "ERROR"):
            r["doi_resolves_crossref"] = "NO" if m is None else "ERROR"
        else:
            cr_auth = [a.get("family", a.get("name", "")) for a in m.get("author", [])]
            bib_fam = [a.split(",")[0].strip() for a in r["bib_authors"].split(" and ") if a.strip()]
            r.update({"doi_resolves_crossref": "YES", "cr_title": (m.get("title") or [""])[0], "cr_journal": (m.get("container-title") or [""])[0],
                      "cr_year": year_of(m), "cr_volume": m.get("volume", ""), "cr_issue": m.get("issue", ""),
                      "cr_pages": m.get("page", m.get("article-number", "")), "cr_authors": "; ".join(cr_auth), "cr_type": m.get("type", "")})
            ta, tb = " ".join(norm(r["bib_title"])), " ".join(norm(r["cr_title"]))
            r["title_similarity"] = round(difflib.SequenceMatcher(None, ta, tb).ratio(), 3)
            r["year_match"] = str(r["cr_year"]) == r["bib_year"]
            ncr = [" ".join(norm(x)) for x in cr_auth]; nb = [" ".join(norm(x)) for x in bib_fam]
            r["author_count_bib_cr"] = f"{len(nb)}/{len(ncr)}"
            r["author_families_match"] = nb == ncr
            r["first_author_match"] = bool(nb and ncr and nb[0] == ncr[0])
            r["volume_match"] = (not r["bib_volume"]) or r["bib_volume"] == str(r["cr_volume"])
            r["pages_match"] = (not r["bib_pages"]) or r["bib_pages"].replace("--", "-") == str(r["cr_pages"])
            r["metadata_incomplete_in_bib"] = ",".join([f for f, v in [("volume", r["bib_volume"] or not r["cr_volume"]), ("pages", r["bib_pages"] or not r["cr_pages"])] if not v])
        time.sleep(0.4)
    elif r["bib_url"]:
        st, title = url_ok(r["bib_url"]); r["url_status"] = st; r["url_title"] = title[:150]
    rows.append(r)
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
extra = {}
for d in sys.argv[4:]:
    m = crossref(d)
    extra[d] = None if m in (None, "ERROR") else {"title": (m.get("title") or [""])[0], "authors": [(a.get("given", ""), a.get("family", a.get("name", ""))) for a in m.get("author", [])],
                                                    "journal": (m.get("container-title") or [""])[0], "year": year_of(m), "volume": m.get("volume", ""),
                                                    "issue": m.get("issue", ""), "pages": m.get("page", m.get("article-number", "")), "type": m.get("type", ""),
                                                    "publisher": m.get("publisher", "")}
    time.sleep(0.4)
json.dump({"bib_crossref": raw, "extra_candidates": extra}, open(sys.argv[3], "w", encoding="utf-8"), indent=1, default=str)
print(pd.DataFrame(rows)[["key", "doi_resolves_crossref", "title_similarity", "year_match", "author_families_match", "volume_match", "pages_match", "metadata_incomplete_in_bib"]].to_string())
print(json.dumps(extra, indent=1))
