"""Phase 5 - build the corrected references.bib from Crossref metadata (DOI entries) plus verified official URLs,
and write REFERENCE_AUDIT.csv describing every correction.

Usage: python -I audit_10_build_corrected_bib.py <AUDIT_REF_DIR> <OUT_BIB> <OUT_AUDIT_CSV>
Inputs in AUDIT_REF_DIR: references_ORIGINAL_input.bib, crossref_check_raw.csv, crossref_records_bib.json,
                         crossref_records_candidate_additions.json
"""
import json, re, sys, unicodedata, urllib.request, urllib.parse, time
from pathlib import Path
import pandas as pd

d = Path(sys.argv[1]); out_bib = Path(sys.argv[2]); out_csv = Path(sys.argv[3])
chk = pd.read_csv(d / "crossref_check_raw.csv").set_index("key")
cr = json.load(open(d / "crossref_records_bib.json", encoding="utf-8"))["bib_crossref"]
UA = {"User-Agent": "MDPI-Data-reference-audit/1.0"}


def tex(s):
    s = str(s).replace("‐", "-").replace("‑", "-").replace("–", "--").replace("—", "---").replace("’", "'").replace("&amp;", "\\&")
    s = s.replace("&", "\\&") if "\\&" not in s else s
    rep = {"á": "{\\'a}", "é": "{\\'e}", "í": "{\\'i}", "ó": "{\\'o}", "ú": "{\\'u}", "ñ": "{\\~n}", "ü": '{\\"u}', "ö": '{\\"o}', "ä": '{\\"a}',
           "è": "{\\`e}", "à": "{\\`a}", "ç": "{\\c{c}}", "ã": "{\\~a}", "õ": "{\\~o}", "â": "{\\^a}", "ê": "{\\^e}", "ô": "{\\^o}", "É": "{\\'E}", "Á": "{\\'A}", "Ó": "{\\'O}", "ø": "{\\o}", "å": "{\\aa}"}
    return "".join(rep.get(ch, ch) for ch in s)


def year_of(m):
    for k in ["published-print", "journal-issue", "published-online", "issued"]:
        v = m.get(k, {})
        if isinstance(v, dict) and v.get("date-parts"):
            return v["date-parts"][0][0]
        if k == "journal-issue" and isinstance(v, dict) and v.get("published-print", {}).get("date-parts"):
            return v["published-print"]["date-parts"][0][0]


def fetch(doi):
    return json.loads(urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(doi), headers=UA), timeout=40).read())["message"]


# Title overrides: keep subtitle verified on publisher page / English title only
TITLE_OVERRIDE = {
    "EuropeanRegulation2023": "Challenges and effects of short-term rentals regulation: A counterfactual assessment of European cities",
    "Kirchner2024": "Selling out the city? {Airbnb}'s amateur and professional listings across cities, time, and hotspots",
}
PROTECT = ["Airbnb", "Sydney", "Dublin", "Berlin", "London", "Shanghai", "Corsica", "Virginia", "San Francisco", "New Zealand", "Xi'an", "China",
           "European", "United States", "P2P", "AI", "COVID-19", "XGBoost", "BERT", "Sentence-BERT", "Siamese", "ROC", "FAIR"]


def protect(t):
    for w in PROTECT:
        t = re.sub(r"(?<![{\w])" + re.escape(w) + r"(?![\w}])", "{" + w + "}", t)
    return t


entries, audit = [], []
orig = open(d / "references_ORIGINAL_input.bib", encoding="utf-8").read()
keys = [m.group(1).strip() for m in re.finditer(r"@\w+\s*\{\s*([^,]+),", orig)]
for k in keys:
    r = chk.loc[k]
    if isinstance(r.get("bib_doi"), str) and r.bib_doi:
        m = cr[k]
        auth = " and ".join(tex(f"{a.get('family', a.get('name', ''))}, {a.get('given', '')}".strip(", ")) for a in m.get("author", []))
        title = TITLE_OVERRIDE.get(k, (m.get("title") or [r.bib_title])[0])
        journal = (m.get("container-title") or [r.bib_journal])[0]
        yr = year_of(m); vol = m.get("volume", ""); iss = m.get("issue", ""); pg = m.get("page", "") or m.get("article-number", "")
        typ = "inproceedings" if m.get("type") == "proceedings-article" else "article"
        f = [f"  author = {{{auth}}}", f"  title = {{{protect(tex(title))}}}", (f"  booktitle = {{{tex(journal)}}}" if typ == "inproceedings" else f"  journal = {{{tex(journal)}}}"), f"  year = {{{yr}}}"]
        if vol: f.append(f"  volume = {{{vol}}}")
        if iss: f.append(f"  number = {{{iss}}}")
        if pg: f.append(f"  pages = {{{tex(str(pg).lstrip('0') if re.match(r'^0\\d', str(pg)) else pg).replace('-', '--') if '-' in str(pg) else tex(pg)}}}")
        f.append(f"  doi = {{{r.bib_doi}}}")
        entries.append(f"@{typ}{{{k},\n" + ",\n".join(f) + "\n}")
        changes = []
        if not bool(r.author_families_match): changes.append(f"authors corrected to Crossref ({r.cr_authors})")
        if not bool(r.year_match): changes.append(f"year {r.bib_year} -> {yr} (Crossref print/issue year)")
        if isinstance(r.metadata_incomplete_in_bib, str) and r.metadata_incomplete_in_bib: changes.append(f"added {r.metadata_incomplete_in_bib}")
        if float(r.title_similarity) < 0.98: changes.append("title kept with subtitle (Crossref title lacks subtitle)")
        if k == "AirbnbRentalPrice2023": changes.append("entry type article -> inproceedings; booktitle from Crossref")
        audit.append(dict(key=k, status="VERIFIED (DOI resolves; metadata match)" if not changes else "CORRECTED", doi=r.bib_doi, crossref_title=r.cr_title,
                          crossref_year=yr, crossref_container=journal, changes="; ".join(changes)))
    else:
        audit.append(dict(key=k, status="VERIFIED (official URL, HTTP 200)" if str(r.get("url_status")) in ("200", "200.0") else "CHECK", doi="", crossref_title="", crossref_year="", crossref_container="", changes=""))

misc = {
    "StatsNZ2025Consents": "@misc{StatsNZ2025Consents,\n  author = {{Stats NZ}},\n  title = {Building Consents Issued: {August} 2025},\n  year = {2025},\n  howpublished = {Information release, 1 October 2025, 10:45 NZT},\n  url = {https://www.stats.govt.nz/information-releases/building-consents-issued-august-2025/},\n  note = {Accessed 12 October 2026}\n}",
    "StatsNZ2026Consents": "@misc{StatsNZ2026Consents,\n  author = {{Stats NZ}},\n  title = {Building Consents Issued: {May} 2026},\n  year = {2026},\n  howpublished = {Information release, 2 July 2026, 10:45 NZT},\n  url = {https://www.stats.govt.nz/information-releases/building-consents-issued-may-2026/},\n  note = {Accessed 12 October 2026}\n}",
    "InsideAirbnb2026": "@misc{InsideAirbnb2026,\n  author = {{Inside Airbnb}},\n  title = {Get the Data: {New Zealand} Listings, Calendars and Reviews},\n  year = {2026},\n  url = {https://insideairbnb.com/get-the-data/},\n  note = {Licensed under CC BY 4.0; snapshots dated 5 October 2025 to 13 August 2026; accessed 12 October 2026}\n}",
}
NEW = {
    "MBIE2026Bonds": "@misc{MBIE2026Bonds,\n  author = {{Ministry of Business, Innovation and Employment}},\n  title = {Rental Bond Data: Detailed Monthly Data by Territorial Authority, {February} 1993 to {July} 2026},\n  year = {2026},\n  howpublished = {Tenancy Services; file last updated 10 September 2026; CC BY 3.0 NZ},\n  url = {https://www.tenancy.govt.nz/about-tenancy-services/data-and-statistics/rental-bond-data/},\n  note = {Accessed 12 October 2026}\n}",
    "MBIE2026ADP": "@misc{MBIE2026ADP,\n  author = {{Ministry of Business, Innovation and Employment}},\n  title = {Accommodation Data Programme: All Measures},\n  year = {2026},\n  howpublished = {Tourism data release; CSV extract downloaded 4 October 2026},\n  url = {https://www.mbie.govt.nz/immigration-and-tourism/tourism-research-and-data/tourism-data-releases/accommodation-data-programme},\n  note = {Accessed 12 October 2026}\n}",
    "StatsNZ2025TA": "@misc{StatsNZ2025TA,\n  author = {{Stats NZ}},\n  title = {Territorial Authority 2025 (Generalised) Boundaries},\n  year = {2025},\n  howpublished = {Stats NZ Geographic Data Service},\n  url = {https://datafinder.stats.govt.nz/},\n  note = {Accessed 28 September 2026}\n}",
}
ADD_DOI = {"Chen2016": "10.1145/2939672.2939785", "Lundberg2020": "10.1038/s42256-019-0138-9",
           "Youden1950": "10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3", "Tashman2000": "10.1016/S0169-2070(00)00065-0",
           "SaitoRehmsmeier2015": "10.1371/journal.pone.0118432", "FieldWelsh2007": "10.1111/j.1467-9868.2007.00593.x",
           "Brier1950": "10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2", "NiculescuMizil2005": "10.1145/1102351.1102430",
           "ReimersGurevych2020": "10.18653/v1/2020.emnlp-main.365", "Wilkinson2016": "10.1038/sdata.2016.18"}
for k, doi in ADD_DOI.items():
    m = fetch(doi); time.sleep(0.4)
    au = m.get("author", [])
    auth = " and ".join(tex(f"{a.get('family', a.get('name', ''))}, {a.get('given', '')}".strip(", ")) for a in (au[:10] if len(au) <= 10 else au[:10])) + (" and others" if len(au) > 10 else "")
    typ = "inproceedings" if m.get("type") == "proceedings-article" else "article"
    title = (m.get("title") or [""])[0]
    if k == "Brier1950": title = "Verification of forecasts expressed in terms of probability"
    if k == "Chen2016": title = "{XGBoost}: A Scalable Tree Boosting System"
    jt = (m.get("container-title") or [""])[0]
    # mdpi.bst prefixes "In Proceedings of the"; avoid duplicated wording and normalise names
    jt = {"ReimersGurevych2020": "2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
          "NiculescuMizil2005": "22nd International Conference on Machine Learning (ICML 2005)",
          "Chen2016": "22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining"}.get(k, jt)
    if k == "Brier1950":
        auth = "Brier, Glenn W."
    f = [f"  author = {{{auth}}}", f"  title = {{{title if k == 'Chen2016' else protect(tex(title))}}}", (f"  booktitle = {{{tex(jt)}}}" if typ == "inproceedings" else f"  journal = {{{tex(jt)}}}"), f"  year = {{{year_of(m)}}}"]
    if m.get("volume"): f.append(f"  volume = {{{m['volume']}}}")
    if m.get("issue"): f.append(f"  number = {{{m['issue']}}}")
    pg = m.get("page", "") or m.get("article-number", "")
    if pg: f.append(f"  pages = {{{str(pg).replace('-', '--')}}}")
    f.append(f"  doi = {{{doi}}}")
    NEW[k] = f"@{typ}{{{k},\n" + ",\n".join(f) + "\n}"
    audit.append(dict(key=k, status="ADDED (methodological source; DOI verified via Crossref)", doi=doi, crossref_title=(m.get("title") or [""])[0],
                      crossref_year=year_of(m), crossref_container=jt, changes="new entry"))
for k in ["MBIE2026Bonds", "MBIE2026ADP", "StatsNZ2025TA"]:
    audit.append(dict(key=k, status="ADDED (data source; official URL)", doi="", crossref_title="", crossref_year="", crossref_container="", changes="new entry"))
for a in audit:
    if a["key"] == "StatsNZ2026Consents":
        a["changes"] = "replaced 'August 2026' release (published after T10) by the 'May 2026' release (2 Jul 2026) that supplied the T10 consent month"
        a["status"] = "CORRECTED"
    if a["key"] == "InsideAirbnb2026":
        a["changes"] = "licence and snapshot range added"

REMOVED = {"BingolYang2025": "removed: smart-destination ecosystem paper does not support the 'platform continuity' claim it was cited for"}
for a in audit:
    if a["key"] in REMOVED:
        a["status"] = "REMOVED"; a["changes"] = REMOVED[a["key"]]
body = "\n\n".join([misc.get(k) if k in misc else next(e for e in entries if e.startswith("@") and "{" + k + "," in e) for k in keys if k not in REMOVED] + list(NEW.values()))
out_bib.write_text("% Corrected bibliography generated by audit_10_build_corrected_bib.py from Crossref metadata (12 Oct 2026).\n\n" + body + "\n", encoding="utf-8")
pd.DataFrame(audit).to_csv(out_csv, index=False)
print(pd.DataFrame(audit).status.value_counts())
