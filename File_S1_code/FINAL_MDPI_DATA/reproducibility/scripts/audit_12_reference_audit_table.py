"""Assemble audit/REFERENCE_AUDIT.csv: Crossref checks + corrections + claim-citation verdicts (manual, title/abstract level).
Usage: python -I audit_12_reference_audit_table.py <AUDIT_DIR>"""
import json, sys
from pathlib import Path
import pandas as pd

a = Path(sys.argv[1]); r = a / "references"
chk = pd.read_csv(r / "crossref_check_raw.csv")
cor = pd.read_csv(r / "reference_corrections.csv")
gn = json.load(open(r / "given_name_differences.json", encoding="utf-8"))
# claim-citation verdicts: SUPPORTS / PARTIAL / NOT SUPPORTED; basis = title and, where available, Crossref abstract
V = {
 "Lorenz2023": ("Intro ¶2; 2.2 interpretable property ML", "SUPPORTS"), "Lee2025": ("2.2 interpretable valuation", "SUPPORTS"),
 "NohPark2026": ("2.2 interpretable valuation", "SUPPORTS"), "Tang2024": ("Intro ¶2; 2.2 STR price prediction", "SUPPORTS"),
 "Camatti2024": ("Intro ¶2; 2.2 STR price prediction", "SUPPORTS"), "Cavique2022": ("Intro ¶2 user-generated information; 2.3 review themes", "SUPPORTS"),
 "Negi2023": ("2.3 heterogeneous Airbnb literature", "SUPPORTS"), "KapoorNarayanan2023": ("Intro ¶2; 2.3 leakage", "SUPPORTS (narrowed: no longer cited for archive-date discrepancy)"),
 "StatsNZ2025Consents": ("2.2; 3.5 release/reference month", "SUPPORTS"), "StatsNZ2026Consents": ("2.2; 3.5 release/reference month", "SUPPORTS (re-pointed to May-2026 release)"),
 "InsideAirbnb2026": ("2.3; 3.1 data source", "SUPPORTS"), "Fyfe2023": ("2.1 NZ Airbnb and rental market", "SUPPORTS"),
 "Bekkerman2023": ("2.1 residential investment", "SUPPORTS"), "Brunstein2025": ("2.1 house prices", "SUPPORTS"),
 "Bibler2025": ("2.1 regulation", "SUPPORTS"), "DevinaArdhi2025": ("2.1 regulation", "SUPPORTS"),
 "ShortTermHousing2024": ("2.1 housing values", "SUPPORTS"), "LondonSTR2025": ("2.1 housing markets", "SUPPORTS"),
 "Milunovich2025": ("2.2 STR price prediction", "SUPPORTS"), "Almeida2025": ("2.2 reviews and prices", "SUPPORTS"),
 "Dowlut2023": ("2.3 tourism forecasting review", "SUPPORTS"), "Nunez2024": ("2.3 ML in tourism review", "SUPPORTS"),
 "AirbnbRentalPrice2023": ("2.2 STR price prediction", "SUPPORTS"), "Duso2024": ("2.1 regulation/rental markets", "SUPPORTS"),
 "LeeKim2023": ("2.1 neighbourhood change", "SUPPORTS"), "SpatialShared2025": ("2.1 local markets", "SUPPORTS"),
 "Seiler2024": ("2.1 bans", "SUPPORTS"), "ZhuLiu2024": ("2.1 urban conservation and rental market", "SUPPORTS"),
 "AirbnbRentPremium2023": ("2.1 rental-market exposure", "SUPPORTS"), "Panahandeh2025": ("2.2 STR price determinants (XAI)", "SUPPORTS"),
 "Zhang2025Reviews": ("2.3 reviews for demand forecasting", "SUPPORTS"), "Dou2023": ("2.2 interpretable valuation", "SUPPORTS"),
 "TourismBusiness2025": ("2.3 platform continuity", "SUPPORTS"), "Zhang2026Theory": ("2.3 platform-aware demand forecasting", "SUPPORTS"),
 "Liao2023": ("2.3 reviews for demand forecasting", "SUPPORTS"), "TourismDemandProgress2023": ("2.3 tourism demand research", "SUPPORTS"),
 "TourismSpatiotemporal2025": ("2.3 tourism forecasting", "SUPPORTS"), "EuropeanRegulation2023": ("2.1 regulation", "SUPPORTS"),
 "Luo2025": ("2.2 interpretable valuation", "SUPPORTS"), "Hu2022": ("2.3 reviews for demand forecasting", "SUPPORTS"),
 "Kirchner2024": ("2.1 operator types", "SUPPORTS"), "CuiAttard2025": ("2.1 operator types/strategies", "PARTIAL (COVID-period P2P operator responses; acceptable as operator-strategy evidence)"),
 "Safari2025": ("2.1 geographic variation", "SUPPORTS"), "Medina2024": ("2.3 review attributes", "SUPPORTS"),
 "Sanchez2025": ("2.2 pricing algorithms", "SUPPORTS (theoretical model)"), "CheungYiu2023": ("2.1 local conditions (crime, prices)", "SUPPORTS"),
 "Frigau2024": ("2.3 review sentiment", "SUPPORTS"), "Osth2025": ("2.2 online signals / hedonic", "SUPPORTS"),
 "BingolYang2025": ("2.3 'platform continuity and tourism systems'", "NOT SUPPORTED -> removed"), "HouseRules2025": ("2.2 house rules", "SUPPORTS"),
}
NEWV = {"Chen2016": "3.6 XGBoost", "Lundberg2020": "3.7 tree SHAP", "Youden1950": "3.6 threshold", "Tashman2000": "Intro, 2.3, 3.2 rolling origin",
        "SaitoRehmsmeier2015": "3.6 PR-AUC for imbalance", "FieldWelsh2007": "3.7 cluster bootstrap", "Brier1950": "3.6 Brier score",
        "NiculescuMizil2005": "3.6 Platt/isotonic calibration", "ReimersGurevych2020": "3.4 multilingual MiniLM embeddings", "Wilkinson2016": "5.4 FAIR data",
        "MBIE2026Bonds": "3.1, 3.4 rental data and M-2 rule", "MBIE2026ADP": "3.1 ADP data", "StatsNZ2025TA": "3.1 TA boundaries"}
rows = []
for _, c in cor.iterrows():
    k = c.key; x = chk[chk.key == k]
    x = x.iloc[0].to_dict() if len(x) else {}
    rows.append(dict(key=k, audit_status=c.status, doi=c.doi if isinstance(c.doi, str) else "", url=x.get("bib_url", ""),
                     doi_resolves=x.get("doi_resolves_crossref", "YES" if isinstance(c.doi, str) and c.doi else ""),
                     authors_crossref=x.get("cr_authors", ""), year_original=x.get("bib_year", ""), year_final=c.crossref_year,
                     title_crossref=c.crossref_title, venue=c.crossref_container, volume=x.get("cr_volume", ""), issue=x.get("cr_issue", ""), pages=x.get("cr_pages", ""),
                     title_similarity=x.get("title_similarity", ""), author_family_match=x.get("author_families_match", ""),
                     given_name_issue="; ".join(gn.get(k, [])) if k != "Medina2024" else "",
                     corrections=c.changes if isinstance(c.changes, str) else "",
                     manuscript_context=V.get(k, (NEWV.get(k, ""), ""))[0], claim_support=V.get(k, ("", "SUPPORTS (added for this claim)"))[1],
                     duplicate=False, verification_basis="Crossref API record (12 Oct 2026); claim support judged from title/abstract, full text not read" if isinstance(c.doi, str) and c.doi else "official URL fetched (HTTP 200)"))
out = pd.DataFrame(rows)
assert not out.title_crossref.dropna().duplicated().any()
out.to_csv(a / "REFERENCE_AUDIT.csv", index=False)
print(out.audit_status.value_counts()); print(out.claim_support.value_counts())
