"""Fixture tests for the Port-of-Switzerland bulletin parser (no network).

The text fixtures are taken from the real PDFs (pypdf extraction, control
bytes stripped) so they reproduce the layouts that the parser has to handle:

    2018 bulletin  legacy groups 0..99, apostrophe separators, totals glued
                   to the "Zufuhr"/"Abfuhr" marker line
    2022 bulletin  legacy groups, comma separators, per-direction "Summen"
    2025 bulletin  NST-2007 groups 01..19 + detail groups 07.21..07.25
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import collect_rhein_cargo as cargo  # noqa: E402

# ---------------------------------------------------------------------------
# fixtures (real bulletin text)
# ---------------------------------------------------------------------------
LEGACY_COMPARISON_PAGE = """Schweizerische Rheinhäfen
Monatsbulletin
Oktober 2018
Schiffsumschlag nach Warengattung in Tonnen
Umschlag im Oktober Umschlag seit Jahresbeginn
20182017 Δ% 2017 2018 Δ%
-42.2 219'948 380'586Zufuhr  3'926'672  3'325'623 -15.3
0 Landwirtschaftliche Erzeugnisse  10'338  57.4 16'270  194'275  185'911 -4.3
3 Erdöl, Mineralölerzeugnisse  162'568 -47.1 86'032  2'043'604  1'523'623 -25.4
99 Übrige Güter  14'754 -48.1 7'660  113'676  125'616  10.5
-58.9 41'206 100'337Abfuhr  848'146  845'502 -0.3
2 Feste, mineralische Brennstoffe  0  0.0 0  0  0  0.0
3 Erdöl, Mineralölerzeugnisse  16'077 -100.0 0  129'253  63'602 -50.8
Total  480'923  261'154 -45.7  4'774'818  4'171'125 -12.6
Umschlag im Oktober Umschlag seit Jahresbeginn
20182017 Δ% 2017 2018 Δ%
Auhafen  127'030  83'896 -34.0  1'474'540  1'336'177 -9.4
Birsfelden  162'184  79'765 -50.8  1'608'253  1'303'773 -18.9
Total  480'923  261'154 -45.7  4'774'818  4'171'125 -12.6
13.11.2018 Seite 3 von 6
"""

LEGACY_DETAIL_PAGE = """Schweizerische Rheinhäfen
Monatsbulletin
Oktober 2018
Umgeschlagene Waren in Mengen grösser 1000 Tonnen
Zu- und Abfuhr, 01.10.2018 bis 31.10.2018
Zufuhr TotalAbfuhrPos. Warengattung.
011 Weizen, Mengkorn     0  3'511 3'511
321 Motorbenzin u. ä. Leichtöle     0  22'659 22'659
323 Petroleum, Turbinenkraftstoff und Spezialbenzin    0  3'050 3'050
325 Gas-,Diesel-+leichtes Heizöl     0  60'323 60'323
13.11.2018 Seite 6 von 6
"""

NST2007_COMPARISON_PAGE = """SchiffsumschlagnachWarengattunginTonnen
SchweizerischeRheinhäfen
Monatsbulletin
November2025
Kennzahlen MonatVorjahr MonataktuellesJahr ďiff 2024 2025 ďiffJahr
Richtung NST-2007 ezeichnung
Zufuhr 01 LandwirtschafteErzeugnisse 15'406 22'913 48,7 220'978 374'940 69,7
07 Mineralölerzeugnisse 154'066 220'978 43,4 2'402'339 1'825'997 -24,0
16 Geräte/MaterialfürGüterbeförderung(u.a.Verpackungen,ontainer) 8215 7951 -3,2 88697 100398 13,2
Summen 288'943 374'940 29,8 4'128'963 3'468'707 -16,0
bfuhr 01 LandwirtschafteErzeugnisse 6'113 8'113 32,7 71'113 91'113 28,1
07 Mineralölerzeugnisse 11'024 11'024 0,0 96'480 83'480 -13,5
Summen 62'503 81'931 31,1 855'039 917'809 7,3
19.12.2025 Seite4
"""

NST2007_DETAIL_PAGE = """November2025
SchweizerischeRheinhäfen
Monatsbulletin
UmgeschlageneWareninTonnen
Kennzahlen Nettogewicht
Richtung Zufuhr bfuhr Summen
NST-2007 ezeichnung
01.1 Getreide 17939 – 17939
07.21 MotorbenzinundandereLeichtöle 66115 3400 69515
07.22 Petroleum,Turbinenkraftstoff 7967 – 7967
07.23 Gas-,ieselölundleichtesHeizöl 146896 6830 153726
07.25 SchmieröleundFette – 794 794
Summen 374940 81931 456871
19.12.2025 Seite11
"""

CONTAINER_PAGE = """SchweizerischeRheinhäfen
Monatsbulletin
November2025
Kennzahlen 20 40 TEU
Umschlagsdatum
01.2025 1,949 2,698 7,345
02.2025 2,154 3,736 9,626
Summen 12,345 13,456 45,678
"""

HAFENTEIL_PAGE = """SchweizerischeRheinhäfen
Monatsbulletin
November2025
Kennzahlen MonatVorjahr MonataktuellesJahr ďiff 2024 2025 ďiffJahr
ez.Hafenteil
uhafen 104,300 142,535 36.7 1,577,259 1,162,838 -26.3
irsfelden 107,993 168,221 55.8 1,743,313 1,407,627 -19.3
Summen 409,319 456,871 11.6 3,117,441 3,468,707 11.3
19.12.2025 Seite6
"""


def _records_for(pages, published_date="2025-12-19", url="https://x/Bulletin_11_2025.pdf"):
    """Run parse_bulletin on fixture pages without a PDF."""
    original = cargo._page_texts
    cargo._page_texts = lambda pdf_bytes: pages
    try:
        return cargo.parse_bulletin(b"ignored", published_date, url)
    finally:
        cargo._page_texts = original


def _find(records, group, direction):
    for record in records:
        if record["group"] == group and record["direction"] == direction:
            return record
    return None

# ---------------------------------------------------------------------------
# number handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "blob, tonnages, percentages",
    [
        # 2018 bulletin: apostrophe separators, delta glued between columns
        ("162'568 -47.1 86'032  2'043'604  1'523'623 -25.4",
         [162568, 86032, 2043604, 1523623], [-47.1, -25.4]),
        # 2025 bulletin: comma deltas
        ("154'066 220'978 43,4 2'402'339 1'825'997 -24,0",
         [154066, 220978, 2402339, 1825997], [43.4, -24.0]),
        # 2026 bulletin: comma separators, dot deltas
        ("183,946 38,791 -78.9 1,303,717 1,038,131 -20.4",
         [183946, 38791, 1303717, 1038131], [-78.9, -20.4]),
        # detail page: no thousands separators at all
        ("66115 3400 69515", [66115, 3400, 69515], []),
    ],
)
def test_split_numbers_layouts(blob, tonnages, percentages):
    found_tonnages, found_percentages = cargo._split_numbers(blob)
    assert found_tonnages == pytest.approx(tonnages)
    assert found_percentages == pytest.approx(percentages)


def test_empty_cell_dash_is_zero():
    # "–" is the bulletins' "no turnover" cell; "-" in front of a digit is a sign
    tonnages, _ = cargo._split_numbers("146896 – 153726")
    assert tonnages == pytest.approx([146896, 0, 153726])
    tonnages, percentages = cargo._split_numbers("1'574 -47.1 86'032")
    assert tonnages == pytest.approx([1574, 86032])
    assert percentages == pytest.approx([-47.1])


def test_german_dot_thousands_separator():
    assert cargo._to_number("181.622") == 181622
    assert cargo._to_number("1.825.997") == 1825997
    # a 1-2 digit group after the dot is a decimal part, not a thousands group
    assert cargo._to_number("-25.4") == -25.4
    assert cargo._to_number("1'825'997") == 1825997
    assert cargo._to_number("1,825,997") == 1825997


# ---------------------------------------------------------------------------
# reporting period + publication date
# ---------------------------------------------------------------------------
def test_bulletin_period_from_header_and_url():
    pages = [LEGACY_COMPARISON_PAGE]
    assert cargo._bulletin_period(pages, "https://x/Bulletin-Oktober-2018.pdf") == (2018, 10)
    assert cargo._bulletin_period([NST2007_COMPARISON_PAGE], "https://x/Bulletin_11_2025.pdf") == (2025, 11)
    assert cargo._bulletin_period(["no header"], "https://x/Bulletin-Oktober-2018.pdf") == (2018, 10)
    assert cargo._bulletin_period(["no header"], "https://x/SomewhereElse.pdf") == (None, None)


def test_publication_date_is_the_footer_date():
    # the WordPress media date of the whole back-archive is 2018-07-30 (upload
    # date), the bulletin itself is dated 13.11.2018
    assert cargo._publication_date([LEGACY_COMPARISON_PAGE], "2018-07-30") == "2018-11-13"
    assert cargo._publication_date([NST2007_COMPARISON_PAGE], "2026-01-12") == "2025-12-19"
    assert cargo._publication_date(["no date"], "2018-07-30") == "2018-07-30"


# ---------------------------------------------------------------------------
# page routing
# ---------------------------------------------------------------------------
def test_nst2007_bulletin_records():
    records = _records_for([NST2007_COMPARISON_PAGE, NST2007_DETAIL_PAGE])
    assert {r["classification"] for r in records} == {"NST-2007"}

    oil_in = _find(records, "07", "zufuhr")
    assert oil_in["tonnes"] == pytest.approx(220978)
    assert oil_in["prev_year_month_tonnes"] == pytest.approx(154066)
    assert oil_in["prev_year_month"] == "2024-11"
    assert oil_in["current_ytd_tonnes"] == pytest.approx(1825997)

    assert _find(records, "07", "abfuhr")["tonnes"] == pytest.approx(11024)

    total_in = _find(records, cargo.TOTAL_GROUP, "zufuhr")
    assert total_in["tonnes"] == pytest.approx(374940)
    assert total_in["current_ytd_tonnes"] == pytest.approx(3468707)

    assert _find(records, "07.23", "zufuhr")["tonnes"] == pytest.approx(146896)
    assert _find(records, "07.21", "zufuhr")["tonnes"] == pytest.approx(66115)
    assert _find(records, "07.22", "abfuhr")["tonnes"] == pytest.approx(0)
    assert _find(records, "07.25", "abfuhr")["tonnes"] == pytest.approx(794)

    # the detail groups are the >1000 t items of the main group
    detail = sum(r["tonnes"] for r in records if r["group"].startswith("07.") and r["direction"] == "zufuhr")
    assert detail <= oil_in["tonnes"]
    assert oil_in["tonnes"] - detail < 1000


def test_legacy_bulletin_records_and_direction_reconciliation():
    records = _records_for(
        [LEGACY_COMPARISON_PAGE, LEGACY_DETAIL_PAGE],
        published_date="2018-07-30",
        url="https://x/Bulletin-Oktober-2018.pdf",
    )
    assert {r["classification"] for r in records} == {"legacy"}

    oil_in = _find(records, "3", "zufuhr")
    assert oil_in["tonnes"] == pytest.approx(86032)
    assert oil_in["prev_year_month_tonnes"] == pytest.approx(162568)
    assert oil_in["current_ytd_tonnes"] == pytest.approx(1523623)

    total_in = _find(records, cargo.TOTAL_GROUP, "zufuhr")
    assert total_in["tonnes"] == pytest.approx(219948)
    assert total_in["prev_year_month_tonnes"] == pytest.approx(380586)

    # The legacy detail page lists the direction columns in the opposite order
    # of the comparison page; 321+323+325 = 86'032 = group-3 Zufuhr, so the
    # detail rows are re-labelled from the comparison page.
    assert _find(records, "07.23", "zufuhr")["tonnes"] == pytest.approx(60323)
    assert _find(records, "07.23", "abfuhr")["tonnes"] == pytest.approx(0)
    detail = sum(r["tonnes"] for r in records if r["group"].startswith("07.") and r["direction"] == "zufuhr")
    assert detail == pytest.approx(oil_in["tonnes"])

    # publication date taken from the PDF footer, not from the media API
    assert {r["published_date"] for r in records} == {"2018-11-13"}


def test_container_and_hafenteil_pages_are_not_parsed():
    records = _records_for([CONTAINER_PAGE, HAFENTEIL_PAGE, NST2007_COMPARISON_PAGE])
    assert all("TEU" not in r["group_label"] for r in records)
    # these pages alone yield nothing
    assert _records_for([CONTAINER_PAGE]) == []
    assert _records_for([HAFENTEIL_PAGE]) == []
    # with a real goods-group page present, its own totals are kept
    assert _find(records, cargo.TOTAL_GROUP, "zufuhr")["tonnes"] == pytest.approx(374940)


def test_unparsable_bulletin_yields_no_records():
    assert _records_for(["Impressum\nHafenstrasse 1\n"]) == []

