"""Collect the monthly cargo / oil turnover of the Swiss Rhine ports (Basel).

Port of Switzerland (Schweizerische Rheinhaefen) publishes a monthly PDF
bulletin "Schiffsumschlag nach Warengattung in Tonnen" that contains the
water-side turnover of the Swiss Rhine ports per NST-2007 goods group -
including the detailed fuel groups:

    07.21 Motorbenzin und andere Leichtoel
    07.22 Petroleum, Turbinenkraftstoff
    07.23 Gas-, Dieseloel und leichtes Heizoel   <- the heating-oil volume
    07.25 Schmieroel und Fette

There is no structured (JSON) API for these numbers, but the site runs on
WordPress and its media API is open, so the media API is used as the discovery
index and the values themselves are parsed out of the PDF.

Output (one row per bulletin month x goods group x direction):
    s3://oilprice-features/raw/rhein_cargo.csv
    columns: month, published_date, bulletin_url, bulletin_year, bulletin_month,
             classification, group, group_label, direction, tonnes
    s3://oilprice-features/raw/rhein_bulletins_parsed.json
    bulletins already fetched (including the ones without a parsable table), so
    a daily run does not re-download the whole archive. Use --all to rebuild.

Publication lag is ~4-7 weeks, so these are LAGGED features only.

Run:
    python src/collect_rhein_cargo.py            # parse bulletins not stored yet
    python src/collect_rhein_cargo.py --all      # re-parse the whole archive
    python src/collect_rhein_cargo.py --limit 3  # only the 3 newest bulletins
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

import pandas as pd
import requests

# pypdf emits "Skipping broken line ..." for the bulletins' broken ToUnicode
# CMaps; the extracted text is still usable, so keep the noise out of the logs.
logging.getLogger("pypdf").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rhein_config import (  # noqa: E402
    CARGO_DETAIL_GROUPS,
    LOCAL_CARGO_CSV,
    POS_BULLETIN_SEARCH,
    POS_MEDIA_URL,
    RHEIN_BULLETINS_KEY,
    RHEIN_CARGO_KEY,
    S3_BUCKET,
)

HEADERS = {"User-Agent": "Mozilla/5.0 (oilprice rhine cargo collector)"}

CARGO_COLUMNS = [
    "month",
    "published_date",
    "bulletin_year",
    "bulletin_month",
    "bulletin_url",
    "classification",
    "group",
    "group_label",
    "direction",
    "tonnes",
]

MONTH_NAMES = {
    "januar": 1, "februar": 2, "maerz": 3, "märz": 3, "april": 4, "mai": 5, "juni": 6,
    "juli": 7, "august": 8, "september": 9, "oktober": 10, "november": 11, "dezember": 12,
}

# Bulletins published before ~2020 use the old goods-group classification
# (0..99, e.g. "3 Erdöl, Mineralölerzeugnisse"); newer ones use NST-2007
# (01..20 with detail groups 07.21..07.25).
NST2007_FROM_YEAR = 2020

# The bulletins switched to NST-2007 goods groups with the layout change in
# 2024, but the switch is detected from the PDF itself ("NST-2007" column
# header) because the archive contains NST-2007-era years (2020-2023) that are
# still printed with the legacy groups 0..99.
NST2007_MARKER = "NST-2007"

# Synthetic group code used for the per-direction table totals ("Summen" /
# "Total"). Not a real goods group, so it cannot collide with legacy group 99.
TOTAL_GROUP = "TOTAL"

# Legacy (pre-NST-2007) detail positions of the mineral-oil group 3 on the
# "Umgeschlagene Waren in Mengen groesser 1000 Tonnen" page, mapped onto the
# NST-2007 fuel codes so that one feature definition covers both eras:
#   321 Motorbenzin u. a. Leichtoel        -> 07.21
#   323 Petroleum, Turbinenkraftstoff      -> 07.22
#   325 Gas-, Diesel- + leichtes Heizoel   -> 07.23
LEGACY_DETAIL_GROUPS = {"321": "07.21", "323": "07.22", "325": "07.23", "341": "07.25"}

CONTROL_CHARS = re.compile(r"[\x00-\x09\x0b-\x1f]")
# A number token: 1'825'997 / 220'978 / 43,4 / -24,0. En-dashes ("–" = no
# turnover) are not matched at all and therefore simply ignored.
NUMBER_RE = re.compile(r"-?\d[\d'.,\u2013-]*")
BULLETIN_TITLE_RE = re.compile(r"Monatsbulletin\s*([A-Za-zÄÖÜäöüß]+)\s*(\d{4})", re.IGNORECASE)

# Trailing run of numeric cells of a table row.
NUMBER_BLOB = r"((?:[\d'.,\u2013-]|\s){4,})$"
# Goods-group tables only. Container pages (TEU) and Hafenteil pages have the
# same numeric layout but a different meaning, so a page is only parsed when it
# carries one of these titles.
COMPARISON_TITLE = "seit Jahresbeginn"
DETAIL_TITLES = ("Umgeschlagene Waren", "Warengattung in Tonnen")
# A percentage cell: at most 3 integer digits and 1-2 decimal digits
# (43,4 / -78.9 / 100.0). Tonnages always use a 3-digit group separator, so
# they can never fullmatch this pattern (1,303,717 / 122'759 / 8,324).
PERCENT_RE = re.compile(r"(-?)(\d{1,3})[,.](\d{1,2})")
# Standalone "no turnover" cell ("146'896 - 6'830" keeps its minus signs).
DASH_CELL_RE = re.compile(r"[\u2013\u2014-]+(?!\d)")
# Detailed goods groups: "07.23 Gas-, Dieseloel und leichtes Heizoel 146'896 6'830 153'726"
# and the legacy 3-digit positions "325 Gas-,Diesel-+leichtes Heizoel 0 60'323 60'323"
DETAIL_LINE_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,2})?)\s+(.+?)\s+" + NUMBER_BLOB)
# Main goods groups: "07 Mineraloelerzeugnisse 1'825'997 188'812 ... 220'978"
# (legacy bulletins use single-digit group codes such as "3 Erdöl, Mineralölerzeugnisse")
MAIN_LINE_RE = re.compile(r"^(?:Zufuhr|Abfuhr|bfuhr)?\s*(\d{1,2})\s+(.+?)\s+" + NUMBER_BLOB)


def _get(url, params=None, timeout=60, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            last_error = RuntimeError(f"{url} -> HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(2 ** attempt)
    raise last_error


def _to_number(token):
    """Parse a bulletin number: \"1'825'997\", \"1,825,997\", \"181.622\", \"-24,0\".

    Group separators (' , .) are dropped; a trailing 1-2 digit group is read as
    the decimal part. \"-\" / \"--\" / \"–\" (no turnover) -> None.
    """
    token = token.strip().replace("\u2013", "").replace(" ", "")
    if not token or token in {"-", "--"}:
        return None
    token = token.replace("'", "")
    # Some bulletins use the German dot separator (181.622 = 181622 t). A
    # decimal part of these tables never has three digits, so a trailing
    # 3-digit group is always a thousands group.
    if re.search(r"\.\d{3}(?:\.\d{3})*$", token):
        token = token.replace(".", "")
    elif "," in token:
        head, _, tail = token.rpartition(",")
        if tail.isdigit() and len(tail) <= 2:
            token = head.replace(",", "") + "." + tail
        else:
            token = token.replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Discovery: WordPress media API (JSON)
# ---------------------------------------------------------------------------
def list_bulletins():
    """Return every monthly-bulletin PDF known to the Port-of-Switzerland media API."""
    items, page = [], 1
    while True:
        resp = _get(POS_MEDIA_URL, params={"search": POS_BULLETIN_SEARCH, "per_page": 100, "page": page, "orderby": "date", "order": "asc"})
        batch = resp.json()
        if not batch:
            break
        for media in batch:
            url = media.get("guid", {}).get("rendered") or media.get("source_url", "")
            slug = media.get("slug", "")
            if not url.lower().endswith(".pdf"):
                continue
            # annual reports / statistics PDFs and multi-month summaries are not
            # monthly bulletins; they have no single reporting month to parse
            if re.search(r"jahresbulletin|quartals|halbjahr|jan.?juni|janmaerz", slug.lower()):
                continue
            items.append(
                {
                    "slug": slug,
                    "url": url,
                    "published_date": (media.get("date_gmt") or media.get("date", ""))[:10],
                }
            )
        total_pages = int(resp.headers.get("X-WP-TotalPages", "1"))
        if page >= total_pages:
            break
        page += 1
    # a few months were re-published under duplicate slugs (e.g. "-2" suffix)
    deduped = {}
    for item in items:
        deduped.setdefault(item["url"], item)
    return sorted(deduped.values(), key=lambda item: item["published_date"])



# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------
def _page_texts(pdf_bytes):
    """Extract the text of every page, dropping the control bytes pypdf emits
    for umlauts/ligatures (they break the number columns otherwise)."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [CONTROL_CHARS.sub("", page.extract_text() or "") for page in reader.pages]


def _bulletin_period(texts, url):
    """Reporting (year, month) from the bulletin header, URL as fallback."""
    for text in texts[:3]:
        match = BULLETIN_TITLE_RE.search(text)
        if match:
            month = MONTH_NAMES.get(match.group(1).lower())
            if month:
                return int(match.group(2)), month
    # "Bulletin_11_2025.pdf" / "Bulletin_11_2025.pdf"
    match = re.search(r"Bulletin[-_](\d{1,2})[-_](\d{4})", url)
    if match:
        return int(match.group(2)), int(match.group(1))
    # "Bulletin-Oktober-2018.pdf"
    match = re.search(r"Bulletin[-_]([A-Za-zÄÖÜäöüß]+)[-_](\d{4})", url)
    if match:
        month = MONTH_NAMES.get(match.group(1).lower())
        if month:
            return int(match.group(2)), month
    return None, None


DATE_STAMP_RE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")


def _publication_date(texts, fallback):
    """Publication date printed on the bulletin itself (footer "13.11.2018").

    The WordPress media date is only the date the PDF was uploaded - the whole
    back-archive was uploaded at once (2018-07-30), so it cannot be used as the
    event time of a bulletin. The footer date is the real publication date.
    """
    for text in reversed(texts):
        # the last date of a page is the footer ("13.11.2018 Seite 3 von 6");
        # earlier dates are the reporting period ("01.10.2018 bis 31.10.2018")
        for match in reversed(DATE_STAMP_RE.findall(text)):
            day, month, year = (int(part) for part in match)
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return fallback


def _split_numbers(blob):
    """Split a numeric cell run into tonnages (integers) and percentages.

    Percentages are the only values whose group separator is followed by
    exactly two digits, which separates them from the tonnage columns without
    relying on column positions. The thousands separator itself is an
    apostrophe in older bulletins (1'825'997) and a comma in newer ones
    (1,825,997), so it is not relied upon:

      2018 layout: 162'568 -47.1 86'032  2'043'604  1'523'623 -25.4
      2025 layout: 154'066 220'978 43,4 2'402'339 1'825'997 -24,0
      2026 layout: 183,946 38,791 -78.9 1,303,717 1,038,131 -20.4

    An en dash ("–") is the bulletins' "no turnover" cell and is read as 0.
    Tonnage order in all layouts:
      previous-year month, reporting month, previous-year YTD, reporting YTD.
    """
    tonnages, percentages = [], []
    # "146'896 - 6'830" would be a minus sign, "146'896 -" is an empty cell:
    # only dashes that are not followed by a digit are zero cells.
    blob = DASH_CELL_RE.sub("0", blob)
    for token in NUMBER_RE.findall(blob):
        match = PERCENT_RE.fullmatch(token)
        if match:
            percentages.append(_to_number(match.group(1) + match.group(2) + "." + match.group(3)))
            continue
        value = _to_number(token)
        if value is not None:
            tonnages.append(value)
    return tonnages, percentages


def _blob_starts_with_percent(blob):
    """True when the first number of a cell run is a percentage."""
    match = NUMBER_RE.search(blob)
    return bool(match) and bool(PERCENT_RE.fullmatch(match.group(0)))


def _record(**kwargs):
    base = {
        "prev_year_month": None,
        "prev_year_month_tonnes": None,
        "prev_year_ytd_tonnes": None,
        "current_ytd_tonnes": None,
    }
    base.update(kwargs)
    return base


def _parse_comparison_page(text, year, month, classification):
    """Parse a month/YTD comparison page (present in old and new bulletins).

    The direction column is carried by a "Zufuhr" / "Abfuhr" marker. pypdf
    renders the leading "A" of "Abfuhr" as a control byte, so the marker is
    matched as "bfuhr" as well.

    Totals are stored under the synthetic group "TOTAL" (legacy bulletins use
    group 99 for "Übrige Güter", which must not be confused with a total):

      NST-2007: the table ends with a per-direction "Summen" row.
      legacy:   the per-direction totals are glued to the marker line, e.g.
                "-42.2 219'948 380'586Zufuhr 3'926'672 3'325'623 -15.3"
    """
    records = []
    totals = {}
    groups_seen = set()
    direction = "zufuhr"

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        marker = None
        if "Zufuhr" in line:
            marker = direction = "zufuhr"
        elif re.search(r"bfuhr", line):
            marker = direction = "abfuhr"

        match = MAIN_LINE_RE.match(line)
        if match:
            group, label, blob = match.group(1), match.group(2), match.group(3)
            groups_seen.add(direction)
        elif "Summen" in line[:12]:
            group, label, blob = TOTAL_GROUP, "Summen", line.split("Summen", 1)[1]
        elif marker and classification != "NST-2007":
            group, label, blob = TOTAL_GROUP, "Total", line
        else:
            continue

        tonnages, _ = _split_numbers(blob)
        if len(tonnages) < 4:
            continue

        prev_month, current_month, prev_ytd, current_ytd = tonnages[:4]
        # Legacy marker line: "-42.2 219'948 380'586Zufuhr 3'926'672 3'325'623
        # -15.3". pypdf emits the month delta BEFORE the two month columns,
        # which reverses only the month pair (the YTD pair keeps the
        # prev/current order). A percentage in front of the first tonnage is
        # the signature of that layout; "Summen" rows never start with one.
        if group == TOTAL_GROUP and classification != "NST-2007" and _blob_starts_with_percent(blob):
            prev_month, current_month = current_month, prev_month
        record = _record(
            classification=classification,
            group=group,
            group_label=label.strip(),
            direction=direction,
            month=f"{year}-{month:02d}",
            tonnes=current_month,
            prev_year_month=f"{year - 1}-{month:02d}",
            prev_year_month_tonnes=prev_month,
            prev_year_ytd_tonnes=prev_ytd,
            current_ytd_tonnes=current_ytd,
        )
        if group == TOTAL_GROUP:
            totals[direction] = record
        else:
            records.append(record)

    # a total is only trustworthy when the same page/direction produced the
    # goods-group rows it sums up (guards against Hafenteil/Container tables)
    records.extend(record for direction, record in totals.items() if direction in groups_seen)
    if not any(record["group"] != TOTAL_GROUP for record in records):
        return []
    return records


def _parse_detail_page(text, year, month, classification):
    """Parse the detailed goods-group page (columns Zufuhr / Abfuhr / Summen).

    Newer bulletins print this table without a page title, so the caller also
    accepts pages that contain the "Richtung Zufuhr ... Summen" header.
    """
    records = []
    for raw_line in text.split("\n"):
        match = DETAIL_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        group, label, blob = match.groups()
        if classification == "legacy":
            # legacy bulletins label the fuel positions 321/323/325 instead of
            # 07.21/07.22/07.23; map them so one feature definition works for
            # both classifications.
            group = LEGACY_DETAIL_GROUPS.get(group)
            if group is None:
                continue
        tonnages, _ = _split_numbers(blob)
        if len(tonnages) < 3:
            continue
        zufuhr, abfuhr = tonnages[0], tonnages[1]
        for direction, value in (("zufuhr", zufuhr), ("abfuhr", abfuhr)):
            if value is None:
                continue
            records.append(
                _record(
                    classification=classification,
                    group=group,
                    group_label=label.strip(),
                    direction=direction,
                    month=f"{year}-{month:02d}",
                    tonnes=value,
                )
            )
    return records


def _fix_legacy_detail_directions(records):
    """Re-label the direction of legacy detail rows against the comparison page.

    The "Umgeschlagene Waren" page of the legacy bulletins prints its two
    direction columns in the opposite order of the newer ones (verified with
    October 2018: positions 321+323+325 = 86'032 t, which is the group-3
    *Zufuhr* figure of the comparison page, while the detail page lists it in
    its second column). The comparison page is authoritative, so the detail
    columns are matched against its group-3 totals instead of trusting the
    mangled page header.
    """
    targets = {}
    for record in records:
        if record["group"] == "3" and record["tonnes"] is not None:
            targets[record["direction"]] = record["tonnes"]
    detail = [record for record in records if record["group"] in CARGO_DETAIL_GROUPS]
    if len(targets) != 2 or not detail:
        return

    sums = {"zufuhr": 0.0, "abfuhr": 0.0}
    for record in detail:
        sums[record["direction"]] += record["tonnes"] or 0.0

    straight = abs(sums["zufuhr"] - targets["zufuhr"]) + abs(sums["abfuhr"] - targets["abfuhr"])
    swapped = abs(sums["abfuhr"] - targets["zufuhr"]) + abs(sums["zufuhr"] - targets["abfuhr"])
    if swapped + 1 < straight:
        for record in detail:
            record["direction"] = "abfuhr" if record["direction"] == "zufuhr" else "zufuhr"


def parse_bulletin(pdf_bytes, published_date, url):
    """Parse one monthly bulletin PDF into cargo records."""
    texts = _page_texts(pdf_bytes)
    year, month = _bulletin_period(texts, url)
    if not year or not month:
        print(f"  skipped (no reporting month found): {url}")
        return []
    published_date = _publication_date(texts, published_date)

    classification = "NST-2007" if any(NST2007_MARKER in text for text in texts) else "legacy"
    records = []
    for text in texts:
        # Container (TEU) pages have the same numeric layout as the goods-group
        # tables but a different meaning, so a page is only parsed when it
        # carries one of the goods-group table titles.
        if "TEU" in text or "Container" in text:
            continue
        if COMPARISON_TITLE in text or "Vorjahr" in text:
            records.extend(_parse_comparison_page(text, year, month, classification))
        elif any(title in text for title in DETAIL_TITLES) or re.search(r"Richtung\s*Zufuhr", text):
            # newer bulletins render the detail-page title without spaces
            # ("UmgeschlageneWareninTonnen"), the direction header is stable
            records.extend(_parse_detail_page(text, year, month, classification))

    # keep only the goods groups the feature pipeline needs; the legacy
    # classification has no 07 group - oil is group 3 ("Erdoel, Mineraloel-
    # erzeugnisse") and its detail positions are mapped onto 07.21..07.25
    if classification != "NST-2007":
        _fix_legacy_detail_directions(records)
    if classification == "NST-2007":
        wanted = {"07", TOTAL_GROUP} | set(CARGO_DETAIL_GROUPS)
    else:
        wanted = {"3", TOTAL_GROUP} | set(CARGO_DETAIL_GROUPS)
    records = [record for record in records if record["group"] in wanted]

    for record in records:
        record["published_date"] = published_date
        record["bulletin_url"] = url
        record["bulletin_year"] = year
        record["bulletin_month"] = month
        # Tonnages are whole numbers in every bulletin layout; a fractional one
        # means a thousands separator was mistaken for a decimal point.
        for field in ("tonnes", "prev_year_month_tonnes", "prev_year_ytd_tonnes", "current_ytd_tonnes"):
            value = record[field]
            if value is not None and abs(value) >= 1 and value != int(value):
                print(f"  WARNING {field}={value} in {record['month']} {record['group']}/{record['direction']}: {url}")
                break
    return records


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def _s3_enabled():
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))


def _s3():
    import boto3

    return boto3.client("s3")  # endpoint from AWS_ENDPOINT_URL


def load_cargo():
    if _s3_enabled():
        import boto3
        from botocore.exceptions import ClientError

        try:
            body = _s3().get_object(Bucket=S3_BUCKET, Key=RHEIN_CARGO_KEY)["Body"].read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "NotFound", "404"):
                return None
            raise
        return pd.read_csv(io.StringIO(body.decode("utf-8")))
    if LOCAL_CARGO_CSV.exists():
        return pd.read_csv(LOCAL_CARGO_CSV)
    return None


def save_cargo(df):
    csv = df.to_csv(index=False)
    if _s3_enabled():
        _s3().put_object(Bucket=S3_BUCKET, Key=RHEIN_CARGO_KEY, Body=csv.encode("utf-8"))
        print(f"saved s3://{S3_BUCKET}/{RHEIN_CARGO_KEY} ({len(df)} rows)")
    else:
        LOCAL_CARGO_CSV.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_CARGO_CSV.write_text(csv, encoding="utf-8")
        print(f"saved {LOCAL_CARGO_CSV} ({len(df)} rows)")


def _parsed_index_path():
    return LOCAL_CARGO_CSV.with_name("rhein_bulletins_parsed.json")


def load_parsed_bulletins():
    """URLs of bulletins that were already fetched (also the empty ones)."""
    if _s3_enabled():
        try:
            body = _s3().get_object(Bucket=S3_BUCKET, Key=RHEIN_BULLETINS_KEY)["Body"].read()
        except Exception:  # NoSuchKey on the first run
            return set()
        return set(json.loads(body.decode("utf-8")))
    path = _parsed_index_path()
    if path.exists():
        return set(json.loads(path.read_text(encoding="utf-8")))
    return set()


def save_parsed_bulletins(urls):
    payload = json.dumps(sorted(urls), indent=1)
    if _s3_enabled():
        _s3().put_object(Bucket=S3_BUCKET, Key=RHEIN_BULLETINS_KEY, Body=payload.encode("utf-8"))
    else:
        path = _parsed_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def merge_cargo(existing, new):
    """Merge on (month, classification, group, direction); first release wins.

    The archive contains the same month several times (re-uploads under
    "-2"/"NEU_" slugs). Point-in-time features must not change retroactively,
    so the value of the earliest published bulletin is kept and later copies
    are ignored.
    """
    if existing is None or existing.empty:
        merged = new
    else:
        merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.sort_values("published_date", kind="stable")
    merged = merged.drop_duplicates(subset=["month", "classification", "group", "direction"], keep="first")
    return merged.sort_values(["month", "group", "direction"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect Swiss Rhine port cargo (oil) turnover")
    parser.add_argument("--all", action="store_true", help="re-parse every bulletin in the archive")
    parser.add_argument("--limit", type=int, help="parse at most N bulletins (the newest ones)")
    parser.add_argument("--no-s3", action="store_true", help="force local files even when S3 credentials exist")
    args = parser.parse_args()

    if args.no_s3:
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

    bulletins = list_bulletins()
    print(f"media index: {len(bulletins)} bulletin PDFs")

    existing = load_cargo()
    parsed = load_parsed_bulletins()
    if existing is not None and not existing.empty:
        parsed |= set(existing["bulletin_url"])
    if args.all:
        parsed = set()
    todo = [bulletin for bulletin in bulletins if bulletin["url"] not in parsed]
    if args.limit:
        todo = todo[-args.limit :]

    print(f"bulletins to parse: {len(todo)}")
    records = []
    attempted = set()
    for bulletin in todo:
        try:
            pdf = _get(bulletin["url"], timeout=90).content
        except Exception as exc:  # a few annual/statistics PDFs 404 intermittently
            print(f"  fetch failed ({exc}): {bulletin['url']}")
            continue  # not attempted -> retried on the next run
        attempted.add(bulletin["url"])
        rows = parse_bulletin(pdf, bulletin["published_date"], bulletin["url"])
        name = bulletin["url"].rsplit("/", 1)[-1]
        print(f"  {name} -> {len(rows)} records")
        records.extend(rows)

    # remember every bulletin that was fetched, including the ones without any
    # parsable goods-group table (annual reports, scanned layouts)
    save_parsed_bulletins(parsed | attempted)

    if not records:
        print("no new cargo records")
        return 0

    merged = merge_cargo(existing, pd.DataFrame(records))
    save_cargo(merged)

    oil = merged[(merged["group"].isin(["07", "3"])) & (merged["direction"] == "zufuhr")]
    if not oil.empty:
        print(f"oil turnover records: {len(oil)}, months {oil['month'].min()} .. {oil['month'].max()}")
        print(oil.tail(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
