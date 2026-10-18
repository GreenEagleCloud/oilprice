"""Tests for the frontend UX: readable feature labels and template wiring.

No network and no feature store - these tests read the Flask app of
src/frontend and the single-page template it renders. src/frontend/app.py
repeats the model feature names because its image cannot import src/hw_config.py,
so the label coverage is checked against the real feature lists here.
"""

import pathlib
import re
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "frontend"))

import app as frontend  # noqa: E402  (src/frontend/app.py)
import hw_config  # noqa: E402

TEMPLATE = REPO_ROOT / "src" / "frontend" / "templates" / "index.html"


@pytest.fixture(scope="module")
def page():
    """Rendered index page (read once per module)."""
    return frontend.app.test_client().get("/").get_data(as_text=True)


def all_model_features():
    return hw_config.weekly_model_features() + hw_config.forward_model_features()


def test_every_model_feature_has_a_label():
    """Without a label the table falls back to the raw feature-store key."""
    for name in all_model_features():
        assert name in frontend.FEATURE_LABELS, f"missing label for {name}"


def test_labels_have_the_shape_the_template_expects():
    # index.html reads entry.label / entry.unit / entry.group
    for name, entry in frontend.FEATURE_LABELS.items():
        assert set(entry) == {"label", "unit", "group"}, name
        assert isinstance(entry["label"], str) and entry["label"], name
        assert isinstance(entry["group"], str) and entry["group"], name
        assert isinstance(entry["unit"], str), name


def test_origin_features_are_labelled_as_origin_values():
    for name in hw_config.forward_model_features():
        if not name.startswith("origin_"):
            continue
        assert frontend.FEATURE_LABELS[name]["label"].startswith("Ausgangstag:"), name


def test_frontend_feature_names_match_the_pipeline():
    """The duplicated list in src/frontend/app.py must not drift."""
    assert set(frontend.FEATURE_LABELS) >= set(all_model_features())


def test_feature_table_uses_the_label_text(page):
    assert "FEATURE_LABELS" in page
    assert "row.label.label" in page
    assert "row.label.unit" in page


def test_training_label_is_not_rendered_as_a_feature(page):
    """The weekly feature view carries target_next_week_price (the answer for
    that date); showing it as a model input would leak the target."""
    start = page.index("const LABEL_KEYS")
    body = page[start: page.index("function prefillForm")]
    assert "target_next_week_price" in body
    assert "LABEL_KEYS.has(key)" in body


def test_every_element_the_script_touches_exists(page):
    """Guards against renaming a card without renaming its getElementById()."""
    ids = set(re.findall(r"getElementById\('([^']+)'\)", page))
    assert ids, "the inline script references no element"
    for element_id in sorted(ids):
        assert f'id="{element_id}"' in page, f"missing element #{element_id}"


def test_price_mean_is_derived_from_the_price_series(page):
    """The 7-day mean must come from the loaded prices, not from the API."""
    assert "movingAverage(priceData.price || [], 7)" in page
    assert "priceMean7" in page


def test_cargo_fallback_destroys_the_chart(page):
    """A missing cargo payload must not leave an empty canvas behind."""
    start = page.index("function renderCargoChart")
    body = page[start: page.index("Feature table with readable names")]
    assert "cargoChart.destroy()" in body
    assert "cargoChart = null" in body
    assert "document.getElementById('cargoChart').hidden = true" in body
    assert "document.getElementById('cargoSkeleton').hidden = true" in body
    assert "Keine Fracht-Daten verfügbar." in body


def test_template_is_well_formed(page):
    assert page.count("<html") == 1
    assert page.rstrip().endswith("</html>")


def test_style_block_has_no_unclosed_rule(page):
    """An unclosed @media swallows every rule that follows it: those rules then
    apply only under that media condition, so the page looks unstyled for
    everyone without prefers-reduced-motion."""
    style = page[page.index("<style>") + len("<style>"): page.index("</style>")]
    assert style.count("{") == style.count("}"), "unbalanced braces in <style>"


def test_reduced_motion_block_is_scoped_to_animations_only(page):
    """The reduced-motion override must not become the container of the layout."""
    start = page.index("@media (prefers-reduced-motion: reduce)")
    body = page[start: page.index("*/", page.index("Info tooltips"))]
    assert "animation: none" in body
    assert ".info-btn" not in body and ".stats" not in body and ".two-col" not in body


# ---------------------------------------------------------------------------
# Forecast curve on the price chart
# ---------------------------------------------------------------------------

def test_price_chart_draws_the_forecast_curve(page):
    """The prediction must not live only in a card: the line continues into the
    future with the weekly model first and the forward model further out."""
    body = page[page.index("function renderPriceChart"): page.index("function levelFooter")]
    assert "'Prognose (Wochenmodell)'" in body
    assert "'Prognose (Vorhersagemodell)'" in body
    # the cone is drawn by filling the upper edge down to the lower edge, in the
    # brand blue - it is the swing band, not a second price line
    assert "fill: '+1'" in body
    assert "const FORECAST_BAND = 'rgba(6, 106, 171, .16)'" in page
    assert "filter: (item) => !item.text.startsWith('Prognose-Band')" in body
    assert "forecastSeries(dates)" in body
    assert "crosshairPlugin, todayPlugin" in body
    # the x axis has to reach into the future or the forecast points have no
    # label to sit on; the history arrays stay aligned by padding with nulls
    assert "const dates = extendedDates(historyDates);" in body
    assert "new Array(dates.length - values.length).fill(null)" in body


def test_forecast_continues_the_seven_day_mean(page):
    """The red forecast must not start in mid-air: the black 7-day mean runs on
    to the origin day, the weekly model is drawn on each of the seven days it
    describes, and the forward model continues at the end of that week."""
    body = page[page.index("function forecastSeries"): page.index("function measuredForecastPoints")]
    assert "const weeklyPoint = points.find((p) => p.model === 'weekly')" in body
    assert "mean[originIdx] = meanAtOrigin;" in body
    assert "for (let day = 1; day <= WEEKLY_FORECAST_DAYS; day++)" in body
    assert "weekly[i] = weeklyPoint.value;" in body
    assert "mean[weekEndIdx] = weeklyPoint.value;" in body
    assert "line[weekEndIdx] = weeklyPoint.value;" in body
    # the cone covers the first week as well, not only the forward model
    assert "upper[i] = Math.round((weeklyPoint.value + weeklyPoint.band) * 100) / 100;" in body
    # the models answer one value per week; the line through those values must
    # still be one line instead of a row of islands
    chart = page[page.index("function renderPriceChart"): page.index("function levelFooter")]
    assert "borderDash: [7, 5], tension: .2, pointRadius: 0, spanGaps: true" in chart
    # the days of the weekly week need axis labels
    axis = page[page.index("function extendedDates"): page.index("function forecastSeries")]
    assert "for (let day = 1; day <= WEEKLY_FORECAST_DAYS; day++)" in axis
    chart = page[page.index("function renderPriceChart"): page.index("function levelFooter")]
    assert "{label: 'Prognose 7-Tage-Mittel', data: forecast[4]" in chart
    # the weekly model predicts an average, so it is not a single big point
    weekly_line = chart[chart.index("{label: 'Prognose (Wochenmodell)'"): chart.index("{label: 'Prognose 7-Tage-Mittel'")]
    assert "pointRadius: 0" in weekly_line
    # a weekly prediction is compared with the mean of its seven days
    diff = page[page.index("function measuredForecastPoints"): page.index("function renderPriceChart")]
    assert "const mean = week.reduce((sum, v) => sum + v, 0) / week.length;" in diff


def test_forecast_origin_reads_the_value_from_the_whole_history(page):
    """The chart labels are a window of the history plus the forecast days, so
    their numbers are not the numbers of the price arrays. The value the
    forecast hangs on has to be looked up in the full history - reading it by
    the window index picks the mean of some day years back and drops the
    forecast line down onto the demand curve."""
    body = page[page.index("function forecastSeries"): page.index("function measuredForecastPoints")]
    assert "valueAt(priceData.dates, priceMean7, origin)" in body
    assert "valueAt(priceData.dates, priceData.price, origin)" in body
    assert "valueAt(dates, priceMean7" not in body
    assert "valueAt(dates, priceData.price" not in body


def test_forecast_curve_is_loaded_without_pressing_the_button(page):
    """The chart is extended by default, so the forecast follows the history."""
    assert "loadPriceHistory()\n            .then(() => loadForecast())" in page
    assert "function forecastSeries(dates)" in page
    # the cone reaches three months: the forward model serves up to 90 days
    assert frontend.FORECAST_MAX_HORIZON >= 90


def test_measured_prices_run_into_the_forecast_region(page):
    """The real daily prices must not stop at the forecast origin: the measured
    days of the forecast window are a dataset of their own, and the x axis has
    to carry a label for each of them."""
    body = page[page.index("function extendedDates"): page.index("function renderPriceChart")]
    assert "forecastData.measured" in body
    assert "return [upper, lower, line, weekly, mean, measured];" in body
    chart = page[page.index("function renderPriceChart"): page.index("function levelFooter")]
    assert "'Tatsächlich gemessen'" in chart
    assert "data: forecast[5]" in chart
    # days of the forecast window that are not measured yet must not pretend a
    # measurement: the readout says so explicitly for those days
    assert "kein gemessener Preis" in page
    # the update path writes every forecast array, so the new series keeps its
    # data when the timeframe or the forecast changes (the three history series
    # come first, then the six forecast arrays)
    assert "forecast.forEach((values, i) => { priceChart.data.datasets[3 + i].data = values; });" in chart


def test_measured_history_is_one_solid_line_of_every_day(page):
    """The past until today must read as the real daily price: a solid line and
    every single day of the series - no dotted history, no sampled subset."""
    chart = page[page.index("function renderPriceChart"): page.index("function levelFooter")]
    price_line = chart[chart.index("{label: 'Preis (gemessen)'"): chart.index("{label: '7-Tage-Mittel'")]
    assert "borderDash" not in price_line
    assert "data: history[0]" in price_line
    assert "spanGaps: true" in price_line
    # the arrays handed to the chart are the raw slices of the daily series
    assert "pad(priceData.price.slice(start))" in chart
    assert "pad(priceMean7.slice(start))" in chart
    # the old sampler that dropped days out of long windows must stay removed
    assert "thinIndices" not in page
    # the measured days of the forecast window are drawn in the same green, so
    # history and measurement are one line and not two competing ones
    measured_line = chart[chart.index("{label: 'Tatsächlich gemessen'"): chart.index("],\n                },\n                options:")]
    assert "borderColor: '#38f75c'" in measured_line
    assert "borderDash" not in measured_line


def test_demand_index_is_a_series_of_its_own(page):
    """The Nachfrageindex belongs back into the price chart, on its own axis."""
    chart = page[page.index("function renderPriceChart"): page.index("function levelFooter")]
    assert "{label: 'Nachfrageindex', data: history[2]" in chart
    assert "yAxisID: 'y1'" in chart
    assert "y1: {" in chart
    # a second grid over the price grid would make the chart unreadable
    assert "grid: {drawOnChartArea: false}" in chart
    # the index is not a price: the hover points must only sit on the price axis
    marker = page[page.index("function hoverMarkers"): page.index("const crosshairPlugin")]
    assert "dataset.yAxisID !== 'y'" in marker


def test_x_axis_is_labelled_by_month(page):
    """Like heizoel24.ch: the axis names the months, not a date every few ticks."""
    body = page[page.index("function monthTicks"): page.index("function formatDate")]
    assert "autoSkip: false" in body
    assert "return monthLabel(label)" in body
    # long windows would overlap, so every n-th month gets a label and January
    # always does - the year is never lost
    assert "Math.ceil(months.length / maxLabels)" in body
    assert "if (starts.get(month) !== index) return ''" in body
    assert "MONTH_NAMES = ['Jan', 'Feb', 'Mär'" in page
    assert "month === 1 ? `${name} ${String(dateStr).slice(0, 4)}` : name" in page
    for chart_name in ("function renderPriceChart", "function renderRheinChart"):
        chart = page[page.index(chart_name): page.index(chart_name) + 6000]
        assert "x: {ticks: monthTicks(dates)" in chart
    # the exact date is not lost, it moves into the tooltip
    assert "title: (items) => formatDate(items[0].label)" in page


def test_readout_box_shows_price_and_date(page):
    """The box above the chart answers "what price, which date?" without hovering."""
    assert '<div class="readout" id="priceReadout" aria-live="polite">' in page
    body = page[page.index("function readoutParts"): page.index("function renderPriceChart")]
    assert "formatDate(date)" in body
    assert "latestMeasuredDate()" in body
    assert "letzte Messung" in body
    assert "valueAt(priceData.dates, priceData.demand, date)" in body
    # hovering the chart drives the same box
    hover = page[page.index("function setHover"): page.index("function handleHover")]
    assert "updateReadout()" in hover


def test_crosshair_is_a_thin_grey_line(page):
    """heizoel24.ch marks the hovered day with a thin grey line, not a coloured
    bar - the price line has to stay the loudest thing in the chart."""
    body = page[page.index("const crosshairPlugin"): page.index("const todayPlugin")]
    assert "ctx.lineWidth = 1" in body
    assert "rgba(92, 107, 122" in body
    assert "setLineDash" not in body


def test_page_carries_the_green_eagle_brand(page):
    """Logo and favicon are static/logo.svg; the palette is the brand palette of
    greeneagle.cloud (logo green, brand blue, alert red, ink from the logo)."""
    assert '<link rel="icon" type="image/svg+xml" href="/static/logo.svg">' in page
    assert '<img class="brand-logo" src="/static/logo.svg" alt="greenEagle.cloud-Logo">' in page
    assert "greenEagle.cloud" in page
    for token in ("--green: #38f75c", "--accent: #066aab", "--red: #ec5146", "--ink: #222f3e"):
        assert token in page
    # the logo has to ship inside the frontend image, otherwise both 404
    assert (REPO_ROOT / "src" / "frontend" / "static" / "logo.svg").exists()


def test_forecast_note_shows_prediction_against_measurement(page):
    """Once a forecasted week is measured, the deviation belongs under the chart."""
    body = page[page.index("function updateForecastNote"): page.index("async function loadForecast")]
    assert "measuredForecastPoints()" in body
    assert "Prognose minus gemessen" in body
    assert "function measuredForecastPoints()" in page


def test_forecast_failure_renders_a_retry_card(page):
    """A failed forecast request must not leave the chart silently unextended."""
    body = page[page.index("function showForecastError"): page.index("function updateForecastNote")]
    assert "forecastError" in body
    assert "box.hidden = false" in body
    assert "forecastRetry" in body
    assert 'id="forecastError" class' not in body  # the card is in the markup, not only in JS
    assert '<div class="error-card" id="forecastError" hidden></div>' in page


def test_forecast_note_states_origin_and_horizon(page):
    """A forecast on a chart is only trustworthy with its origin and its band."""
    body = page[page.index("function updateForecastNote"): page.index("async function loadForecast")]
    assert "Ausgangstag ${formatDate(forecastData.origin_date)}" in body
    assert "Wochenmodell" in body and "Vorhersagemodell" in body
    assert "Blauer Kegel" in body
    assert 'id="forecastNote" aria-live="polite"' in page


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def test_view_is_split_into_three_tabs(page):
    tabs = re.findall(r'id="(tab-[a-z]+)" role="tab"', page)
    assert tabs == ["tab-preise", "tab-rhein", "tab-fracht"]
    panels = re.findall(r'id="(panel-[a-z]+)" role="tabpanel"', page)
    assert panels == ["panel-preise", "panel-rhein", "panel-fracht"]
    # every tab controls exactly the panel that carries its sections
    for name in ("preise", "rhein", "fracht"):
        assert f'aria-controls="panel-{name}"' in page
        assert f'aria-labelledby="tab-{name}"' in page
    assert 'id="tab-preise" role="tab"\n                aria-selected="true"' in page
    # the two inactive panels start hidden
    assert page.count('role="tabpanel" aria-labelledby="tab-rhein" hidden') == 1
    assert page.count('role="tabpanel" aria-labelledby="tab-fracht" hidden') == 1


def test_tab_panels_hold_the_right_sections(page):
    panel = page[page.index('id="panel-rhein"'): page.index('id="panel-fracht"')]
    assert 'id="rheinChart"' in panel and 'id="rheinStats"' in panel
    assert 'id="priceChart"' not in panel
    panel = page[page.index('id="panel-fracht"'): page.index("</main>")]
    assert 'id="cargoChart"' in panel and 'id="cargoNote"' in panel
    # the forecast belongs to the price tab: the chart draws the same curve
    price_panel = page[page.index('id="panel-preise"'): page.index('id="panel-rhein"')]
    assert 'id="priceChart"' in price_panel and 'id="forecastForm"' in price_panel
    assert 'id="forecastNote"' in price_panel


def test_tabs_are_keyboard_navigable(page):
    body = page[page.index("function selectTab"): page.index("* Info tooltips (info icon")]
    assert "ArrowRight" in body and "ArrowLeft" in body and "Home" in body and "End" in body
    assert "setAttribute('aria-selected'" in body
    assert "panel.hidden = !selected" in body


def test_date_input_is_clamped_to_the_supported_horizon(page):
    """The forward model serves 1-90 days from the origin date; the field must
    not offer a date the backend would reject."""
    body = page[page.index("function prefillForm"): page.index("addEventListener('submit'")]
    assert "dateInput.min = formatDate(today);" in body
    assert "dateInput.max = formatDate(latest);" in body
    assert "addDays(origin, 90)" in body
    # local date arithmetic: toISOString() would shift the day for every zone
    # east of UTC and the field would show the wrong range
    for anchor in ("const addDays", "function todayISO"):
        chunk = page[page.index(anchor): page.index(anchor) + 400]
        assert "toISOString" not in chunk


# ---------------------------------------------------------------------------
# Forecast form inputs and the hover marking of the price chart
# ---------------------------------------------------------------------------

def test_forecast_form_only_asks_for_the_date(page):
    """Price and demand come from the feature store: asking for them as live
    inputs only produced values nobody could check."""
    form = page[page.index('id="forecastForm"'): page.index("</form>")]
    assert 'id="dateInput"' in form
    assert 'id="priceInput"' not in form and 'id="demandInput"' not in form
    assert "current_price" not in form and "current_demand" not in form
    # the prediction request sends the date only
    body = page[page.index("const body = {date"): page.index("const res = await fetch('/predict'")]
    assert body.strip().startswith("const body = {date: isoDate};")
    # and so does the curve request
    assert "fetch('/api/forecast')" in page


def test_forecast_date_is_written_as_dd_mm_yyyy(page):
    """The page speaks dd.mm.yyyy. A native date field prints the locale of the
    browser instead - 10/19/2026 in en-US - which is not this page's format."""
    form = page[page.index('id="forecastForm"'): page.index("</form>")]
    assert 'type="date"' not in form
    assert 'inputmode="numeric"' in form
    assert "TT.MM.JJJJ" in form
    # the field text is turned back into the ISO date the API expects, and a
    # date that does not exist (31.02.) is refused instead of rolled over
    parse = page[page.index("function parseGermanDate"): page.index("function defaultForecastDate")]
    assert "const match = String(text || '').trim().match(/^(\\d{1,2})\\.(\\d{1,2})\\.(\\d{4})$/);" in parse
    assert "date.getFullYear() !== year" in parse
    sent = page[page.index("const isoDate = parseGermanDate"): page.index("const res = await fetch('/predict'")]
    assert "if (!isoDate)" in sent
    assert "TT.MM.JJJJ" in sent
    # every date the page prints goes through formatDate()
    assert "Prognose für die Woche nach ${formatDate(payload.date)}" in page


def test_forecast_date_defaults_to_the_first_day_without_measurement(page):
    """Standard is the day after the last measured price - a forecast asks about
    the future, not about a day that is already measured. With the last
    measurement on 18.10.2026 the field must read 19.10.2026."""
    body = page[page.index("function defaultForecastDate"): page.index("addEventListener('submit'")]
    assert "const next = origin ? addDays(origin, 1) : today;" in body
    assert "return next > today ? next : today;" in body
    assert "const standard = defaultForecastDate();" in body
    assert "dateInput.value = formatDate(standard);" in body
    assert "const today = todayISO();" in body
    assert "const origin = latestMeasuredDate();" in body


def test_hover_marks_the_day_without_a_box_in_the_chart(page):
    """The grey line snaps to the nearest day and its points mark it. The values
    of that day are read above the chart, so nothing overlaps the diagram."""
    for removed in ("priceHoverCard", "hover-card", "renderHoverCard", "positionHoverCard"):
        assert removed not in page, f"{removed} must be gone with the hover box"
    # the built-in tooltip stays off: the readout above the chart says the same
    chart = page[page.index("function renderPriceChart"): page.index("function levelFooter")]
    assert "tooltip: {enabled: false}" in chart
    # hovering drives the readout, leaving the chart clears the mark again
    hover = page[page.index("function setHover"): page.index("function handleHover")]
    assert "updateReadout()" in hover
    # the cursor is snapped to a day, never to the pixel between two days
    snap = page[page.index("function handleHover"): page.index("const hoverOptions")]
    assert "Math.round(chart.scales.x.getValueForPixel(evt.x))" in snap
    # a hovered day without a measurement says so instead of reading as 0
    assert "kein gemessener Preis" in page


def test_hover_points_mark_every_line_of_the_snapped_day(page):
    """The line must point at something: one point per drawn line of the price
    axis, each in the colour of its own line."""
    body = page[page.index("const crosshairPlugin"): page.index("const todayPlugin")]
    assert "for (const marker of hoverMarkers(chart, idx))" in body
    assert "ctx.arc(x, y, 4" in body
    assert "ctx.fillStyle = marker.color" in body
    marker = page[page.index("function hoverMarkers"): page.index("const crosshairPlugin")]
    # the band edges stay invisible, the demand index has its own axis
    assert "if (dataset.borderWidth === 0) continue" in marker
    assert "if (dataset.hoverMarker === false) continue" in marker
    assert "markers.push({value: value, color: dataset.borderColor" in marker
    # the horizontal low-water marks of the Rhine chart get no point
    assert page.count("hoverMarker: false}") == 2



def test_hover_is_wired_the_way_chartjs_calls_it(page):
    """Chart.js ruft onHover mit (event, aktive Elemente, chart) auf. Stehen
    die Parameter in einer anderen Reihenfolge, hält handleHover ein Array in
    der Hand und Linie und Punkte tauchen nie auf - auch nicht nach dem
    Deploy auf die lebende Seite."""
    options = page[page.index("const hoverOptions"): page.index("Skeleton loaders")]
    assert "onHover: (evt, elements, chart) => handleHover(chart, evt, elements)" in options
    # Chart.js kennt kein onLeave: der Canvas meldet das Verlassen der Fläche
    assert "onLeave:" not in options
    assert "addEventListener('mouseleave', () => setHover(null))" in options
    # beide Zeitdiagramme reagieren, nicht nur das Preisdiagramm
    assert "['priceChart', 'rheinChart'].forEach" in options
