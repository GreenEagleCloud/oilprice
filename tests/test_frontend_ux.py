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
    # the band is drawn by filling the upper edge down to the lower edge
    assert "fill: '+1'" in body
    assert "forecastSeries(dates)" in body
    assert "crosshairPlugin, todayPlugin" in body
    # the x axis has to reach into the future or the forecast points have no
    # label to sit on; the history arrays stay aligned by padding with nulls
    assert "const dates = extendedDates(historyDates);" in body
    assert "new Array(dates.length - values.length).fill(null)" in body


def test_forecast_curve_is_loaded_without_pressing_the_button(page):
    """The chart is extended by default, so the forecast follows the history."""
    assert "loadPriceHistory()\n            .then(() => loadForecast())" in page
    assert "function forecastSeries(dates)" in page
    # the dotted line has to start at the last measured price, not in mid-air
    assert "line[firstIdx - 1] = anchor;" in page


def test_measured_prices_run_into_the_forecast_region(page):
    """The real daily prices must not stop at the forecast origin: the measured
    days of the forecast window are a dataset of their own, and the x axis has
    to carry a label for each of them."""
    body = page[page.index("function extendedDates"): page.index("function renderPriceChart")]
    assert "forecastData.measured" in body
    assert "return [upper, lower, line, weekly, measured];" in body
    chart = page[page.index("function renderPriceChart"): page.index("function levelFooter")]
    assert "'Tatsächlich gemessen'" in chart
    assert "data: forecast[4]" in chart
    # the update path writes every forecast array, so the new series keeps its
    # data when the timeframe or the forecast changes
    assert "forecast.forEach((values, i) => { priceChart.data.datasets[2 + i].data = values; });" in chart


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
    assert "Ausgangstag ${forecastData.origin_date}" in body
    assert "Wochenmodell" in body and "Vorhersagemodell" in body
    assert "Schwankungsband" in body
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
    """The forward model serves 1-90 days from the origin date; the picker must
    not offer a date the backend would reject."""
    assert "dateInput.min = addDays(date, 1);" in page
    assert "dateInput.max = addDays(date, 90);" in page
    # local date arithmetic: toISOString() would shift the day for every zone
    # east of UTC and the picker would offer the wrong range
    body = page[page.index("const addDays"): page.index("const addDays") + 400]
    assert "toISOString" not in body
