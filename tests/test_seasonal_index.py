"""Leakage and contract tests for ExpandingSeasonalIndex (no network, no S3)."""

import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from seasonal_index import ExpandingSeasonalIndex  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402


def _frame(start: str, end: str) -> pd.DataFrame:
    """Daily frame with a rising trend plus a fixed December-peak seasonality."""
    dates = pd.date_range(start, end, freq="D")
    t = np.arange(len(dates), dtype="float64")
    seasonal = 1.0 + 0.10 * (dates.month == 12) - 0.08 * (dates.month == 6)
    price = (40.0 + 0.02 * t) * seasonal
    return pd.DataFrame(
        {
            "date": dates,
            "month": dates.month,
            "price_mean_7d": price,
            "season": pd.Series(dates.month).map(
                {12: "Winter", 1: "Winter", 2: "Winter", 3: "Frühling", 4: "Frühling",
                 5: "Frühling", 6: "Sommer", 7: "Sommer", 8: "Sommer",
                 9: "Herbst", 10: "Herbst", 11: "Herbst"}
            ),
        }
    )


def test_no_lookahead_short_fit_equals_long_fit_for_past_rows():
    """A 2010 row must score identically whether the fit saw 2015 or 2022."""
    short = _frame("2003-01-01", "2015-12-31")
    long = _frame("2003-01-01", "2022-12-31")

    si_short = ExpandingSeasonalIndex().fit(short)
    si_long = ExpandingSeasonalIndex().fit(long)

    probe = long[long["date"].dt.year == 2010]
    a = si_short.transform(probe)["seasonal_index"].to_numpy()
    b = si_long.transform(probe)["seasonal_index"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_same_year_rows_do_not_influence_each_other():
    """Removing the second half of 2015 must not change the first half of 2015."""
    full = _frame("2003-01-01", "2015-12-31")
    truncated = full[full["date"] < "2015-07-01"]

    si_full = ExpandingSeasonalIndex().fit(full)
    si_trunc = ExpandingSeasonalIndex().fit(truncated)

    probe = full[(full["date"] >= "2015-01-01") & (full["date"] < "2015-07-01")]
    np.testing.assert_allclose(
        si_full.transform(probe)["seasonal_index"].to_numpy(),
        si_trunc.transform(probe)["seasonal_index"].to_numpy(),
        rtol=0,
        atol=0,
    )


def test_min_years_fallback_before_enough_history():
    si = ExpandingSeasonalIndex(min_years=2).fit(_frame("2003-01-01", "2005-12-31"))
    early = si.transform(_frame("2003-01-01", "2003-12-31"))["seasonal_index"]
    assert (early == 1.0).all()  # 2003 has zero strictly-past years

    later = si.transform(_frame("2005-01-01", "2005-12-31"))["seasonal_index"]
    assert later.std() > 0  # 2005 can use 2003 + 2004


def test_index_is_trend_robust_and_has_real_spread():
    si = ExpandingSeasonalIndex().fit(_frame("2003-01-01", "2020-12-31"))
    values = si.transform(_frame("2021-01-01", "2021-12-31"))
    by_month = values.groupby(values["month"])["seasonal_index"].mean()
    assert by_month[12] > 1.05          # December peak survives the trend
    assert by_month[6] < 0.95           # June trough survives the trend
    assert by_month.max() - by_month.min() > 0.15


def test_column_is_replaced_not_duplicated_inside_pipeline():
    """out_col already exists in X -> the pipeline must not emit it twice."""
    frame = _frame("2003-01-01", "2015-12-31")
    frame["seasonal_index"] = 0.111  # stale, leaky value that must be overwritten

    numeric_cols = [c for c in frame.columns if c not in ("season", "date")]
    pipe = Pipeline(
        [
            ("feature_engineering", ExpandingSeasonalIndex()),
            ("preprocessing", ColumnTransformer(
                [
                    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["season"]),
                    ("numeric", "passthrough", numeric_cols),
                ]
            )),
        ]
    )
    out = pipe.fit_transform(frame)
    names = list(pipe.named_steps["preprocessing"].get_feature_names_out())
    assert len(names) == len(set(names)), names
    assert names.count("numeric__seasonal_index") == 1
    assert out.shape[1] == len(names)


def test_fitted_attributes_are_not_constructor_params():
    si = ExpandingSeasonalIndex().fit(_frame("2003-01-01", "2010-12-31"))
    assert not [k for k in si.get_params() if k.endswith("_")]
    assert hasattr(si, "index_by_year_") and hasattr(si, "global_index_")


def test_artifact_is_importable_from_another_process():
    """Custom classes must live in an importable module, never in __main__."""
    si = ExpandingSeasonalIndex().fit(_frame("2003-01-01", "2010-12-31"))
    restored = pickle.loads(pickle.dumps(si))
    assert restored.__class__.__module__ == "seasonal_index"
    assert restored.index_by_year_ == si.index_by_year_


def test_serving_frame_without_date_uses_frozen_map():
    si = ExpandingSeasonalIndex().fit(_frame("2003-01-01", "2020-12-31"))
    serving = pd.DataFrame({"month": list(range(1, 13))})
    values = si.transform(serving)["seasonal_index"]
    assert len(values) == 12
    assert values.notna().all()
    assert abs(values.iloc[11] - si.global_index_[12]) < 1e-12


def test_missing_price_column_raises_clearly():
    frame = _frame("2003-01-01", "2010-12-31").drop(columns=["price_mean_7d"])
    with pytest.raises(KeyError, match="price_mean_7d"):
        ExpandingSeasonalIndex().fit(frame)


def test_unfitted_transform_raises():
    with pytest.raises(ValueError, match="not fitted"):
        ExpandingSeasonalIndex().transform(_frame("2003-01-01", "2003-12-31"))


def test_numpy_input_rejected_with_clear_contract_error():
    """The transformer selects columns by name, so arrays must fail loudly."""
    frame = _frame("2003-01-01", "2010-12-31")
    with pytest.raises(TypeError, match="requires a pandas DataFrame"):
        ExpandingSeasonalIndex().fit(frame[["price_mean_7d"]].to_numpy())
