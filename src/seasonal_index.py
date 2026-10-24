"""Point-in-time seasonal index (no look-ahead leakage).

Notebook 04 / src/hw_feature_pipeline.py build `seasonal_index` as

    ratio(year, month) = mean(price | year, month) / mean(price | year)
    seasonal_index(m)  = mean(ratio(y, m) over ALL years in the frame)

The final average is taken over every year present in the frame, so a 2010 row is
scored with a coefficient that used 2022 data. A chronological train/test split
cannot undo this: the leak is baked into the feature value before the split.

`ExpandingSeasonalIndex` keeps the same trend-robust definition but freezes it per
calendar year. A row in year Y is scored with the average ratio over years strictly
older than Y, so the coefficient available to a row never sees that row's own year
or any later year. The fitted state is a dict of cumulative maps, which makes the
same artifact usable for honest walk-forward CV and for production inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["ExpandingSeasonalIndex"]


class ExpandingSeasonalIndex(BaseEstimator, TransformerMixin):
    """Compute the month-vs-year price index from training rows only.

    Parameters
    ----------
    date_col : str
        Column used to derive the calendar year of each row. Optional at
        transform time; when absent the frozen end-of-fit map is used.
    price_col : str
        Price column the ratio table is built from. Must be a feature, never the
        target - target-derived seasonal indices are target encoding, not seasonality.
    out_col : str
        Name of the generated column.
    min_years : int
        Number of strictly-past years required before the index is trusted.
        Rows with fewer history years fall back to `fallback`.
    fallback : float
        Neutral value (1.0 = yearly average) used when no history exists.
    """

    def __init__(
        self,
        date_col: str = "date",
        price_col: str = "price_mean_7d",
        out_col: str = "seasonal_index",
        min_years: int = 2,
        fallback: float = 1.0,
    ):
        # Only constructor parameters here - sklearn forbids fitted attributes
        # in __init__ (check_estimator enforces this).
        self.date_col = date_col
        self.price_col = price_col
        self.out_col = out_col
        self.min_years = min_years
        self.fallback = fallback

    # ------------------------------------------------------------------
    def _check_frame(self, X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"{type(self).__name__} selects columns by name ('{self.date_col}', "
                f"'{self.price_col}') and therefore requires a pandas DataFrame; "
                f"got {type(X).__name__}. This transformer is not compatible with "
                f"numpy-array pipelines or sklearn's check_estimator."
            )

    def _dates(self, X: pd.DataFrame) -> pd.Series:
        if self.date_col in X.columns:
            return pd.to_datetime(pd.Series(X[self.date_col].values), errors="raise")
        return None

    def fit(self, X: pd.DataFrame, y=None):
        self._check_frame(X)
        if self.price_col not in X.columns:
            raise KeyError(
                f"'{self.price_col}' missing during fit; the seasonal ratio table "
                f"needs a price column."
            )
        dates = self._dates(X)
        if dates is None:
            raise KeyError(
                f"'{self.date_col}' missing during fit; the seasonal index must be "
                f"built per calendar year to stay point-in-time."
            )

        table = pd.DataFrame(
            {
                "year": dates.dt.year.values,
                "month": dates.dt.month.values,
                "price": pd.Series(X[self.price_col].values, dtype="float64").values,
            }
        )
        table = table[np.isfinite(table["price"]) & (table["price"] > 0)]
        if table.empty:
            raise ValueError(
                f"No usable positive values in '{self.price_col}' to build the "
                f"seasonal ratio table."
            )

        yearly = table.groupby("year")["price"].mean()
        ym = table.groupby(["year", "month"])["price"].mean()
        ratio = ym / ym.index.get_level_values("year").map(yearly)

        # Cumulative average of the ratios, frozen at the end of each year.
        # index_by_year_[Y] may only use data from years <= Y.
        cumulative: dict[int, dict[int, float]] = {}
        seen: dict[int, list[float]] = {}
        for year in sorted(ratio.index.get_level_values("year").unique()):
            for month, value in ratio.loc[year].items():
                seen.setdefault(int(month), []).append(float(value))
            cumulative[int(year)] = {
                month: float(np.mean(vals)) for month, vals in seen.items()
            }

        self.index_by_year_ = cumulative
        self.years_ = sorted(cumulative)
        self.global_index_ = {
            month: float(np.mean(vals)) for month, vals in seen.items()
        }
        self.fallback_index_ = float(self.fallback)
        return self

    # ------------------------------------------------------------------
    def _lookup(self, year: int, month: int) -> float:
        """Index for (year, month) using only strictly-past years."""
        usable = [y for y in self.years_ if y < year]
        if len(usable) < self.min_years:
            return self.fallback_index_
        return self.index_by_year_[usable[-1]].get(month, self.fallback_index_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_frame(X)
        if not hasattr(self, "index_by_year_"):
            raise ValueError("ExpandingSeasonalIndex is not fitted yet.")

        X_out = X.copy()
        dates = self._dates(X_out)
        if dates is None:
            # Serving frame without dates: use the map frozen at end of fit.
            values = [
                self.global_index_.get(int(m), self.fallback_index_)
                for m in X_out["month"]
            ]
        else:
            values = [
                self._lookup(int(y), int(m))
                for y, m in zip(dates.dt.year.values, dates.dt.month.values)
            ]

        X_out[self.out_col] = values
        return X_out

    def get_feature_names_out(self) -> list[str]:
        return [self.out_col]
