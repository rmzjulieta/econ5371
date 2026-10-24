"""
05_period_analysis.py

Purpose : Split each LFPR series into periods at the break years found by the
          Bai-Perron procedure (script 04), and decide, period by period,
          whether the series in levels is trend-stationary (remove the trend)
          or has a unit root (take first differences).
Input   : data/clean/lfpr_women_by_age.csv
          output/table5_qlr_baiperron.csv
Output  : output/table6_period_analysis.csv
"""

import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

# KPSS p-values are bounded between 0.01 and 0.10; this is noted in the paper,
# so the repeated warning is silenced here to keep the console readable.
warnings.simplefilter("ignore", InterpolationWarning)

# --- Paths -----------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_DIR / "data" / "clean" / "lfpr_women_by_age.csv"
BREAKS_FILE = PROJECT_DIR / "output" / "table5_qlr_baiperron.csv"
OUT_FILE = PROJECT_DIR / "output" / "table6_period_analysis.csv"

df = pd.read_csv(CLEAN_FILE, index_col="year")
breaks = pd.read_csv(BREAKS_FILE, index_col="series")

rows = []

# Outer loop: every age group.
for col in df.columns:
    # Break years are stored as text, e.g. "[1965, 1988]"; convert to a list.
    # Each break year is the FIRST year of the new period.
    bp_years = ast.literal_eval(breaks.loc[col, "bp_break_years"])
    starts = [df.index.min()] + bp_years
    ends = [b - 1 for b in bp_years] + [df.index.max()]

    # Inner loop: every period of this series.
    for start, end in zip(starts, ends):
        segment = df.loc[start:end, col]

        # Unit-root tests in levels, allowing a linear trend ("ct").
        # maxlag=2: periods are short (12+ years), so few lags are affordable.
        adf_p = adfuller(segment, regression="ct", maxlag=2, autolag="AIC",
                         result_object=False)[1]
        kpss_p = kpss(segment, regression="ct", nlags="auto",
                      result_object=False)[1]

        # Decision rule:
        #   trend-stationary  -> ADF rejects a unit root AND KPSS does not
        #                        reject stationarity -> remove the trend
        #   otherwise         -> take first differences. When the tests
        #                        disagree, differencing is the safer choice.
        if adf_p < 0.05 and kpss_p >= 0.05:
            treatment = "detrend"
        else:
            treatment = "difference"

        # Summary of the period in percentage points per year:
        # slope of the linear trend, or mean of the first differences (drift).
        years = segment.index.values
        slope = np.polyfit(years, segment.values, 1)[0]
        drift = segment.diff().mean()

        rows.append({"series": col, "start": start, "end": end,
                     "n_years": len(segment),
                     "adf_p": round(adf_p, 3), "kpss_p": round(kpss_p, 3),
                     "treatment": treatment,
                     "trend_slope": round(slope, 2),
                     "mean_change": round(drift, 2)})

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(OUT_FILE, index=False)