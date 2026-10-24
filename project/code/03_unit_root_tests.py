"""
03_unit_root_tests.py

Purpose : Test each female LFPR series for a unit root, in levels and in
          first differences, using ADF and KPSS. The results decide whether
          the structural break tests are run on levels or on differences.
Input   : data/clean/lfpr_women_by_age.csv
Output  : output/table2_unit_root_tests.csv
"""

from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

# --- Paths -----------------------------------------------------------------
# Same convention as 01 and 02: paths are built from this script's location.
PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_DIR / "data" / "clean" / "lfpr_women_by_age.csv"
OUT_FILE = PROJECT_DIR / "output" / "table2_unit_root_tests.csv"
ZA_FILE = PROJECT_DIR / "output" / "table3_zivot_andrews.csv"

df = pd.read_csv(CLEAN_FILE, index_col="year")

# --- Run both tests on every series, in levels and in differences ------------
# ADF null: the series has a unit root (non-stationary).
# KPSS null: the series is stationary. Using both gives a cross-check.
# Levels trend upward for decades, so the test regression includes a constant
# and a linear trend ("ct"). First differencing removes the trend, so the
# differenced series only needs a constant ("c").
results = []
for col in df.columns:
    for form, series, reg in [("levels", df[col], "ct"),
                              ("differences", df[col].diff().dropna(), "c")]:
        adf_stat, adf_p, *_ = adfuller(series, regression=reg, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(series, regression=reg, nlags="auto")
        results.append({"series": col, "form": form,
                        "adf_stat": round(adf_stat, 2), "adf_p": round(adf_p, 3),
                        "kpss_stat": round(kpss_stat, 2), "kpss_p": round(kpss_p, 3)})

table = pd.DataFrame(results)
print(table)
table.to_csv(OUT_FILE, index=False)

# --- Zivot-Andrews: unit root test allowing one endogenous break ---------------
# Several differenced series failed to reject a unit root in ADF/KPSS. Perron
# (1989) shows a break in the mean can make a stationary series look
# non-stationary. Zivot-Andrews tests H0: unit root (no break) against
# H1: stationary with one break at an unknown date, searched over the same
# 15% trimmed window used by the QLR test later.
# Levels: break allowed in both intercept and trend ("ct").
# Differences: break allowed in the mean ("c"), i.e., a shift in the drift.
za_results = []
for col in df.columns:
    for form, series, reg in [("levels", df[col], "ct"),
                              ("differences", df[col].diff().dropna(), "c")]:
        za_stat, za_p, crit, lags, break_idx = zivot_andrews(
            series, trim=0.15, maxlag=4, regression=reg, autolag="AIC")
        za_results.append({"series": col, "form": form,
                           "za_stat": round(za_stat, 2), "za_p": round(za_p, 3),
                           "crit_5pct": round(crit["5%"], 2),
                           "break_year": series.index[break_idx]})

za_table = pd.DataFrame(za_results)
print(za_table)
za_table.to_csv(ZA_FILE, index=False)