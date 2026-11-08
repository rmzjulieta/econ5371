"""
07_forecast.py

Purpose : Test whether detecting the 1998 structural break improves forecasts
          of the aggregate (16+) female LFPR. Two random walk with drift
          forecasts are compared over 2015-2025, both estimated only with data
          through 2014:
            (a) full-sample drift: the average annual change, 1949-2014
            (b) post-break drift:  the average annual change after the break
          Accuracy is compared with RMSE and the Diebold-Mariano (DM) test,
          for all years and excluding the COVID years (2020-2021).
Input   : data/clean/lfpr_women_by_age.csv
Output  : output/table7_forecast_evaluation.csv
          output/forecasts_2015_2025.csv
          output/fig3_forecasts.png
          output/fig4_cumulative_loss.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from functions import run_qlr_test, dm_test

# --- Paths -----------------------------------------------------------------
# Same convention as the other scripts: paths are built from this script's
# location, so the code runs on any computer without editing paths.
PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_DIR / "data" / "clean" / "lfpr_women_by_age.csv"
TABLE_FILE = PROJECT_DIR / "output" / "table7_forecast_evaluation.csv"
FORECAST_FILE = PROJECT_DIR / "output" / "forecasts_2015_2025.csv"
FIG_FC_FILE = PROJECT_DIR / "output" / "fig3_forecasts.png"
FIG_LOSS_FILE = PROJECT_DIR / "output" / "fig4_cumulative_loss.png"

# --- Settings ----------------------------------------------------------------
SERIES = "age_16plus"        # the aggregate series, as the professor suggested
LAST_TRAIN_YEAR = 2014       # estimation sample ends here
TEST_YEARS = list(range(2015, 2026))
COVID_YEARS = (2020, 2021)   # transitory shock, excluded in the robustness check

df = pd.read_csv(CLEAN_FILE, index_col="year")
y = df[SERIES]

# Annual changes (first differences). In a random walk with drift,
# dy_t = mu + e_t, so the drift mu is simply the mean of the differences.
dy = y.diff().dropna()
dy_train = dy.loc[:LAST_TRAIN_YEAR]

# --- 1. Re-detect the break using only the training sample -------------------
# The 1998 break in the paper was found with data through 2025. Using it to
# "forecast" 2015-2025 would use information from the future (look-ahead
# bias). Re-running the QLR test on data through 2014 checks that the break
# date was already detectable at the forecast origin.
qlr = run_qlr_test(dy_train.values)
break_year = dy_train.index[qlr["break_index"]]
print(f"\nBreak year detected with data through {LAST_TRAIN_YEAR}: {break_year}")

# --- 2. Estimate the two drifts ------------------------------------------------
# (a) ignores the break: one average change for the whole training sample.
# (b) uses the break: the average change only after the break year.
mu_full = dy_train.mean()
mu_post = dy_train.loc[break_year:].mean()
print(f"{'Full-sample drift (a):':<26}{mu_full:>8.3f} pp per year")
print(f"{'Post-break drift (b):':<26}{mu_post:>8.3f} pp per year")

# --- 3. One-step-ahead forecasts, 2015-2025 -----------------------------------
# Each year's forecast starts from the ACTUAL value of the previous year and
# adds the drift: y_hat_t = y_{t-1} + mu. One-step-ahead (h = 1) keeps the DM
# test simple, because the loss differential is then not autocorrelated.
# Both forecasts start from the same y_{t-1}; they differ only in the drift.
fc_full = []
fc_post = []
for t in TEST_YEARS:
    y_prev = y.loc[t - 1]
    fc_full.append(y_prev + mu_full)
    fc_post.append(y_prev + mu_post)

actual = y.loc[TEST_YEARS].values
fc_full = np.array(fc_full)
fc_post = np.array(fc_post)

# Forecast errors: actual minus forecast (negative = over-prediction).
e_full = actual - fc_full
e_post = actual - fc_post

forecasts = pd.DataFrame({"year": TEST_YEARS, "actual": actual,
                          "forecast_full_drift": fc_full.round(3),
                          "forecast_post_break_drift": fc_post.round(3),
                          "error_full_drift": e_full.round(3),
                          "error_post_break_drift": e_post.round(3)})
print(forecasts.to_string(index=False))
forecasts.to_csv(FORECAST_FILE, index=False)

# --- 4. Evaluate: RMSE and DM test, all years and excluding COVID --------------
# COVID is a one-time shock, not a structural break. A single very large
# error can dominate both the RMSE and the DM statistic, so the comparison is
# repeated without 2020-2021 as a robustness check.
# DM convention (course slides): model 1 = full drift, model 2 = post-break
# drift; d_bar > 0 means the post-break forecast is more accurate.
no_covid = np.array([t not in COVID_YEARS for t in TEST_YEARS])
samples = {"All years (2015-2025)": np.ones(len(TEST_YEARS), dtype=bool),
           "Excluding 2020-2021": no_covid}

rows = []
for label, mask in samples.items():
    ea, eb = e_full[mask], e_post[mask]
    dm_hln = dm_test(ea, eb, h=1, small_sample=True)
    dm_norm = dm_test(ea, eb, h=1, small_sample=False)
    rows.append({"Evaluation sample": label, "N": int(mask.sum()),
                 "RMSE full-sample drift": round(np.sqrt(np.mean(ea**2)), 3),
                 "RMSE post-break drift": round(np.sqrt(np.mean(eb**2)), 3),
                 "Mean loss differential (d_bar)": round(dm_hln["d_bar"], 3),
                 "DM normal": round(dm_norm["dm_stat"], 2),
                 "p-value normal": round(dm_norm["p_value"], 3),
                 "DM HLN": round(dm_hln["dm_stat"], 2),
                 "p-value HLN": round(dm_hln["p_value"], 3)})

table = pd.DataFrame(rows)
print(f"\n{'='*72}")
print("Forecast evaluation -- random walk with drift, aggregate 16+")
print(f"{'='*72}")
print(table.to_string(index=False))
table.to_csv(TABLE_FILE, index=False)

# --- 5. Figure: actual values and both forecasts --------------------------------
# A few years before 2015 are shown for context, so the reader can see where
# the forecasts start from.
fig, ax = plt.subplots(figsize=(9, 4.5))
context = y.loc[2005:]
ax.plot(context.index, context.values, color="black", linewidth=1.5,
        label="Actual LFPR")
ax.plot(TEST_YEARS, fc_full, marker="o", linestyle="--", color="tab:blue",
        label=f"(a) Full-sample drift ({mu_full:+.2f} pp/yr)")
ax.plot(TEST_YEARS, fc_post, marker="s", linestyle="--", color="tab:orange",
        label=f"(b) Post-{break_year} drift ({mu_post:+.2f} pp/yr)")
ax.axvline(LAST_TRAIN_YEAR + 0.5, color="gray", linestyle=":", linewidth=1)
ax.text(LAST_TRAIN_YEAR + 0.7, ax.get_ylim()[1] - 0.2, "Forecast period",
        color="gray", fontsize=9, va="top")
ax.set_title("One-step-ahead forecasts of female LFPR, 16+ (2015-2025)")
ax.set_xlabel("Year")
ax.set_ylabel("Participation rate (%)")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # whole years only
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_FC_FILE, dpi=300)
plt.show()

# --- 6. Figure: cumulative loss differential (as in the course slides) -----------
# Cumulative sum of d_t = e_full^2 - e_post^2. When the line rises, the
# post-break forecast is gaining accuracy; when it falls, the full-sample
# forecast is. A single jump shows that one year drives the result.
d = e_full**2 - e_post**2
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(TEST_YEARS, np.cumsum(d), marker="o", color="tab:purple")
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.axvspan(COVID_YEARS[0] - 0.5, COVID_YEARS[1] + 0.5, color="gray", alpha=0.15,
           label="COVID years")
ax.set_title("Cumulative loss differential: full-sample minus post-break "
             "squared error")
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative loss differential")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_LOSS_FILE, dpi=300)
plt.show()