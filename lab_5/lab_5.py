"""
Lab 5: Vector Error Correction Model (VECM)
==============================================

Applies the Johansen cointegration test and VECM estimation to a
synthetic short-rate / long-rate pair -- a classic cointegrated system,
matching the textbook's own yield-curve example.

The Johansen test formatter lives in functions.py, in this same
folder. See that file's docstring for why.

Two new practices this lab:

  1. A CONFIG block, right after imports, collecting every value you
     might reasonably want to change -- file path, test settings,
     lag order, plotting options -- in one place at the top of the
     script. Compare this to earlier labs, where a setting like
     chosen_lag = 2 was buried inside a section halfway through the
     file. Read the CONFIG section before anything else; you should
     be able to adjust this whole lab's behavior without touching any
     code below it.

  2. ERROR HANDLING around the data load. Every lab so far has read
     data from a raw GitHub URL -- and every lab has had at least one
     moment where a wrong URL produced a long, unfriendly traceback.
     Section 1 below wraps that load in a try/except block that
     catches the likely failure and prints a short, actionable
     message instead.

Run this script section by section (recommended: in Positron, run cell
by cell using the '# %%' markers), or top to bottom as a single script.
"""

# %%
# ============================================================================
# Environment activation
#
# 1. Activate it (needed every time you start a new terminal session):
#    conda activate econ5371
#
# 2. Install required packages/dependencies
#    pip install -r requirements.txt
#
# ============================================================================

import os

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

pd.set_option('display.precision', 2)


# %% CONFIG -- everything you might want to change lives here
#
# Adjust any of these and re-run the script -- nothing below this cell
# should need editing to change the lab's behavior.

CONFIG = {
    # Where this lab's folder lives on your machine.
    "lab_folder": r"\Users\ncachanosky\OneDrive\Research\GitHub\ECON-5371-lab\lab_5",

    # Where is the data located
    "data": r"\Users\ncachanosky\OneDrive\Research\GitHub\ECON-5371-lab\lab_5\yield_curve_synthetic.csv",

    # The two variables this lab analyzes, and the order they're
    # analyzed in (this order matters for how the cointegrating vector
    # is normalized).
    "variables": ["short_rate", "long_rate"],

    # Johansen test settings. det_order=0 means a constant is
    # restricted to the cointegrating relation -- appropriate for a
    # pair like interest rates with a stable long-run spread, not a
    # deterministic trend of their own.
    "johansen_det_order": 0,
    "johansen_k_ar_diff": 1,

    # VECM lag order (number of lagged DIFFERENCES -- one less than
    # the corresponding VAR lag order in levels).
    "vecm_k_ar_diff": 1,

    # Cointegrating rank to use when estimating the VECM. Set to None
    # to use whatever the Johansen test suggests automatically;
    # override with an integer (e.g., 1) to set it manually after
    # reviewing the Johansen output yourself.
    "coint_rank_override": None,

    # Lags to check in the residual Ljung-Box diagnostics.
    "ljung_box_lags": [4, 8],

    # Plot colors.
    "color_short": "tab:blue",
    "color_long": "tab:orange",
    "color_spread": "tab:green",
}

os.chdir(CONFIG["lab_folder"])
from functions import run_johansen_test

# %% 1. Load the Data
#
# Wrapped in a try/except: if the URL is wrong (a placeholder never
# swapped out, a typo, a repo that isn't public yet), pd.read_csv
# raises an error with a long, not-very-helpful traceback. We catch
# that here and print a short message pointing at the actual fix.

try:
    df = pd.read_csv(CONFIG["data"], parse_dates=["date"])
    df = df.set_index("date")
    df.index.freq = "MS"
    print(f"Loaded {len(df)} rows from {CONFIG['data']}")
except Exception as error:
    print("Could not load the data. This almost always means the URL")
    print("in CONFIG['data'] is wrong -- check that it:")
    print("  1. Points to YOUR OWN repo, not the instructor's")
    print("  2. Uses the raw.githubusercontent.com form (not github.com/.../blob/...)")
    print("  3. Has no typos in the username, repo name, or file path")
    print(f"\nOriginal error: {error}")
    raise

variables = CONFIG["variables"]

print(df[variables].describe().round(2))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df[variables[0]], color=CONFIG["color_short"], label=variables[0])
ax.plot(df.index, df[variables[1]], color=CONFIG["color_long"], label=variables[1])
ax.legend()
ax.set_title("Short and Long Rate")
fig.tight_layout()
plt.show()

# Discussion: Do these two series appear to move together over the
# long run, even while each wanders on its own in the short run? That
# "moving together in the long run" pattern is the visual signature of
# cointegration -- we confirm it formally below.


# %% 2. Confirm Each Series Is Individually I(1)
#
# Cointegration is only a meaningful question if each variable is
# individually non-stationary. If either series were already
# stationary on its own, there would be nothing for a long-run
# equilibrium relationship to explain.

print(f"\n{'-'*50}")
print(f"ADF tests -- H0: unit root (non-stationary)")
print(f"{'-'*50}")
for col in variables:
    result = adfuller(df[col], autolag="AIC")
    verdict = "unit root (expected)" if result[1] >= 0.05 else "stationary (unexpected)"
    print(f"{col:<15} statistic={result[0]:>8.4f}  p-value={result[1]:.4f}  ({verdict})")
print(f"{'-'*50}")

# Discussion: If a variable failed this test (came back stationary on
# its own), would a VECM still be the right tool? Why or why not?


# %% 3. Johansen Cointegration Test

data = df[variables].values
johansen = run_johansen_test(
    data,
    variables,
    det_order=CONFIG["johansen_det_order"],
    k_ar_diff=CONFIG["johansen_k_ar_diff"],
)

coint_rank = CONFIG["coint_rank_override"] or johansen["suggested_rank"]
print(f"\nUsing cointegrating rank = {coint_rank} "
      f"({'from CONFIG override' if CONFIG['coint_rank_override'] else 'from Johansen suggestion'}).")

# Discussion: Do the trace and max-eigenvalue tests agree on the rank?
# Does a rank of 1 make economic sense for a short-rate/long-rate pair
# -- i.e., can you think of a reason these two rates would share
# exactly one long-run equilibrium relationship?


# %% 4. Estimate the VECM

model = VECM(
    df[variables],
    k_ar_diff=CONFIG["vecm_k_ar_diff"],
    coint_rank=coint_rank,
    deterministic="ci",
)
fit = model.fit()

print(fit.summary())

# Discussion: Look at the "Loading coefficients (alpha)" for each
# equation. Which variable's alpha is statistically significant? What
# does a significant, negative alpha mean about how that variable
# behaves when the spread drifts away from its long-run value?


# %% 5. Residual Diagnostics
#
# As with every model we've fit this semester, check that what's left
# over after fitting looks like white noise.

print(f"\n{'-'*60}")
print(f"Ljung-Box test on residuals, by equation")
print(f"H0: residuals are not autocorrelated (i.e., they look like white noise)")
print(f"{'-'*60}")
resid = pd.DataFrame(fit.resid, columns=variables, index=df.index[-fit.resid.shape[0]:])
for col in variables:
    lb = acorr_ljungbox(resid[col], lags=CONFIG["ljung_box_lags"], return_df=True)
    p_values = lb["lb_pvalue"].values
    verdict = "OK" if min(p_values) > 0.05 else "CHECK -- possible leftover autocorrelation"
    p_str = ", ".join(f"p(lag {lag})={p:.4f}" for lag, p in zip(CONFIG["ljung_box_lags"], p_values))
    print(f"{col:<15} {p_str}   ({verdict})")
print(f"{'-'*60}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, col in zip(axes, variables):
    plot_acf(resid[col], lags=8, ax=ax)
    ax.set_title(f"ACF of residuals -- {col}")
fig.tight_layout()
plt.show()


# %% 6. The Cointegrating Relationship and the Spread
#
# The cointegrating vector beta tells us the specific combination of
# variables that is stationary. For a 2-variable system normalized on
# the first variable, this is easiest to see as a "spread" you can
# plot directly.

spread = df[variables[0]] - df[variables[1]]

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(df.index, spread, color=CONFIG["color_spread"])
ax.axhline(spread.mean(), color="gray", linestyle="--", linewidth=1)
ax.set_title(f"Spread: {variables[0]} - {variables[1]}")
fig.tight_layout()
plt.show()

spread_adf = adfuller(spread, autolag="AIC")
print(f"\nADF test on the spread directly:")
print(f"  statistic = {spread_adf[0]:.4f}, p-value = {spread_adf[1]:.4f}")
print(f"  {'Stationary -- consistent with cointegration.' if spread_adf[1] < 0.05 else 'NOT stationary -- inconsistent with rank 1 cointegration.'}")

# Discussion: Does the spread wander off permanently, or does it keep
# returning to roughly the same level? How does this picture relate to
# the significant alpha coefficient you found in Section 4?


# %% Lab Complete -- Summary

print("\n" + "=" * 65)
print("LAB 5 SUMMARY -- VECTOR ERROR CORRECTION MODEL")
print("=" * 65)

print("Individual unit root tests (ADF, H0: unit root):")
for col in variables:
    result = adfuller(df[col], autolag="AIC")
    print(f"  {col:<15} p = {result[1]:.4f}")

print(f"\nJohansen suggested rank: {johansen['suggested_rank']}")
print(f"Rank used for VECM:      {coint_rank}")

print(f"\nCointegrating vector (normalized on {variables[0]}):")
beta = fit.beta[:, 0]
for var, b in zip(variables, beta):
    print(f"  {var:<15} {b:+.4f}")

print("\nSpeed-of-adjustment (alpha) coefficients:")
for i, var in enumerate(variables):
    a = fit.alpha[i, 0]
    print(f"  {var:<15} {a:+.4f}")

print(f"\nSpread ADF test: p = {spread_adf[1]:.4f} "
      f"({'stationary' if spread_adf[1] < 0.05 else 'NOT stationary'})")

print("=" * 65)
print("BEFORE YOU LEAVE:")
print("  1. Save this script")
print("  2. git add . / git commit -m \"Lab 5: VECM\" / git push")
print("  3. Update README.md if you added new files or changed structure")
print("=" * 65)