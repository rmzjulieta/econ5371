"""
Lab 4: Vector Autoregression
==============================

Applies the full VAR workflow -- estimation, lag selection, stability,
Granger causality, impulse response functions, and forecast error
variance decomposition -- to a synthetic 3-variable macro system.

Two functions used in this lab live in functions.py, in this same
folder: run_granger_causality_grid() and check_var_stability(). See
that file's docstring for why they're separated out.

New this lab: timing around the steps that do meaningfully more
computation than a single model fit (lag selection, IRF, FEVD), and a
single consolidated summary at the end that pulls together the key
result from every section -- so you don't have to scroll back through
the whole script to see what this lab concluded.

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

# Set the lab folder location.
LAB_FOLDER = r"C:\Users\ramir\OneDrive\Economics Master Degree\4.Fall 2026\Week 9"

import os

os.chdir(LAB_FOLDER)

# Import packages, including our own functions.py.
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

from functions import run_granger_causality_grid, check_var_stability

pd.set_option('display.precision', 2)

tab10 = plt.get_cmap("tab10")
COLORS = {"output_gap": tab10(0), "inflation": tab10(1), "policy_rate": tab10(2)}

# Start a timer for the whole script. time.time() returns the current
# moment as a single number (seconds since a fixed reference point) --
# not useful by itself, but subtracting an earlier reading from a later
# one gives elapsed time. We do this once here, and again around any
# specific step worth timing individually below.
script_start = time.time()   #where is taking more time


# %% 1. Load and Visualize the Data

df = pd.read_csv("macro_var_synthetic.csv", parse_dates=["date"])
df = df.set_index("date")
df.index.freq = "QS"

variables = ["output_gap", "inflation", "policy_rate"]

print(f"{'='*50}")
print(f"Macro VAR System -- Loaded Series")
print(f"{'='*50}")
print(f"{'Observations:':<20}{len(df):>10}")
print(f"{'Date range:':<20}{str(df.index.min().date()):>10} to {df.index.max().date()}")
print(df[variables].describe().round(2))
print(f"{'='*50}")
print("\nNote on output_gap: this series is simulated directly as a")
print("single gap variable, not derived from separate real-output and")
print("potential-output series. A positive shock therefore means the")
print("gap widened -- it does not tell us, and this model cannot tell")
print("us, whether that came from real output rising or potential")
print("output falling. Separating those two forces would require")
print("modeling them as distinct series, which this VAR does not do.")

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, col in zip(axes, variables):
    ax.plot(df.index, df[col], color=COLORS[col], linewidth=1.2)
    ax.set_title(col)
fig.tight_layout()
plt.show()

# Discussion: Do these three series look stationary? A VAR requires
# stationary inputs -- we confirm this formally next.


# %% 2. Confirm Stationarity
# ============================================================================
#
# A VAR estimated in levels on non-stationary (I(1)) variables produces
# unreliable, spurious-looking results. Before fitting anything, check
# each variable individually with an ADF test.

print(f"\n{'-'*50}")
print(f"ADF tests -- H0: unit root (non-stationary)")
print(f"{'-'*50}")
adf_results = {}
for col in variables:
    result = adfuller(df[col], autolag="AIC")
    verdict = "stationary" if result[1] < 0.05 else "NOT stationary -- do not proceed"
    print(f"{col:<15} statistic={result[0]:>8.4f}  p-value={result[1]:.4f}  ({verdict})")
    adf_results[col] = {"statistic": result[0], "p_value": result[1]}
print(f"{'-'*50}")

# Discussion: All three series should reject the unit-root null. If any
# did not, what would be the right next step before estimating a VAR?


# %% 3. Lag Order Selection
# ============================================================================
#
# A VAR needs a lag length p, just like an AR model does -- but getting
# p wrong matters more here, since every additional lag adds k^2 new
# parameters (k = number of variables), not just one.
#
# select_order() fits a separate VAR for every candidate lag length
# (1 through maxlags) to compare them -- meaningfully more computation
# than fitting a single model, and a reasonable place to time.

model = VAR(df[variables])

lag_selection_start = time.time()
lag_selection = model.select_order(maxlags=8)
lag_selection_elapsed = time.time() - lag_selection_start

print(lag_selection.summary())
print(f"\nLag selection took {lag_selection_elapsed:.3f} seconds "
      f"(8 candidate VAR specifications fit).")

# Discussion: Do AIC, BIC, FPE, and HQIC agree on the best lag length?
# If they don't agree, which would you lean on, and why? (Recall this
# same question from Lab 2's ARIMA order comparison.)


# %% 4. Estimate the VAR and Check Stability
# ============================================================================
#
chosen_lag = 2
fit = model.fit(chosen_lag)

print(fit.summary())

stability = check_var_stability(fit.coefs)

# Discussion: The summary above reports one equation per variable --
# each variable regressed on lags of ALL THREE variables. Why does a
# VAR estimate a full system like this, rather than one equation at a
# time in isolation?


# %% 4b. Residual Diagnostics
#
# A well-specified VAR should leave behind residuals that look like
# white noise in EVERY equation -- no leftover autocorrelation in any
# of the three. If one equation's residuals still show a pattern, that
# equation's lag structure likely needs revisiting, the same way we
# checked residuals after fitting SARIMA in Lab 1.

print(f"\n{'-'*60}")
print(f"Ljung-Box test on residuals, by equation")
print(f"H0: residuals are not autocorrelated (i.e., they look like white noise)")
print(f"{'-'*60}")
ljung_box_results = {}
for col in variables:
    lb = acorr_ljungbox(fit.resid[col], lags=[4, 8], return_df=True)  #join errors
    ljung_box_results[col] = lb
    p4, p8 = lb.loc[4, "lb_pvalue"], lb.loc[8, "lb_pvalue"]
    verdict = "OK -- no evidence of leftover autocorrelation" if min(p4, p8) > 0.05 \
        else "CHECK -- possible leftover autocorrelation"
    print(f"{col:<15} p(lag 4) = {p4:.4f}   p(lag 8) = {p8:.4f}   ({verdict})")
print(f"{'-'*60}")

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, col in zip(axes, variables):
    ax.plot(fit.resid.index, fit.resid[col], color=COLORS[col], linewidth=1.0)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"{col} -- residuals")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))  #plot autocorrelation for the residuals
for ax, col in zip(axes, variables):
    plot_acf(fit.resid[col], lags=8, ax=ax)
    ax.set_title(col)
fig.tight_layout()
plt.show()

# Discussion: Do any of the three equations show p-values below 0.05,
# or ACF spikes outside the confidence band? If one equation's
# residuals looked problematic while the others were clean, what would
# that suggest about where the model's lag structure might be missing
# something -- and would you look for the fix in that one equation, or
# reconsider the whole system?


# %% 5. Granger Causality
#
# Does the past of one variable improve our ability to predict another,
# beyond what that variable's own past already tells us? This is
# Granger causality -- a statement about predictive content, not
# necessarily about true structural cause and effect.

granger_results = run_granger_causality_grid(fit, variables)

# Discussion: Which pairs show significant Granger causality? Does the
# pattern you see form a directional chain (A -> B -> C), or something
# more tangled? What would a directional chain suggest about how this
# system actually works?


# %% 6. Impulse Response Functions
#
# An IRF traces out how a one-time shock to one variable propagates
# through the whole system over time. The ordering of variables matters
# here -- we use the order already set above (output_gap, inflation,
# policy_rate), which determines what "a shock to output_gap" is
# allowed to affect contemporaneously (everything after it in the
# ordering) versus only with a lag (everything before it).

irf_start = time.time()
irf = fit.irf(periods=12)
irf_elapsed = time.time() - irf_start

fig = irf.plot(orth=True, figsize=(10, 8))
plt.tight_layout()
plt.show()

print(f"\nIRF computation took {irf_elapsed:.3f} seconds.")
print("Reading the grid: each panel title reads 'shock -> response' --")
print("column position is the variable shocked, row position is the")
print("variable responding. E.g., the middle panel of the top row,")
print("'inflation -> output_gap', shows how output_gap responds to a")
print("shock in inflation -- NOT how inflation itself behaves.")

# irf.irfs has shape (periods+1, response_variable, shock_variable).
# For each shock, find the peak (largest absolute) response of EVERY
# variable -- including the shocked variable's own response to itself
# -- along with the horizon at which that peak occurs.
irf_peaks = {}   # for each variable where the large impact take place
for shock_idx, shock_var in enumerate(variables):
    irf_peaks[shock_var] = {}
    for resp_idx, resp_var in enumerate(variables):
        response = irf.irfs[:, resp_idx, shock_idx]
        peak_horizon = np.argmax(np.abs(response))
        irf_peaks[shock_var][resp_var] = {
            "value": response[peak_horizon],
            "horizon": peak_horizon,
        }

print(f"\nPeak response of each variable to each shock "
      f"(horizon 0-{irf.irfs.shape[0]-1}). Each cell shows the largest")
print("absolute response, with the horizon at which it occurs in")
print("parentheses -- e.g. '0.75 (h=7)' means the response peaks at")
print("0.75, seven periods after the shock.")

name_w = 14
cell_w = 16
header = f"{'Shock to':<{name_w}}" + "".join(f"{v:<{cell_w}}" for v in variables)
rule = "-" * (name_w + cell_w * len(variables))
print(rule)
print(header)
print(rule)
for shock_var in variables:
    row = f"{shock_var:<{name_w}}"
    for resp_var in variables:
        peak = irf_peaks[shock_var][resp_var]
        cell = f"{peak['value']:+.4f} (h={peak['horizon']})"
        row += f"{cell:<{cell_w}}"
    print(row)
print(rule)
print("Note: the diagonal (a variable's response to its own shock) is")
print("1.0000 at horizon 0 by construction -- that is how a Cholesky-")
print("orthogonalized one-unit shock is defined, not a finding.")

# Discussion: Look at the response of inflation to an output_gap shock.
# Is the contemporaneous (period 0) response zero? Why, given how the
# variables are ordered? Does the peak response occur immediately, or
# after several periods?


# %% 7. Forecast Error Variance Decomposition (FEVD)
#
# FEVD asks a different question than the IRF: not "what happens after
# a shock," but "of all the uncertainty in forecasting each variable at
# a given horizon, what share comes from each source of shock?"

fevd_start = time.time()
fevd = fit.fevd(12)
fevd_elapsed = time.time() - fevd_start

fig = fevd.plot(figsize=(10, 8))
plt.tight_layout()
plt.show()

fevd.summary()
print(f"\nFEVD computation took {fevd_elapsed:.3f} seconds.")

# fevd.decomp has shape (response_variable, horizon, shock_variable).
# Pull the longest-horizon share explained by each OTHER variable's
# shocks -- the "how much does this variable end up depending on
# something else" number.
fevd_long_horizon = {}
last_h = fevd.decomp.shape[1] - 1
for resp_idx, resp_var in enumerate(variables):
    shares = {shock_var: fevd.decomp[resp_idx, last_h, shock_idx]
              for shock_idx, shock_var in enumerate(variables)}
    fevd_long_horizon[resp_var] = shares

# Discussion: At short horizons, does each variable's forecast error
# come mostly from its own shocks? At longer horizons (look at horizon
# 12), how has that changed for inflation and policy_rate specifically
# -- and does that match the causal chain you found in Section 5?


# %% Lab Complete -- Summary

script_elapsed = time.time() - script_start

# Pull together, in one place, the single most useful number from each
# section above -- so anyone (including you, re-reading this later)
# can see the whole lab's conclusions without scrolling back through
# every cell's output.

print("\n" + "=" * 65)
print("LAB 4 SUMMARY -- VECTOR AUTOREGRESSION")
print("=" * 65)

print("Stationarity (ADF, H0: unit root):")
for col in variables:
    p = adf_results[col]["p_value"]
    print(f"  {col:<15} p = {p:.4f}  ({'stationary' if p < 0.05 else 'NOT stationary'})")

print(f"\nChosen lag order: {chosen_lag}")
print(f"VAR stability:    max eigenvalue modulus = {stability['max_modulus']:.4f} "
      f"({'stable' if stability['is_stable'] else 'UNSTABLE'})")

print("\nGranger causality (pairs with evidence of causation, p < 0.05):")
causal_pairs = [(pair, r) for pair, r in granger_results.items() if r["p_value"] < 0.05]
if causal_pairs:
    for (cause, effect), r in causal_pairs:
        print(f"  {cause} -> {effect}  (F = {r['F_stat']:.2f}, p = {r['p_value']:.4f})")
else:
    print("  None found.")

print(f"\nIRF -- strongest cross-variable response to each shock "
      f"(horizon 0-{irf.irfs.shape[0]-1}; full grid in Section 6):")
for shock_var in variables:
    cross = {v: irf_peaks[shock_var][v] for v in variables if v != shock_var}
    strongest_var = max(cross, key=lambda v: abs(cross[v]["value"]))
    strongest = cross[strongest_var]
    print(f"  {shock_var:<12} -> {strongest_var:<12} "
          f"{strongest['value']:+.4f} (h={strongest['horizon']})")

print(f"\nFEVD -- share of each variable's forecast error variance at "
      f"horizon {last_h}, by source:")
for resp_var, shares in fevd_long_horizon.items():
    share_str = ", ".join(f"{v}={s:.1%}" for v, s in shares.items())
    print(f"  {resp_var:<12} {share_str}")

print("\nComputation time:")
print(f"  Lag selection (8 specifications): {lag_selection_elapsed:.3f} sec")
print(f"  IRF:                              {irf_elapsed:.3f} sec")
print(f"  FEVD:                             {fevd_elapsed:.3f} sec")
print(f"  Total script runtime:             {script_elapsed:.3f} sec")

print("=" * 65)
print("BEFORE YOU LEAVE:")
print("  1. Save this script")
print("  2. git add . / git commit -m \"Lab 4: vector autoregression\" / git push")
print("  3. Update README.md if you added new files or changed structure")
print("=" * 65)