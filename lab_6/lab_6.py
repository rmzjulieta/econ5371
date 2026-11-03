"""
Lab 6: Volatility Models -- ARCH, GARCH, GJR-GARCH
=====================================================

Applies the ARCH-LM test, GARCH(1,1) estimation, and GJR-GARCH(1,1)
estimation to synthetic daily returns with genuine volatility
clustering and a genuine leverage effect (asymmetric response to
negative shocks).

Two functions used in this lab live in functions.py, in this same
folder -- each followed by a quick assert-based sanity check. See
that file's docstring for what this checks and why.

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
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf

from functions import run_arch_lm_test, compare_garch_models

pd.set_option('display.precision', 4)


# %% CONFIG -- everything you might want to change lives here

CONFIG = {
    "lab_folder": r"C:\Users\ramir\OneDrive\Economics Master Degree\econ5371\lab_6",

    # Raw GitHub URL for this lab's data. Replace with YOUR OWN repo's
    # raw URL once you've committed the CSV.
    "data": r"C:\Users\ramir\OneDrive\Economics Master Degree\econ5371\lab_6\returns_synthetic.csv",

    # Lags for the ARCH-LM test.
    "arch_lm_lags": 5,

    # GARCH mean specification: "Constant" is standard for daily
    # returns, which have a small, roughly constant average return.
    "mean_spec": "Constant",

    "color_returns": "tab:blue",
    "color_volatility": "tab:red",
}

os.chdir(CONFIG["lab_folder"])


# %% 1. Load the Data
#
# Wrapped in a try/except, same pattern as Lab 5: a wrong URL should
# produce a short, actionable message instead of a raw traceback.

try:
    df = pd.read_csv(CONFIG["data"], parse_dates=["date"])
    df = df.set_index("date")
    print(f"Loaded {len(df)} rows from {CONFIG['data']}")
except Exception as error:
    print("Could not load the data. This almost always means the URL")
    print("in CONFIG['data'] is wrong -- check that it:")
    print("  1. Points to YOUR OWN repo, not the instructor's")
    print("  2. Uses the raw.githubusercontent.com form")
    print("  3. Has no typos in the username, repo name, or file path")
    print(f"\nOriginal error: {error}")
    raise

returns = df["return"]

print(returns.describe())

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(returns.index, returns, color=CONFIG["color_returns"], linewidth=0.6)
ax.set_title("Daily Returns (%)")
ax.set_xlim(returns.index.min(), returns.index.max())
fig.tight_layout()
plt.show()

# Discussion: Do you see periods of calm alternating with periods of
# large swings? That pattern -- volatility clustering -- is what
# ARCH/GARCH models are built to capture. A plain ARMA model, which
# only models the MEAN, has no way to represent this at all.


# %% 2. Test for ARCH Effects

arch_lm_result = run_arch_lm_test(returns, nlags=CONFIG["arch_lm_lags"], label="daily returns")

# Discussion: The ARCH-LM test regresses squared returns on their own
# lags. Why would clustering in volatility show up as predictability
# in SQUARED returns, even though the returns themselves are close to
# unpredictable?


# %% 3. Fit GARCH(1,1)
#
# sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2
#
# GARCH(1,1) treats a positive and a negative shock of the same size
# as equally important for tomorrow's volatility -- it only sees the
# SQUARED shock, which erases the sign.

garch = arch_model(returns, mean=CONFIG["mean_spec"], vol="GARCH", p=1, q=1, dist="normal")
garch_fit = garch.fit(disp="off")

print(garch_fit.summary())

# Discussion: Look at alpha[1] + beta[1]. This sum measures how
# persistent volatility shocks are -- close to 1 means today's
# turbulence fades slowly. What does the estimated sum suggest here?


# %% 4. Fit GJR-GARCH(1,1)
#
# sigma_t^2 = omega + alpha*eps_{t-1}^2 + gamma*I(eps_{t-1}<0)*eps_{t-1}^2
#             + beta*sigma_{t-1}^2
#
# The new term, gamma, ONLY activates when yesterday's shock was
# negative (I(eps_{t-1}<0) = 1). This is the "leverage effect": bad
# news is allowed to raise volatility by more than equally-sized good
# news does.

gjr = arch_model(returns, mean=CONFIG["mean_spec"], vol="GARCH", p=1, o=1, q=1, dist="normal")
gjr_fit = gjr.fit(disp="off")

print(gjr_fit.summary())

# Discussion: Is gamma[1] statistically significant? If it is, what
# does that tell you about how this series responds to bad news
# versus good news of the same size -- something plain GARCH(1,1)
# cannot represent at all?


# %% 5. Compare the Two Models

comparison = compare_garch_models(
    [garch_fit, gjr_fit],
    ["GARCH(1,1)", "GJR-GARCH(1,1)"],
)

# Discussion: Which model does AIC favor? Does that match your
# conclusion about gamma[1]'s significance in Section 4? A model with
# one more parameter should only "win" if that parameter is doing
# real work -- is that the case here?


# %% 6. Conditional Volatility and Residual Diagnostics
#
# The fitted conditional volatility (sigma_t) is the model's own
# estimate of how turbulent each day was. Standardized residuals
# (return / sigma_t) should look like plain white noise -- if the
# model correctly captured the volatility pattern, dividing by its own
# volatility estimate should remove the clustering entirely.

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(returns.index, gjr_fit.conditional_volatility,
             color=CONFIG["color_volatility"], linewidth=1.0)
axes[0].set_title("GJR-GARCH(1,1) Conditional Volatility")

std_resid = gjr_fit.std_resid
axes[1].plot(returns.index, std_resid, color=CONFIG["color_returns"], linewidth=0.5)
axes[1].axhline(0, color="gray", linewidth=0.7, linestyle="--")
axes[1].set_title("Standardized Residuals")
fig.tight_layout()
plt.show()

std_resid_arch_check = run_arch_lm_test(
    std_resid.dropna(), nlags=CONFIG["arch_lm_lags"], label="standardized residuals"
)

# Discussion: Re-running the ARCH-LM test on the standardized
# residuals -- does it now fail to reject H0? What would it mean if
# ARCH effects were STILL present after fitting GJR-GARCH?


# %% 7. ACF of Squared Standardized Residuals
#
# A second, visual check of the same idea: if volatility clustering
# is fully captured, the ACF of squared standardized residuals should
# show no significant spikes.

fig, ax = plt.subplots(figsize=(8, 3.5))
plot_acf(std_resid.dropna() ** 2, lags=15, ax=ax)
ax.set_title("ACF of Squared Standardized Residuals")
fig.tight_layout()
plt.show()


# %% Lab Complete -- Summary

print("\n" + "=" * 65)
print("LAB 6 SUMMARY -- VOLATILITY MODELS")
print("=" * 65)

print(f"ARCH-LM test (raw returns):  p = {arch_lm_result['lm_pvalue']:.6f} "
      f"({'ARCH effects present' if arch_lm_result['has_arch_effects'] else 'no ARCH effects'})")

print(f"\nGARCH(1,1):     AIC = {garch_fit.aic:.3f}")
print(f"GJR-GARCH(1,1): AIC = {gjr_fit.aic:.3f}")
print(f"Best model by AIC: {comparison['best_model_name']}")

gamma_coef = gjr_fit.params.get("gamma[1]", None)
gamma_pvalue = gjr_fit.pvalues.get("gamma[1]", None)
if gamma_coef is not None:
    print(f"\nLeverage term (gamma): {gamma_coef:.4f}, p = {gamma_pvalue:.4f} "
          f"({'significant' if gamma_pvalue < 0.05 else 'not significant'})")

print(f"\nARCH-LM test (standardized residuals): p = {std_resid_arch_check['lm_pvalue']:.6f} "
      f"({'ARCH effects still present -- model incomplete' if std_resid_arch_check['has_arch_effects'] else 'no remaining ARCH effects -- model captured the clustering'})")

print("=" * 65)
print("BEFORE YOU LEAVE:")
print("  1. Save this script")
print("  2. git add . / git commit -m \"Lab 6: ARCH/GARCH/GJR-GARCH\" / git push")
print("  3. Update README.md if you added new files or changed structure")
print("=" * 65)