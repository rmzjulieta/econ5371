"""
functions.py -- Lab 6 (Volatility Models: ARCH, GARCH, GJR-GARCH)

Two functions this lab, for the same reason as prior labs' functions.py
files: each does non-trivial formatting work, worth keeping in one
place rather than rewriting by hand.

New this lab: a QUICK SANITY CHECK using Python's assert statement,
placed right after each function definition. This is the smallest
version of a real professional habit -- after writing a function,
confirm it behaves correctly on a case where you already know the
right answer, before trusting it on real data.

An assert statement does nothing if its condition is True. If the
condition is False, Python raises an AssertionError immediately and
stops -- which is exactly what you want: a function that's silently
wrong is far worse than one that fails loudly the moment you test it.

This is not full software testing (that would be a tool like pytest,
run separately from your analysis script). It's the minimum version
of the same idea: check your own work before you rely on it.

lab_6.py imports both with:

    from functions import run_arch_lm_test, compare_garch_models
"""

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch


def run_arch_lm_test(returns, nlags=5, label=""):
    """Run the ARCH-LM test and print a formatted summary.

    Tests for ARCH effects (volatility clustering) in a return series.

    Parameters
    ----------
    returns : array-like
        The return series to test.
    nlags : int, default 5
        Number of lags to include in the auxiliary regression.
    label : str, default ""
        A short description, used in printed output.

    Returns
    -------
    dict
        Keys: "lm_stat", "lm_pvalue", "has_arch_effects" (bool).
    """
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns, nlags=nlags)
    has_arch_effects = lm_pvalue < 0.05

    print(f"\n{'-'*55}")
    print(f"ARCH-LM test{': ' + label if label else ''}")
    print(f"H0: no ARCH effects (returns show no volatility clustering)")
    print(f"{'-'*55}")
    print(f"{'LM statistic:':<20}{lm_stat:>10.4f}")
    print(f"{'p-value:':<20}{lm_pvalue:>10.6f}")
    print(f"{'-'*55}")
    verdict = "Reject H0 -- ARCH effects present" if has_arch_effects \
        else "Fail to reject H0 -- no evidence of ARCH effects"
    print(verdict)

    return {"lm_stat": lm_stat, "lm_pvalue": lm_pvalue, "has_arch_effects": has_arch_effects}


# --- Quick sanity check -----------------------------------------------------
# Two toy series with KNOWN properties: one built to have obvious
# volatility clustering (calm, then turbulent, then calm again), one
# built as plain white noise (no clustering by construction). If the
# function works correctly, it must tell these two cases apart.
_rng = np.random.default_rng(0)
_clustered = np.concatenate([
    _rng.normal(0, 0.5, 100),   # calm
    _rng.normal(0, 5.0, 100),   # turbulent
    _rng.normal(0, 0.5, 100),   # calm again
])
_white_noise = _rng.normal(0, 1.0, 300)

_clustered_result = run_arch_lm_test(_clustered, nlags=5, label="[sanity check: clustered]")
_noise_result = run_arch_lm_test(_white_noise, nlags=5, label="[sanity check: white noise]")

assert _clustered_result["has_arch_effects"], \
    "run_arch_lm_test failed to detect obvious volatility clustering"
assert not _noise_result["has_arch_effects"], \
    "run_arch_lm_test found ARCH effects in plain white noise"
print("\nrun_arch_lm_test: sanity checks passed.")


def compare_garch_models(fitted_models, names):
    """Print a side-by-side comparison of fitted volatility models.

    Parameters
    ----------
    fitted_models : list of arch.univariate.base.ARCHModelResult
        Fitted model results, e.g. from arch_model(...).fit().
    names : list of str
        Display name for each model, same order as fitted_models.

    Returns
    -------
    dict
        Keys: "best_model_name" (str, lowest AIC), "aic_values" (dict).
    """
    aic_values = {}

    print(f"\n{'-'*50}")
    print(f"{'Model':<20}{'AIC':>12}{'BIC':>12}")
    print(f"{'-'*50}")
    for name, fit in zip(names, fitted_models):
        print(f"{name:<20}{fit.aic:>12.3f}{fit.bic:>12.3f}")
        aic_values[name] = fit.aic
    print(f"{'-'*50}")

    best_model_name = min(aic_values, key=aic_values.get)
    print(f"Best model by AIC: {best_model_name}")

    return {"best_model_name": best_model_name, "aic_values": aic_values}


# --- Quick sanity check -----------------------------------------------------
# A tiny stand-in class with just an .aic and .bic attribute -- enough
# to test the comparison logic without needing to fit a real GARCH
# model here. Two fake "models" with known AIC values; the function
# must correctly identify which one is lower.
class _FakeFit:
    def __init__(self, aic, bic):
        self.aic = aic
        self.bic = bic

_fake_better = _FakeFit(aic=100.0, bic=110.0)
_fake_worse = _FakeFit(aic=150.0, bic=160.0)

_comparison = compare_garch_models([_fake_worse, _fake_better], ["Worse Model", "Better Model"])
assert _comparison["best_model_name"] == "Better Model", \
    "compare_garch_models did not correctly identify the lower-AIC model"
print("\ncompare_garch_models: sanity check passed.")