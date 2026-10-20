"""
functions.py -- Lab 4 (Vector Autoregression)

Two functions this lab, for the same reason as Lab 3: each is called
more than once with identical formatting, so wrapping them avoids
repeating that formatting logic by hand.

  - run_granger_causality_grid(): this lab tests SIX directional pairs
    (does A Granger-cause B, for every ordered pair among three
    variables) -- a strong case for a function, since copy-pasting the
    same test-and-print block six times would be six chances to
    introduce a typo or inconsistent formatting.

  - check_var_stability(): computes the companion-matrix eigenvalues
    used to confirm a fitted VAR is stable. You will likely want this
    exact check again in a future lab (VECM, in Lab 5, relies on a
    similar stability idea) -- a good candidate for a function you
    keep around, not just one used twice in a single script.

lab_4.py imports both with:

    from functions import run_granger_causality_grid, check_var_stability
"""

import numpy as np
import matplotlib.pyplot as plt


def run_granger_causality_grid(fit, variables):
    """Run Granger causality tests for every ordered pair of variables.

    For each ordered pair (cause, effect), tests whether `cause`'s
    lags help predict `effect`, beyond what `effect`'s own lags
    already explain.

    Parameters
    ----------
    fit : statsmodels VARResults
        A fitted VAR model (the result of VAR(df).fit(p)).
    variables : list of str
        Variable names, in the order you want them tested. Every
        ordered pair (excluding a variable against itself) is tested.

    Returns
    -------
    dict
        Keys are (cause, effect) tuples; values are dicts with
        "F_stat" and "p_value".
    """
    results = {}

    print(f"\n{'-'*60}")
    print(f"{'Cause':>12} -> {'Effect':<12}{'F-stat':>12}{'p-value':>12}  Result")
    print(f"{'-'*60}")

    for cause in variables:
        for effect in variables:
            if cause == effect:
                continue
            test = fit.test_causality(effect, [cause], kind="f")
            verdict = "CAUSES" if test.pvalue < 0.05 else "no evidence"
            print(f"{cause:>12} -> {effect:<12}{test.test_statistic:>12.4f}"
                  f"{test.pvalue:>12.4f}  {verdict}")
            results[(cause, effect)] = {
                "F_stat": test.test_statistic,
                "p_value": test.pvalue,
            }

    print(f"{'-'*60}")

    return results


def check_var_stability(coef_matrices):
    """Check VAR stability via the companion matrix's eigenvalues.

    A VAR is stable (and therefore appropriate to estimate on
    stationary data without further transformation) if every
    eigenvalue of its companion matrix has modulus strictly less
    than 1 -- equivalently, every eigenvalue lies strictly inside the
    unit circle when plotted on the complex plane.

    This prints every eigenvalue found, not just the largest, and
    plots them against the unit circle so you can see at a glance how
    close (or not) the system is to instability.

    Parameters
    ----------
    coef_matrices : list of np.ndarray
        The VAR's lag coefficient matrices [A1, A2, ..., Ap], each of
        shape (k, k), in lag order.

    Returns
    -------
    dict
        Keys: "eigenvalues" (all roots found), "max_modulus" (largest
        eigenvalue modulus), "is_stable" (bool).
    """
    k = coef_matrices[0].shape[0]
    p = len(coef_matrices)

    companion = np.zeros((p * k, p * k))
    for i, A in enumerate(coef_matrices):
        companion[:k, i * k:(i + 1) * k] = A
    companion[k:, :(p - 1) * k] = np.eye((p - 1) * k)

    eigenvalues = np.linalg.eigvals(companion)
    moduli = np.abs(eigenvalues)
    max_modulus = np.max(moduli)
    is_stable = max_modulus < 1

    print(f"\n{'-'*50}")
    print(f"VAR Stability Check -- All Eigenvalues")
    print(f"{'-'*50}")
    print(f"{'#':<4}{'Real':>12}{'Imaginary':>12}{'Modulus':>12}")
    for i, (ev, mod) in enumerate(zip(eigenvalues, moduli), start=1):
        print(f"{i:<4}{ev.real:>12.4f}{ev.imag:>12.4f}{mod:>12.4f}")
    print(f"{'-'*50}")
    print(f"{'Largest modulus:':<28}{max_modulus:>10.4f}")
    print(f"{'Stable?':<28}{'Yes' if is_stable else 'No':>10}")
    print(f"{'-'*50}")
    if not is_stable:
        print("WARNING: at least one root lies on or outside the unit")
        print("circle. IRFs and FEVD below would not be reliable --")
        print("reconsider the lag order or check for unit roots in the")
        print("underlying variables before proceeding.")

    # Plot every eigenvalue on the complex plane against the unit circle.
    # A stable system has every point strictly inside the circle; a point
    # on or outside it signals instability.
    fig, ax = plt.subplots(figsize=(5, 5))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="gray", linewidth=1, linestyle="--")
    ax.scatter(eigenvalues.real, eigenvalues.imag, color="firebrick", zorder=3)
    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title("VAR Companion Matrix Roots vs. the Unit Circle")
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    fig.tight_layout()
    plt.show()

    return {"eigenvalues": eigenvalues, "max_modulus": max_modulus, "is_stable": is_stable}