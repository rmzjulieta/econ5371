"""
functions.py -- Lab 5 (Vector Error Correction Model)

One function this lab: run_johansen_test(). It's called only once in
lab_5.py. This isn't here to avoid repeating a call several times. 
It's here for the following reason: Johansen test's raw output is
dense -- eigenvalues, trace statistics, max-eigenvalue statistics,
critical values at three significance levels, for every candidate
rank -- and formatting all of that clearly is exactly the kind of
logic worth keeping in one place you can reuse in a future lab or
your own project, rather than rewriting the formatting by hand each
time you need this test.

lab_5.py imports it with:

    from functions import run_johansen_test
"""

from statsmodels.tsa.vector_ar.vecm import coint_johansen


def run_johansen_test(data, variable_names, det_order=0, k_ar_diff=1):
    """Run the Johansen cointegration test and print a formatted summary.

    Reports both the trace statistic and the maximum-eigenvalue
    statistic, each against critical values at 90%, 95%, and 99%
    confidence, for every candidate cointegrating rank from 0 up to
    k-1 (where k is the number of variables).

    Parameters
    ----------
    data : array-like, shape (n_obs, k)
        The variables to test, in levels (NOT differenced).
    variable_names : list of str
        Names of the variables, in the same column order as `data`,
        used only for labeling printed output.
    det_order : int, default 0
        Deterministic trend specification passed to coint_johansen.
        0 = a constant is restricted to the cointegrating relation
        (appropriate when the variables have no deterministic trend
        of their own, only a long-run equilibrium level -- the usual
        case for a pair like short and long interest rates).
    k_ar_diff : int, default 1
        Number of lagged differences to include -- one less than the
        VAR lag order you would otherwise choose (a VECM(1) here
        corresponds to a VAR(2) in levels).

    Returns
    -------
    dict
        Keys: "result" (the raw coint_johansen output object),
        "suggested_rank" (int, based on the trace test at 95%).
    """
    k = data.shape[1]
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    print(f"\n{'='*70}")
    print(f"Johansen Cointegration Test: {', '.join(variable_names)}")
    print(f"{'='*70}")

    print(f"\nTrace statistic (H0: cointegrating rank <= r)")
    print(f"{'-'*70}")
    print(f"{'r':<6}{'Trace stat':>14}{'90% CV':>14}{'95% CV':>14}{'99% CV':>14}")
    print(f"{'-'*70}")
    for r in range(k):
        print(f"{r:<6}{result.lr1[r]:>14.4f}{result.cvt[r,0]:>14.4f}"
              f"{result.cvt[r,1]:>14.4f}{result.cvt[r,2]:>14.4f}")
    print(f"{'-'*70}")

    print(f"\nMaximum eigenvalue statistic (H0: cointegrating rank = r)")
    print(f"{'-'*70}")
    print(f"{'r':<6}{'Max-eig stat':>14}{'90% CV':>14}{'95% CV':>14}{'99% CV':>14}")
    print(f"{'-'*70}")
    for r in range(k):
        print(f"{r:<6}{result.lr2[r]:>14.4f}{result.cvm[r,0]:>14.4f}"
              f"{result.cvm[r,1]:>14.4f}{result.cvm[r,2]:>14.4f}")
    print(f"{'-'*70}")

    # Suggest a rank from the trace test at the 95% level: find the
    # smallest r for which we FAIL to reject H0 (rank <= r). That is
    # the standard sequential decision rule -- test r=0 first; if
    # rejected, test r=1; stop at the first rank not rejected.
    suggested_rank = k  # default: reject every rank up to k-1
    for r in range(k):
        if result.lr1[r] <= result.cvt[r, 1]:  # 95% column
            suggested_rank = r
            break

    print(f"\nSuggested cointegrating rank (trace test, 95%): {suggested_rank}")
    print("This is a suggestion from a mechanical decision rule, not a")
    print("substitute for judgment -- confirm it makes economic sense")
    print("before using it below.")

    return {"result": result, "suggested_rank": suggested_rank}