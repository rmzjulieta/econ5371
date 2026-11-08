"""
functions.py -- ECON 5371 research project

Purpose : Reusable test functions for the structural break and forecast
          analysis of U.S. female labor force participation by age group.
Used by : 04_break_tests.py, 05_period_analysis.py, 07_forecast.py

Functions:
  - run_chow_test()          Chow test for a break at a known date
  - run_qlr_test()           QLR test for a break at an unknown date
  - detect_multiple_breaks() Simplified Bai-Perron procedure (recursive QLR)
  - dm_test()                Diebold-Mariano test of equal forecast accuracy

The first three functions were first developed in Lab 3 and adapted for
this project. They live in a separate file so that every script can
import them, instead of copying code between scripts.
"""

import numpy as np
from scipy import stats


def run_chow_test(series, break_index, label=""):
    """Run a Chow test for a structural break at a known point.

    Tests whether the mean of `series` differs before and after
    `break_index`, by comparing the residual sum of squares from one
    pooled regression (series on a constant) against two separate
    subsample regressions.

    Parameters
    ----------
    series : array-like
        The series to test, as a plain array of values (not a
        pandas Series with a date index -- pass `series.values`).
    break_index : int
        The position (0-based) in `series` where the hypothesized
        break occurs. Everything before this index is subsample 1;
        everything from this index onward is subsample 2.
    label : str, default ""
        A short description of the break point, used in printed
        output.

    Returns
    -------
    dict
        Keys: "f_stat", "p_value", "reject_h0" (bool).
    """
    series = np.asarray(series)
    n = len(series)
    k = 1  # one parameter (the mean) per subsample regression

    # Split the series into the two subsamples implied by break_index.
    # y1 is everything strictly before the break; y2 is everything
    # from the break point onward. Together they cover the whole
    # series exactly once, with no overlap and no gap.
    y1 = series[:break_index]
    y2 = series[break_index:]

    # Pooled model: one constant fit to the whole series.
    rss_pooled = np.sum((series - series.mean()) ** 2)

    # Unpooled model: a separate constant fit to each subsample.
    rss1 = np.sum((y1 - y1.mean()) ** 2)
    rss2 = np.sum((y2 - y2.mean()) ** 2)
    rss_unpooled = rss1 + rss2

    # Chow F-statistic: how much RSS falls when we let the mean differ
    # across subsamples, relative to how much unexplained variation
    # remains once we do.
    f_stat = ((rss_pooled - rss_unpooled) / k) / (rss_unpooled / (n - 2 * k))
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)
    reject_h0 = p_value < 0.05

    print(f"\n{'-'*50}")
    print(f"Chow Test{': ' + label if label else ''}")
    print(f"H0: no break -- the mean is the same in both subsamples")
    print(f"{'-'*50}")
    print(f"{'F-statistic:':<20}{f_stat:>10.4f}")
    print(f"{'p-value:':<20}{p_value:>10.4f}")
    print(f"{'-'*50}")
    verdict = "Reject H0 -- evidence of a break" if reject_h0 \
        else "Fail to reject H0 -- no evidence of a break"
    print(verdict)

    return {"f_stat": f_stat, "p_value": p_value, "reject_h0": reject_h0}


def run_qlr_test(series, trim=0.15):
    """Run the QLR (Quandt Likelihood Ratio) test for an unknown break.

    Computes a Chow F-statistic at every candidate break point within
    the trimmed interior of the series, and reports the one that
    produces the largest F-statistic -- the break point the data
    itself supports most strongly.

    Parameters
    ----------
    series : array-like
        The series to test, as a plain array of values.
    trim : float, default 0.15
        Fraction of the series to exclude from BOTH ends when
        searching for a break. Candidate break points too close to
        either end leave one subsample too small to estimate
        reliably -- trimming keeps the search to interior points
        where both subsamples have enough observations.

    Returns
    -------
    dict
        Keys: "break_index", "f_stat", "p_value".
    """
    series = np.asarray(series)
    n = len(series)

    # lo/hi mark the search window: we only test candidate breaks
    # strictly between these two points. Without this trim, a
    # candidate break near either end would leave one subsample with
    # only a handful of observations -- too few to estimate a mean
    # reliably, and the Chow F-statistic for that candidate would be
    # driven by sample-size noise rather than genuine evidence of a
    # break.
    lo = int(n * trim)
    hi = int(n * (1 - trim))

    # best_F must start at -inf, not None, because every candidate's
    # F-statistic is compared against it with `F > best_F` below --
    # None can't be compared to a float. best_t, by contrast, is only
    # ever ASSIGNED (never compared), so starting it at None is safe:
    # it simply means "no best break point found yet."
    best_F = -np.inf
    best_t = None

    for t in range(lo, hi):
        y1 = series[:t]
        y2 = series[t:]

        rss_pooled = np.sum((series - series.mean()) ** 2)
        rss1 = np.sum((y1 - y1.mean()) ** 2)
        rss2 = np.sum((y2 - y2.mean()) ** 2)
        rss_unpooled = rss1 + rss2

        k = 1
        F = ((rss_pooled - rss_unpooled) / k) / (rss_unpooled / (n - 2 * k))

        if F > best_F:
            best_F = F
            best_t = t

    p_value = 1 - stats.f.cdf(best_F, 1, n - 2)

    print(f"\n{'-'*50}")
    print(f"QLR Test (search window: trim={trim})")
    print(f"H0: no break anywhere in the search window")
    print(f"{'-'*50}")
    print(f"{'Best break index:':<20}{best_t:>10}")
    print(f"{'F-statistic:':<20}{best_F:>10.4f}")
    print(f"{'p-value:':<20}{p_value:>10.4f}")
    print(f"{'-'*50}")

    return {"break_index": best_t, "f_stat": best_F, "p_value": p_value}


def _qlr_on_segment(segment, trim=0.15):
    """Internal helper: run one QLR search within a single segment.

    Same logic as run_qlr_test(), but silent (no printing) and
    returning None when the segment is too short to search -- used
    by detect_multiple_breaks() below, which calls this many times
    over shrinking sub-segments and only wants to print once, at
    the end, for the full set of breaks found.
    """
    n = len(segment)
    lo = int(n * trim)
    hi = int(n * (1 - trim))
    if hi - lo < 2:
        return None

    best_F = -np.inf
    best_t = None
    for t in range(lo, hi):
        y1 = segment[:t]
        y2 = segment[t:]
        rss_pooled = np.sum((segment - segment.mean()) ** 2)
        rss_unpooled = np.sum((y1 - y1.mean()) ** 2) + np.sum((y2 - y2.mean()) ** 2)
        F = ((rss_pooled - rss_unpooled) / 1) / (rss_unpooled / (n - 2))
        if F > best_F:
            best_F, best_t = F, t

    p_value = 1 - stats.f.cdf(best_F, 1, n - 2)
    return best_t, best_F, p_value


def detect_multiple_breaks(series, min_size=12, trim=0.15, alpha=0.0005):
    """Detect multiple structural breaks via recursive QLR segmentation.

    This is a simplified, from-scratch version of Bai-Perron multiple-
    break detection: it repeatedly applies the SAME single-break QLR
    logic as run_qlr_test() to successively smaller segments of the
    series, splitting a segment in two whenever QLR finds a
    sufficiently significant break inside it, and stopping when no
    remaining segment has one.

    Why build this instead of using a dedicated package (e.g.
    `ruptures`, which implements the actual Bai-Perron / PELT
    algorithm)? That package requires a compiled C extension, and
    installing it fails on some machines without the
    Microsoft C++ Build Tools (Windows) already installed -- a
    heavyweight dependency for one step of one lab. Binary
    segmentation via repeated QLR is conceptually the same idea
    (search for the single strongest break, then recurse on each
    side) and needs nothing beyond NumPy and SciPy, which every
    other test in this lab already uses.

    A note on `alpha`: naively re-running QLR at the usual 0.05
    significance level on every segment tends to find SPURIOUS extra
    breaks, because we are effectively running many hypothesis tests
    (one per candidate split point, repeated over every segment) and
    some will look "significant" by chance alone. `alpha=0.0005` is a
    rough Bonferroni-style correction: with roughly 100-150 candidate
    split points tested per full-series search (set by `trim`), a
    family-wise error rate of 0.05 implies an per-test alpha near
    0.05/126 ~= 0.0004. We round to 0.0005. This is not a free
    parameter to tune until the answer looks right -- it is a
    deliberately conservative threshold chosen to avoid exactly the
    over-detection problem described above.

    Parameters
    ----------
    series : array-like
        The series to search, as a plain array of values.
    min_size : int, default 12
        Minimum number of observations required on EITHER side of a
        candidate break for it to be accepted. Prevents the
        recursion from splitting off segments too short to estimate
        reliably.
    trim : float, default 0.15
        Passed to the QLR search within each segment -- see
        run_qlr_test().
    alpha : float, default 0.0005
        Significance threshold for accepting a candidate break. See
        the note above.

    Returns
    -------
    list of int
        Sorted list of detected break indices (0-based positions in
        the original series).
    """
    series = np.asarray(series)
    breaks = []

    def recurse(start, end):
        segment = series[start:end]
        if end - start < 2 * min_size:
            return

        result = _qlr_on_segment(segment, trim=trim)
        if result is None:
            return

        local_t, f_stat, p_value = result
        global_break = start + local_t

        # Reject candidates too close to either edge of this segment
        # -- symmetric with the min_size check in run_qlr_test()'s
        # trim logic, but enforced in RAW observation counts here
        # rather than as a fraction, since segments shrink as the
        # recursion proceeds.
        if local_t < min_size or (len(segment) - local_t) < min_size:
            return

        if p_value < alpha:
            breaks.append(global_break)
            recurse(start, global_break)
            recurse(global_break, end)

    recurse(0, len(series))
    return sorted(breaks)

def dm_test(e_1, e_2, h=1, small_sample=True):
    """Diebold-Mariano test of equal forecast accuracy (squared-error loss).

    Follows the course slides: d_t = L(e_1) - L(e_2), H0: E[d_t] = 0,
    HA: E[d_t] != 0 (two-sided). d_bar > 0 means model 2 is more accurate.
    The variance of d_bar is a Newey-West long-run variance with K = h - 1
    lags; for h = 1 this reduces to the simple variance.

    Parameters
    ----------
    e_1, e_2 : array-like
        Forecast errors (actual minus forecast) of models 1 and 2.
    h : int, default 1
        Forecast horizon.
    small_sample : bool, default True
        If True, apply the Harvey-Leybourne-Newbold (1997) correction and
        use a t(n-1) distribution; if False, use N(0,1) as in the slides.

    Returns
    -------
    dict
        Keys: "d_bar", "dm_stat", "p_value".
    """
    e_1, e_2 = np.asarray(e_1), np.asarray(e_2)

    # Step 1-2 (slides): loss differential and its average.
    d = e_1**2 - e_2**2
    n = len(d)
    d_bar = d.mean()

    # Autocovariance of d at lag k: gamma_k = (1/n) * sum (d_t - d_bar)(d_{t-k} - d_bar)
    def gamma(k):
        return np.sum((d[k:] - d_bar) * (d[:n - k] - d_bar)) / n

    # Long-run variance with Newey-West (Bartlett) weights w_k = 1 - k/(K+1).
    # h-step errors overlap, so d_t is autocorrelated up to lag h-1.
    K = h - 1
    lrv = gamma(0) + 2 * sum((1 - k / (K + 1)) * gamma(k) for k in range(1, K + 1))

    dm = d_bar / np.sqrt(lrv / n)

    if small_sample:
        # HLN: with few forecasts, N(0,1) is unreliable; rescale and use t(n-1).
        dm = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        p_value = 2 * (1 - stats.t.cdf(abs(dm), n - 1))
    else:
        p_value = 2 * (1 - stats.norm.cdf(abs(dm)))

    return {"d_bar": d_bar, "dm_stat": dm, "p_value": p_value}