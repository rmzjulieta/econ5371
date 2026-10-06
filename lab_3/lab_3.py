"""
Lab 3: Structural Breaks
=========================

Applies the Chow test, the QLR test, and Bai-Perron multiple-break
detection to a synthetic monthly inflation series with two genuine
regime shifts.

New this lab: the Chow and QLR test functions live in a separate file,
functions.py, in this same folder. See the docstring at the top of
that file for why.

Run this script section by section (recommended: in Positron, run cell
by cell using the '# %%' markers), or top to bottom as a single script.
"""

# %% Setup
#
# Step 1: Install the required packages.
#
#     pip install -r requirements.txt
#
# Run this once in a terminal, in this lab's folder -- not as part of
# this script.
#
# Step 2: Set the lab folder location.

LAB_FOLDER = r"C:\Users\ramir\OneDrive\Economics Master Degree\4.Fall 2026\Week 7"

import os

os.chdir(LAB_FOLDER)

# Step 3: Import packages, including our own functions.py.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functions import run_chow_test, run_qlr_test, detect_multiple_breaks

pd.set_option('display.precision', 2)


# %% 1. Load and Visualize the Data

df = pd.read_csv("inflation_synthetic.csv", parse_dates=["date"])
series = df.set_index("date")["inflation_rate"]
series.index.freq = "MS"

print(f"{'='*50}")
print(f"Inflation Rate -- Loaded Series")
print(f"{'='*50}")
print(f"{'Observations:':<20}{len(series):>10}")
print(f"{'Date range:':<20}{str(series.index.min().date()):>10} to {series.index.max().date()}")
print(f"{'Mean:':<20}{series.mean():>10.2f}")
print(f"{'Std. Dev.:':<20}{series.std():>10.2f}")
print(f"{'='*50}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series.index, series.values, color="tab:blue", linewidth=1.2)
ax.set_title("Synthetic Inflation Rate (Monthly)")
ax.set_xlabel("Date")
ax.set_ylabel("Inflation Rate (%)")
plt.tight_layout()
plt.show()

# Discussion: Does the mean of this series look constant over time, or
# do you see one or more points where its behavior seems to shift?


# %% 2. Chow Test -- A Known Break Date
#
# Suppose we have a specific reason to suspect a break at a particular
# date -- 2016-01, say. The Chow test asks: does the series' mean
# differ before and after this exact point?
#
# H0: no break -- the mean is the same in both subsamples.

break_date = "2016-01-01"
break_index = series.index.get_loc(break_date)

chow_result = run_chow_test(series.values, break_index, label=f"break at {break_date}")

# Discussion: Given the F-statistic and p-value above, do we reject H0?
# What does that tell us -- and what does it NOT tell us, if we hadn't
# already known to look at this specific date?


# %% 3. QLR Test -- An Unknown Break Date
#
# In practice we rarely know the break date in advance. The QLR test
# searches across candidate break points and reports the one that
# produces the largest Chow F-statistic.

qlr_result = run_qlr_test(series.values, trim=0.15)

qlr_break_date = series.index[qlr_result["break_index"]].date()
print(f"QLR's best break point corresponds to: {qlr_break_date}")

# Discussion: Does QLR find the same break date we assumed in Section
# 2 -- without being told where to look? What does it mean that QLR's
# F-statistic at its best point matches the Chow F-statistic from
# Section 2?


# %% 4. Bai-Perron -- Multiple Break Detection
#
# Chow and QLR are both single-break tools: QLR finds the ONE break
# point with the strongest evidence, even if the series has more than
# one regime change. Bai-Perron generalizes this to multiple breaks.
#
# detect_multiple_breaks() (in functions.py) implements a simplified
# version of this idea from scratch: it applies the SAME QLR search
# you just ran in Section 3, first to the whole series, then
# recursively to each half whenever a significant break is found,
# until no remaining segment has one left. See that function's
# docstring for why this lab builds it this way instead of using a
# dedicated package, and for why it uses a stricter-than-usual
# significance threshold.

detected_breaks = detect_multiple_breaks(series.values, min_size=12)

print(f"\n{'-'*50}")
print(f"Bai-Perron (recursive QLR segmentation)")
print(f"{'-'*50}")
print(f"{'Breaks detected:':<20}{len(detected_breaks):>10}")
for bp in detected_breaks:
    print(f"  index {bp:>4} -> {series.index[bp].date()}")
print(f"{'-'*50}")

# segment_bounds turns the list of break points into a list of
# (start, end) boundaries covering the whole series -- e.g. two
# breaks at indices 72 and 132 become three segments: [0:72],
# [72:132], [132:end]. We'll compute each segment's own mean and
# standard deviation below, to show what actually changed at each
# break, not just where it happened.
segment_bounds = [0] + detected_breaks + [len(series)]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series.index, series.values, color="tab:blue", linewidth=1.2,
         label="Inflation Rate", zorder=3)

for bp in detected_breaks:
    ax.axvline(series.index[bp], color="firebrick", linestyle="--",
               linewidth=1, zorder=2)

# For each segment between breaks: draw its mean as a horizontal
# line, and shade a +/- 2 standard deviation band around that mean.
# The mean line makes the level shift Bai-Perron detected visible
# directly on the plot, rather than just in the printed break dates.
# The +/- 2 SD band adds something the mean line alone can't show:
# whether the REGIME's dispersion changed too, not just its level --
# compare the width of the shaded band across segments.
for i in range(len(segment_bounds) - 1):
    start, end = segment_bounds[i], segment_bounds[i + 1]
    seg_dates = series.index[start:end]
    seg_values = series.values[start:end]
    seg_mean = seg_values.mean()
    seg_sd = seg_values.std()

    ax.hlines(seg_mean, seg_dates[0], seg_dates[-1],
              color="black", linewidth=1.5, zorder=4,
              label="Segment mean" if i == 0 else None)
    ax.fill_between(seg_dates, seg_mean - 2 * seg_sd, seg_mean + 2 * seg_sd,
                     color="gray", alpha=0.15, zorder=1,
                     label="Segment mean +/- 2 SD" if i == 0 else None)

ax.set_title("Inflation Rate with Bai-Perron Breakpoints")
ax.set_xlabel("Date")
ax.set_ylabel("Inflation Rate (%)")
ax.set_xlim(series.index.min(), series.index.max())
ax.legend()
plt.tight_layout()
plt.show()

# Discussion: How many breaks did Bai-Perron find? Does it agree with
# QLR on at least one of them? Is there a break here that neither Chow
# nor QLR could have told us about -- and why not, given how each test
# is built?
#
# Now look at the segment means and +/- 2 SD bands you just plotted:
# did the break points only shift the LEVEL of inflation, or did the
# spread around that level change too? What would it mean, economically,
# for a regime to be both higher-inflation AND more volatile at the
# same time -- rather than just higher?


# %% 5. Putting It Together -- Which Test, When?
#
# This lab used three different tools on the same series, each
# answering a different version of "is there a break?" Before moving
# on, it's worth being explicit about what each one actually needs
# from you, and what you get back -- because in your own project data,
# nobody hands you the right test. You have to choose.
#
#   Chow test:  You supply a candidate break date. Most powerful of
#               the three WHEN you have a good reason to suspect a
#               specific date (a policy change, a known shock) --
#               it isn't searching for anything, so it isn't paying
#               a statistical price for a search.
#
#   QLR test:   You supply a search window (via `trim`), not a date.
#               Finds the single break the data supports most
#               strongly. The right tool when you suspect ONE break
#               but don't know when -- but it is built to find only
#               one, so it will stay silent about a second break
#               even if one is there.
#
#   Bai-Perron: You supply nothing but the series (plus some tuning
#               knobs: `min_size`, `alpha`). Finds however many
#               breaks the data supports. The right tool when you
#               don't know IF there's one break, several, or none --
#               but that flexibility is exactly why it needs the
#               stricter significance threshold from Section 4: the
#               more places a test is allowed to look, the more
#               places it can find something by chance alone.
#
# None of these is simply "better" than the others -- each buys
# generality (searching over more unknowns) at the cost of a weaker
# test, and that tradeoff is unavoidable, not a defect of any one
# method's implementation.
#
# Discussion: When you apply unit root tests to your own project's,
# you will face the same choice about trend and seasonal specification.
# If you suspect your series has a structural break somewhere in its
# history but you are not sure when, or how many -- which of these three
# tools would you reach for first, and what would make you switch to a different one?


# %% Lab Complete -- Before You Leave

print("\n" + "="*65)
print("LAB COMPLETE -- BEFORE YOU LEAVE:")
print("="*65)
print("1. Save this script")
print("2. Commit your changes:")
print('     git add .')
print('     git commit -m "Lab 3: structural breaks"')
print('     git push')
print("3. Update README.md if you added new files or changed structure")
print("="*65)