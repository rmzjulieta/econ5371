"""
06_figure_breaks.py

Purpose : Plot each LFPR series with its Bai-Perron break years (dashed lines)
          and a linear trend fitted within each period, so the change in the
          pace of participation growth across periods is visible.
Input   : data/clean/lfpr_women_by_age.csv
          output/table6_period_analysis.csv
Output  : output/fig2_breaks_by_age.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Paths -----------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_DIR / "data" / "clean" / "lfpr_women_by_age.csv"
PERIODS_FILE = PROJECT_DIR / "output" / "table6_period_analysis.csv"
FIG_FILE = PROJECT_DIR / "output" / "fig2_breaks_by_age.png"

df = pd.read_csv(CLEAN_FILE, index_col="year")
periods = pd.read_csv(PERIODS_FILE)

# Readable panel titles for each column name.
labels = {"age_16plus": "16+ (aggregate)", "age_16_24": "16-24",
          "age_25_34": "25-34", "age_35_44": "35-44", "age_45_54": "45-54",
          "age_55_64": "55-64", "age_65plus": "65+"}

# 7 series -> a 4 x 2 grid; the eighth panel is left empty.
fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True)
axes = axes.flatten()

# Loop over every series: one panel each.
for ax, col in zip(axes, df.columns):
    ax.plot(df.index, df[col], color="gray", linewidth=1.2, label="LFPR")

    # Inner loop over this series' periods: fitted trend and break line.
    for _, p in periods[periods["series"] == col].iterrows():
        seg = df.loc[p["start"]:p["end"], col]
        coef = np.polyfit(seg.index, seg.values, 1)
        ax.plot(seg.index, np.polyval(coef, seg.index), color="firebrick",
                linewidth=1.8)
        # Slope label: percentage points per year in this period.
        # Placed in a reserved strip at the top of the panel (x in years,
        # y as a fraction of panel height), so it never covers the data.
        ax.text(seg.index.values.mean(), 0.92, f"{p['trend_slope']:+.2f}",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=8,
                color="firebrick")
        if p["start"] != df.index.min():
            ax.axvline(p["start"], color="black", linestyle="--", linewidth=0.8)

    # Add 15% empty space above the data for the slope labels.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))
    ax.set_title(labels[col], fontsize=10)
    ax.set_ylabel("LFPR (%)")

axes[-1].axis("off")
axes[-2].set_xlabel("Year")
axes[-3].set_xlabel("Year")
fig.suptitle("Female labor force participation by age group: "
             "Bai-Perron breaks and period trends\n"
             "(red labels: trend slope, percentage points per year)",
             fontsize=11)
fig.tight_layout()
fig.savefig(FIG_FILE, dpi=200)
plt.show()