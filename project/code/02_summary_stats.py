"""
02_summary_stats.py

Purpose : Produce the summary-statistics table and the figure of the raw series
          used in the data memo and in the Data section of the paper.
Input   : data/clean/lfpr_women_by_age.csv  (created by 01_clean_data.py)
Outputs : output/table1_summary_stats.csv
          output/fig1_lfpr_by_age.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# --- Paths -----------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_DIR / "data" / "clean" / "lfpr_women_by_age.csv"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN_FILE, index_col="year")

# Readable labels for tables and figures (the column names are for code only).
labels = {
    "age_16plus": "16+ (aggregate)",
    "age_16_24": "16-24",
    "age_25_34": "25-34",
    "age_35_44": "35-44",
    "age_45_54": "45-54",
    "age_55_64": "55-64",
    "age_65plus": "65+",
}

# --- 1. Summary statistics table ---------------------------------------------
# Besides the usual moments, we report the first and last values and the year
# of the peak: for trending series these say more than the mean alone.
table = pd.DataFrame({
    "N": df.count(),
    "Mean": df.mean(),
    "Std. dev.": df.std(),
    "Min": df.min(),
    "Max": df.max(),
    "Year of max": df.idxmax(),
    "1948": df.loc[1948],
    "2025": df.loc[2025],
})
table.index = [labels[c] for c in table.index]
table.index.name = "Age group"
table = table.round(1)

table.to_csv(OUTPUT_DIR / "table1_summary_stats.csv")
print(table.to_string())

# --- 2. Figure of the raw series ----------------------------------------------
# The six age groups use distinct, colorblind-checked colors; the aggregate is
# drawn in black and dashed so it reads as the benchmark, not a seventh group.
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
age_groups = [c for c in df.columns if c != "age_16plus"]

fig, ax = plt.subplots(figsize=(9, 5.5))

for col, color in zip(age_groups, colors):
    ax.plot(df.index, df[col], color=color, linewidth=2, label=labels[col])

ax.plot(df.index, df["age_16plus"], color="black", linewidth=2.5,
        linestyle="--", label=labels["age_16plus"])

ax.set_title("Labor force participation rate of U.S. women by age group, 1948-2025")
ax.set_xlabel("Year")
ax.set_ylabel("Participation rate (%)")
ax.set_xlim(df.index.min(), df.index.max())
ax.set_ylim(0, 100)
ax.grid(axis="y", color="0.9")
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
ax.legend(title="Age group", loc="upper left", ncol=2, frameon=False)

fig.text(0.01, 0.01,
         "Source: U.S. Department of Labor, Women's Bureau, from BLS Current "
         "Population Survey (annual averages).",
         fontsize=8, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUTPUT_DIR / "fig1_lfpr_by_age.png", dpi=300)
print("Saved fig1_lfpr_by_age.png and table1_summary_stats.csv")
