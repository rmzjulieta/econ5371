"""
01_clean_data.py

Purpose : Read the raw Women's Bureau file on female labor force participation
          by age group and convert it into a tidy, analysis-ready table.
Input   : data/raw/women_lfpr_by_age.csv
          (tab-separated, UTF-16 encoded, one row per age group, one column per year)
Output  : data/clean/lfpr_women_by_age.csv
          (comma-separated, UTF-8, one row per year, one column per age group)
"""

from pathlib import Path

import pandas as pd

# --- Paths -----------------------------------------------------------------
# Build every path from the location of this script, so the code works no
# matter which folder Positron (or the terminal) is running from.
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "women_lfpr_by_age.csv"
CLEAN_FILE = PROJECT_DIR / "data" / "clean" / "lfpr_women_by_age.csv"

# --- 1. Read the raw file ----------------------------------------------------
# The file downloaded from the DOL chart is not a standard CSV:
#   * it is UTF-16 encoded (pandas assumes UTF-8 unless told otherwise), and
#   * columns are separated by tabs, not commas.
# Row 0 only repeats the word "Year"; row 1 holds the actual years.
# header=1 skips row 0 and uses the years as column names.
# index_col=0 uses the first column (the age-group labels) as row names.
raw = pd.read_csv(RAW_FILE, sep="\t", encoding="utf-16", header=1, index_col=0)

# --- 2. Reshape: years as rows -----------------------------------------------
# Time-series tools expect one observation per row, so we transpose the table:
# years become the index and each age group becomes a column.
df = raw.T
df.index = df.index.astype(int)
df.index.name = "year"

# --- 3. Short, code-friendly column names -------------------------------------
# Long labels with spaces are awkward in code; we keep a clear mapping here so
# the original labels can always be traced back.
rename_map = {
    "16 years and older": "age_16plus",
    "16 to 24 years": "age_16_24",
    "25 to 34 years": "age_25_34",
    "35 to 44 years": "age_35_44",
    "45 to 54 years": "age_45_54",
    "55 to 64 years": "age_55_64",
    "65 years and older": "age_65plus",
}
df = df.rename(columns=rename_map)
df.columns.name = None

# --- 4. Basic checks ------------------------------------------------------------
# Stop with a clear message if something unexpected happened, instead of
# silently producing a wrong file.
assert list(df.columns) == list(rename_map.values()), "Unexpected age-group labels"
assert df.index.min() == 1948 and df.index.max() == 2025, "Unexpected sample period"
assert df.notna().all().all(), "Missing values found"
assert df.index.is_monotonic_increasing and df.index.is_unique, "Years out of order"

# --- 5. Save -------------------------------------------------------------------
CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_FILE)

print(f"Saved {CLEAN_FILE.name}: {df.shape[0]} years x {df.shape[1]} series")
print(df.head())