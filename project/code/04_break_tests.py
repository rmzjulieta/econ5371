"""
04_break_tests.py

Purpose : Detect structural breaks in the first-differenced female LFPR series
          (the average annual change, or drift) using three tests:
            - Chow test at candidate break dates chosen in advance
            - QLR test (one break, unknown date)
            - Bai-Perron style multiple-break detection (recursive QLR)
          The break dates found here define the periods analyzed in script 05.
Input   : data/clean/lfpr_women_by_age.csv
Output  : output/table4_chow_tests.csv
          output/table5_qlr_baiperron.csv
Note    : Test functions come from functions.py (ECON 5371, Lab 3).
"""

from pathlib import Path

import pandas as pd

from functions import run_chow_test, run_qlr_test, detect_multiple_breaks

# --- Paths -----------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
CLEAN_FILE = PROJECT_DIR / "data" / "clean" / "lfpr_women_by_age.csv"
CHOW_FILE = PROJECT_DIR / "output" / "table4_chow_tests.csv"
BREAKS_FILE = PROJECT_DIR / "output" / "table5_qlr_baiperron.csv"
CHOW_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN_FILE, index_col="year")

# --- Candidate break dates for the Chow test ---------------------------------
# One list per series. Each date must be justified in the paper (data memo
# figure, historical or policy event). FILL IN the empty lists.
candidates = {
    "age_16plus": [1999],
    "age_16_24": [1987, 2000],
    "age_25_34": [1972],
    "age_35_44": [],
    "age_45_54": [],
    "age_55_64": [],
    "age_65plus": [1985],
}

chow_rows = []
break_rows = []

# Loop over every age group; the same procedure runs for all seven series.
for col in df.columns:
    # Work with first differences: the break tests look for a change in the
    # MEAN, and the mean of the differenced series is the drift (the average
    # annual change in participation).
    dy = df[col].diff().dropna()
    print(f"\n{'='*50}\n{col}\n{'='*50}")

    # 1. Chow test at each candidate date (inner loop over dates)
    for year in candidates[col]:
        pos = dy.index.get_loc(year)
        res = run_chow_test(dy.values, pos, label=f"{col}, break at {year}")
        chow_rows.append({"series": col, "candidate_year": year,
                          "f_stat": round(res["f_stat"], 2),
                          "p_value": round(res["p_value"], 3)})

    # 2. QLR: single break, unknown date (15% trimmed at each end)
    qlr = run_qlr_test(dy.values, trim=0.15)

    # 3. Bai-Perron style: as many breaks as the data support.
    # min_size=12: each period must span at least 12 years.
    bp = detect_multiple_breaks(dy.values, min_size=12)

    break_rows.append({"series": col,
                       "qlr_break_year": dy.index[qlr["break_index"]],
                       "qlr_f_stat": round(qlr["f_stat"], 2),
                       "bp_break_years": [int(dy.index[b]) for b in bp]})

chow_table = pd.DataFrame(chow_rows)
break_table = pd.DataFrame(break_rows)
print("\n", chow_table, "\n\n", break_table)
chow_table.to_csv(CHOW_FILE, index=False)
break_table.to_csv(BREAKS_FILE, index=False)