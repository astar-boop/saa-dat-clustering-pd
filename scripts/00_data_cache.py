"""
00 - Prepare the data cache: outputs/cache/curated.parquet and outputs/cache/alb.csv.

Almost all scripts read the curated data cut from a copy in parquet format rather than
from the Excel file (14 MB, slow to open), and albumin from an extract of the biospecimen
file (177 MB). This step builds the two copies if they are missing; if they are already
there it leaves them untouched.
"""
import os
import pandas as pd
from _paths import curated_xlsx
from _alb import path_alb

BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
CACHE = os.path.join(OUT, "cache", "curated.parquet")
os.makedirs(os.path.dirname(CACHE), exist_ok=True)

if os.path.exists(CACHE):
    print(f"cache of the curated data cut already present: {CACHE}")
else:
    cur = pd.read_excel(curated_xlsx(BASE), sheet_name="20260511")
    cur.to_parquet(CACHE, index=False)
    print(f'cache of the curated data cut written: {CACHE} ({cur.shape[0]} rows, {cur.shape[1]} '
          f'columns)')
print(f'albumin: {path_alb(OUT)}')
