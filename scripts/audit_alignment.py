from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

ASSETS = [
    "BTC_USDT",
    "ETH_USDT",
    "EURUSD",
    "GOLD",
    "XAUUSD",
]

print("=" * 100)
print("ALIGNMENT AUDIT")
print("=" * 100)

# -------------------------------------------------------------------
# Load raw datasets
# -------------------------------------------------------------------

raw = {}

for asset in ASSETS:
    path = RAW_DIR / f"{asset}_1h_raw.parquet"

    if not path.exists():
        continue

    df = pd.read_parquet(path)
    df = df.sort_index()

    raw[asset] = df

print(f"\nLoaded {len(raw)} raw datasets.\n")

# -------------------------------------------------------------------
# Dataset summary
# -------------------------------------------------------------------

print("=" * 100)
print("RAW DATA SUMMARY")
print("=" * 100)

for asset, df in raw.items():

    print(f"\n{asset}")
    print("-" * 60)
    print(f"Rows        : {len(df):,}")
    print(f"Start       : {df.index.min()}")
    print(f"End         : {df.index.max()}")
    print(f"Duplicates  : {df.index.duplicated().sum()}")

# -------------------------------------------------------------------
# BTC reference
# -------------------------------------------------------------------

btc = raw["BTC_USDT"]
btc_index = btc.index

print("\n")
print("=" * 100)
print("COMPARISON AGAINST BTC")
print("=" * 100)

for asset, df in raw.items():

    if asset == "BTC_USDT":
        continue

    print(f"\n{'='*80}")
    print(asset)
    print("="*80)

    idx = df.index

    missing = btc_index.difference(idx)
    extra = idx.difference(btc_index)

    print(f"BTC timestamps         : {len(btc_index):,}")
    print(f"{asset} timestamps     : {len(idx):,}")
    print(f"Missing vs BTC         : {len(missing):,}")
    print(f"Extra vs BTC           : {len(extra):,}")

    # ------------------------
    # Weekend analysis
    # ------------------------

    weekend = missing[missing.dayofweek >= 5]
    weekday = missing[missing.dayofweek < 5]

    print(f"Weekend missing        : {len(weekend):,}")
    print(f"Weekday missing        : {len(weekday):,}")

    if len(weekday):

        print("\nFirst 20 NON-WEEKEND missing timestamps:")

        for ts in weekday[:20]:
            print(ts)

    # ------------------------
    # Missing by weekday
    # ------------------------

    if len(missing):

        print("\nMissing by weekday:")

        print(
            missing.to_series()
            .dt.day_name()
            .value_counts()
            .sort_index()
        )

    # ------------------------
    # Missing by hour
    # ------------------------

    if len(missing):

        print("\nMissing by hour:")

        print(
            missing.to_series()
            .dt.hour
            .value_counts()
            .sort_index()
        )

    # ------------------------
    # Long gaps
    # ------------------------

    if len(missing):

        print("\nGap lengths:")

        s = missing.to_series()

        diff = s.diff()

        gap_breaks = diff[diff > pd.Timedelta(hours=1)]

        print(f"Gap blocks: {len(gap_breaks)}")

# -------------------------------------------------------------------
# Common intersection
# -------------------------------------------------------------------

print("\n")
print("=" * 100)
print("GLOBAL INTERSECTION")
print("=" * 100)

common = btc_index

for asset, df in raw.items():
    common = common.intersection(df.index)

print(f"BTC rows              : {len(btc_index):,}")
print(f"Common rows           : {len(common):,}")
print(f"Overlap %             : {100*len(common)/len(btc_index):.2f}%")

print(f"Common start          : {common.min()}")
print(f"Common end            : {common.max()}")

# -------------------------------------------------------------------
# What each asset loses after alignment
# -------------------------------------------------------------------

print("\n")
print("=" * 100)
print("ROWS REMOVED DURING ALIGNMENT")
print("=" * 100)

for asset, df in raw.items():

    removed = df.index.difference(common)

    print(f"\n{asset}")
    print("-"*60)
    print(f"Removed rows : {len(removed):,}")

    if len(removed):

        print("First 20 removed:")

        for ts in removed[:20]:
            print(ts)

# -------------------------------------------------------------------
# Check aligned files
# -------------------------------------------------------------------

print("\n")
print("=" * 100)
print("CHECKING ALIGNED FILES")
print("=" * 100)

aligned = {}

for asset in ASSETS:

    path = PROCESSED_DIR / f"{asset}_aligned.parquet"

    if path.exists():

        aligned[asset] = pd.read_parquet(path)

if aligned:

    ref = next(iter(aligned.values())).index

    print()

    for asset, df in aligned.items():

        ok = ref.equals(df.index)

        print(f"{asset:12} identical index : {ok}")

        if not ok:

            miss = ref.difference(df.index)

            extra = df.index.difference(ref)

            print("Missing:")
            print(miss[:20])

            print("Extra:")
            print(extra[:20])

print("\nDone.")