import pandas as pd


# =========================================================
# FILE PATHS
# =========================================================

TRAIN_PATH = "data/raw/train.csv"
STORE_PATH = "data/raw/store.csv"

OUTPUT_PATH = "data/processed/rossmann_merged.csv"


# =========================================================
# LOAD DATA
# =========================================================

print("Loading sales data...")

train = pd.read_csv(
    TRAIN_PATH,
    low_memory=False
)

print("Loading store data...")

store = pd.read_csv(
    STORE_PATH
)


# =========================================================
# BASIC INFORMATION
# =========================================================

print()
print("==========================================")
print("DATASET INFORMATION")
print("==========================================")

print("Sales records :", len(train))
print("Stores        :", train["Store"].nunique())
print("Store records :", len(store))


# =========================================================
# CLEAN COLUMN TYPES
# =========================================================

train["Date"] = pd.to_datetime(
    train["Date"]
)

# Convert StateHoliday to string
train["StateHoliday"] = (
    train["StateHoliday"]
    .astype(str)
    .str.strip()
)


# =========================================================
# MERGE SALES + STORE INFORMATION
# =========================================================

print()
print("Merging sales and store information...")

merged = train.merge(
    store,
    on="Store",
    how="left",
    validate="many_to_one"
)


# =========================================================
# CHECK MERGE
# =========================================================

print()
print("==========================================")
print("MERGE VALIDATION")
print("==========================================")

print("Merged records:", len(merged))

missing_store_data = (
    merged["StoreType"]
    .isna()
    .sum()
)

print(
    "Records with missing store information:",
    missing_store_data
)


# =========================================================
# CREATE CALENDAR FEATURES
# =========================================================

merged["Year"] = (
    merged["Date"].dt.year
)

merged["Month"] = (
    merged["Date"].dt.month
)

merged["Day"] = (
    merged["Date"].dt.day
)

merged["DayOfWeek"] = (
    merged["Date"].dt.dayofweek
)

merged["IsWeekend"] = (
    merged["DayOfWeek"] >= 5
).astype(int)


# =========================================================
# SAVE PROCESSED DATA
# =========================================================

merged.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("==========================================")
print("ROSSMANN ETL COMPLETED")
print("==========================================")

print("Final records:", len(merged))

print(
    "Date range:",
    merged["Date"].min().date(),
    "to",
    merged["Date"].max().date()
)

print(
    "Columns:",
    len(merged.columns)
)

print()
print("Saved to:")
print(OUTPUT_PATH)