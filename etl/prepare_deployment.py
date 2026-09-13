import pandas as pd

INPUT_PATH = "data/processed/rossmann_merged.csv"
OUTPUT_PATH = "data/deployment/store1_history.csv"

print("Loading processed Rossmann data...")

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)

# Keep only Store 1
store = (
    df[df["Store"] == 1]
    .copy()
    .sort_values("Date")
    .reset_index(drop=True)
)

# Keep only columns required by the deployed application
columns = [
    "Store",
    "Date",
    "Sales",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2"
]

store = store[columns]

store.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("==========================================")
print("DEPLOYMENT DATA CREATED")
print("==========================================")

print("Store:", 1)
print("Records:", len(store))
print(
    "Date range:",
    store["Date"].min().date(),
    "to",
    store["Date"].max().date()
)

print()
print("Saved to:")
print(OUTPUT_PATH)