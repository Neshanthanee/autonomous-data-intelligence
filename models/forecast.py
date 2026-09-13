import os
import pandas as pd
import numpy as np

from xgboost import XGBRegressor


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = os.getenv(
    "DATA_PATH",
    "data/processed/rossmann_merged.csv"
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/rossmann_xgb.json"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

df["Date"] = pd.to_datetime(
    df["Date"],
    format="ISO8601"
)

print("Data loaded")
print("Total records:", len(df))


# ============================================================
# SELECT STORE
# ============================================================

STORE_ID = 1

store = (
    df[df["Store"] == STORE_ID]
    .copy()
)

store = (
    store
    .sort_values("Date")
    .reset_index(drop=True)
)


print()
print("==========================================")
print("STORE INFORMATION")
print("==========================================")

print("Store:", STORE_ID)
print("Records:", len(store))

print(
    "Date range:",
    store["Date"].min().date(),
    "to",
    store["Date"].max().date()
)


# ============================================================
# HANDLE CLOSED STORES
# ============================================================

store.loc[
    store["Open"] == 0,
    "Sales"
] = 0


# ============================================================
# CREATE HISTORICAL FEATURES
# ============================================================

store["lag_1"] = (
    store["Sales"]
    .shift(1)
)

store["lag_7"] = (
    store["Sales"]
    .shift(7)
)

store["rolling_7_mean"] = (
    store["Sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

store["rolling_14_mean"] = (
    store["Sales"]
    .shift(1)
    .rolling(14)
    .mean()
)


# ============================================================
# REMOVE INITIAL ROWS
# ============================================================

store = store.dropna(
    subset=[
        "lag_1",
        "lag_7",
        "rolling_7_mean",
        "rolling_14_mean"
    ]
).reset_index(drop=True)


# ============================================================
# LOAD CATEGORICAL INFORMATION
# ============================================================

store["store_type"] = (
    store["StoreType"]
    .astype("category")
    .cat.codes
)

store["assortment"] = (
    store["Assortment"]
    .astype("category")
    .cat.codes
)


# ============================================================
# STORE INFORMATION
# ============================================================

competition_distance = (
    store["CompetitionDistance"]
    .fillna(
        store["CompetitionDistance"].median()
    )
)

promo2 = (
    store["Promo2"]
    .astype(int)
)


# ============================================================
# LOAD SAVED MODEL
# ============================================================

print()
print("==========================================")
print("LOADING SAVED MODEL")
print("==========================================")

model = XGBRegressor()

model.load_model(
    MODEL_PATH
)

print(
    "✅ Model loaded from:",
    MODEL_PATH
)


# ============================================================
# FEATURES
# ============================================================

features = [

    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "is_weekend",

    "lag_1",
    "lag_7",
    "rolling_7_mean",
    "rolling_14_mean",

    "promo",
    "is_state_holiday",
    "school_holiday",
    "open",

    "store_type",
    "assortment",
    "competition_distance",
    "promo2"
]


# ============================================================
# FUTURE BUSINESS INPUTS
# ============================================================

last_date = store["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)


# IMPORTANT:
# These are example values.
# Replace them with actual future business information.

future = pd.DataFrame({

    "Date": future_dates,

    "Open": [
        1,
        1,
        1,
        1,
        1,
        1,
        0
    ],

    "Promo": [
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],

    "StateHoliday": [
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],

    "SchoolHoliday": [
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
})


# ============================================================
# STORE CONSTANTS
# ============================================================

future["store_type"] = (
    store["store_type"].iloc[-1]
)

future["assortment"] = (
    store["assortment"].iloc[-1]
)

future["competition_distance"] = (
    competition_distance.iloc[-1]
)

future["promo2"] = (
    promo2.iloc[-1]
)


# ============================================================
# RECURSIVE FORECAST
# ============================================================

print()
print("==========================================")
print("7-DAY FORECAST")
print("==========================================")


# Historical sales used for creating
# recursive lag features.

sales_history = (
    store["Sales"]
    .tolist()
)

predictions = []


for i in range(
    len(future)
):

    current_date = (
        future.loc[i, "Date"]
    )


    # ========================================================
    # CALENDAR FEATURES
    # ========================================================

    day_of_week = (
        current_date.dayofweek
    )

    day_of_month = (
        current_date.day
    )

    month = (
        current_date.month
    )

    year = (
        current_date.year
    )

    is_weekend = int(
        day_of_week >= 5
    )


    # ========================================================
    # HISTORICAL SALES FEATURES
    # ========================================================

    lag_1 = (
        sales_history[-1]
    )

    lag_7 = (
        sales_history[-7]
    )

    rolling_7_mean = np.mean(
        sales_history[-7:]
    )

    rolling_14_mean = np.mean(
        sales_history[-14:]
    )


    # ========================================================
    # BUSINESS FEATURES
    # ========================================================

    promo = int(
        future.loc[i, "Promo"]
    )

    is_state_holiday = int(
        future.loc[i, "StateHoliday"]
    )

    school_holiday = int(
        future.loc[i, "SchoolHoliday"]
    )

    open_status = int(
        future.loc[i, "Open"]
    )


    # ========================================================
    # CREATE INPUT ROW
    # ========================================================

    row = pd.DataFrame([{

        "day_of_week":
            day_of_week,

        "day_of_month":
            day_of_month,

        "month":
            month,

        "year":
            year,

        "is_weekend":
            is_weekend,

        "lag_1":
            lag_1,

        "lag_7":
            lag_7,

        "rolling_7_mean":
            rolling_7_mean,

        "rolling_14_mean":
            rolling_14_mean,

        "promo":
            promo,

        "is_state_holiday":
            is_state_holiday,

        "school_holiday":
            school_holiday,

        "open":
            open_status,

        "store_type":
            future.loc[
                i,
                "store_type"
            ],

        "assortment":
            future.loc[
                i,
                "assortment"
            ],

        "competition_distance":
            future.loc[
                i,
                "competition_distance"
            ],

        "promo2":
            future.loc[
                i,
                "promo2"
            ]

    }])


    # ========================================================
    # PREDICT
    # ========================================================

    prediction = model.predict(
        row[features]
    )[0]


    # Closed store = zero sales

    if open_status == 0:

        prediction = 0


    # Prevent negative prediction

    prediction = max(
        0,
        prediction
    )


    # Save prediction

    predictions.append(
        prediction
    )


    # Add prediction to history
    # for the next recursive step.

    sales_history.append(
        prediction
    )


# ============================================================
# SAVE FORECAST
# ============================================================

future["PredictedSales"] = (
    np.round(
        predictions,
        2
    )
)


OUTPUT_PATH = (
    "data/processed/forecast.csv"
)


future.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY FORECAST
# ============================================================

print()

print(
    future[
        [
            "Date",
            "Open",
            "Promo",
            "StateHoliday",
            "SchoolHoliday",
            "PredictedSales"
        ]
    ].to_string(index=False)
)


# ============================================================
# TOTAL FORECAST
# ============================================================

total_forecast = (
    future["PredictedSales"]
    .sum()
)


print()

print(
    "7-day predicted sales:",
    round(
        total_forecast,
        2
    )
)


# ============================================================
# FORECAST SAVED
# ============================================================

print()
print("==========================================")
print("FORECAST SAVED")
print("==========================================")

print(
    "Saved to:",
    OUTPUT_PATH
)


print()
print("==========================================")
print("FORECAST COMPLETED")
print("==========================================")