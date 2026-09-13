import os

import pandas as pd
import shap
from xgboost import XGBRegressor


DATA_PATH = os.getenv(
    "DATA_PATH",
    "data/processed/rossmann_merged.csv"
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/rossmann_xgb.json"
)

OUTPUT_PATH = "data/processed/shap_explanation.csv"


print("==========================================")
print("LOADING DATA")
print("==========================================")

df = pd.read_csv(
    DATA_PATH,
    dtype={"StateHoliday": str},
    low_memory=False
)

df["Date"] = pd.to_datetime(
    df["Date"],
    format="mixed",
    dayfirst=True
)


print()
print("==========================================")
print("PREPARING STORE DATA")
print("==========================================")

STORE_ID = 1

store = (
    df[df["Store"] == STORE_ID]
    .copy()
    .sort_values("Date")
    .reset_index(drop=True)
)

store["Sales"] = store["Sales"].fillna(0)


print(
    f"Store: {STORE_ID}"
)

print(
    f"Records: {len(store)}"
)

print(
    f"Date range: "
    f"{store['Date'].min().date()} "
    f"to "
    f"{store['Date'].max().date()}"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

store["day_of_week"] = (
    store["Date"].dt.dayofweek
)

store["day_of_month"] = (
    store["Date"].dt.day
)

store["month"] = (
    store["Date"].dt.month
)

store["year"] = (
    store["Date"].dt.year
)

store["is_weekend"] = (
    store["day_of_week"] >= 5
).astype(int)


store["lag_1"] = (
    store["Sales"].shift(1)
)

store["lag_7"] = (
    store["Sales"].shift(7)
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


store["promo"] = (
    store["Promo"].astype(int)
)

store["is_state_holiday"] = (
    store["StateHoliday"]
    .astype(str)
    .str.strip()
    .isin(["a", "b", "c", "1"])
    .astype(int)
)

store["school_holiday"] = (
    store["SchoolHoliday"].astype(int)
)

store["open"] = (
    store["Open"].astype(int)
)


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

store["competition_distance"] = (
    store["CompetitionDistance"]
    .fillna(
        store["CompetitionDistance"].median()
    )
)

store["promo2"] = (
    store["Promo2"].astype(int)
)


FEATURES = [
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


model_data = store.dropna(
    subset=FEATURES
).copy()


print()
print(
    f"Records available for SHAP: "
    f"{len(model_data)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("==========================================")
print("LOADING MODEL")
print("==========================================")

model = XGBRegressor()

model.load_model(
    MODEL_PATH
)


# ============================================================
# SHAP EXPLANATION
# ============================================================

print()
print("==========================================")
print("GENERATING SHAP EXPLANATION")
print("==========================================")

X_latest = model_data[
    FEATURES
].iloc[[-1]]


latest_date = model_data[
    "Date"
].iloc[-1]


prediction = model.predict(
    X_latest
)[0]


explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_latest
)


shap_values = shap_values[0]


explanation = pd.DataFrame({

    "feature":
        FEATURES,

    "value":
        X_latest.iloc[0].values,

    "shap_value":
        shap_values

})


explanation = (
    explanation
    .sort_values(
        "shap_value",
        key=lambda x: x.abs(),
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE EXPLANATION
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

explanation.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("==========================================")
print("XAI SUMMARY")
print("==========================================")

print(
    f"Date: "
    f"{latest_date.strftime('%Y-%m-%d')}"
)

print(
    f"Predicted sales: "
    f"{prediction:.2f}"
)

print()
print(
    explanation.to_string(
        index=False
    )
)


print()
print("==========================================")
print("SHAP EXPLANATION SAVED")
print("==========================================")

print(
    f"Saved to: {OUTPUT_PATH}"
)

print()
print("==========================================")
print("XAI COMPLETED")
print("==========================================")