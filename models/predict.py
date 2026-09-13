import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# LOAD DATA
# ============================================================

FILE_PATH = "data/processed/rossmann_merged.csv"

df = pd.read_csv(
    FILE_PATH,
    low_memory=False
)

df["Date"] = pd.to_datetime(df["Date"])

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
# CALENDAR FEATURES
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


# ============================================================
# HISTORICAL SALES FEATURES
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
# ROSSMANN BUSINESS FEATURES
# ============================================================

store["promo"] = (
    store["Promo"]
    .astype(int)
)

store["state_holiday"] = (
    store["StateHoliday"]
    .astype(str)
    .str.strip()
)

store["is_state_holiday"] = (
    store["state_holiday"] != "0"
).astype(int)

store["school_holiday"] = (
    store["SchoolHoliday"]
    .astype(int)
)

store["open"] = (
    store["Open"]
    .astype(int)
)


# ============================================================
# STORE INFORMATION
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

store["competition_distance"] = (
    store["CompetitionDistance"]
    .fillna(
        store["CompetitionDistance"].median()
    )
)

store["promo2"] = (
    store["Promo2"]
    .astype(int)
)


# ============================================================
# REMOVE INITIAL ROWS WITHOUT HISTORY
# ============================================================

store = store.dropna(
    subset=[
        "lag_1",
        "lag_7",
        "rolling_7_mean",
        "rolling_14_mean"
    ]
).reset_index(drop=True)


print()
print(
    "Records after feature creation:",
    len(store)
)


# ============================================================
# FEATURES
# ============================================================

features = [

    # Calendar
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "is_weekend",

    # Historical sales
    "lag_1",
    "lag_7",
    "rolling_7_mean",
    "rolling_14_mean",

    # Business factors
    "promo",
    "is_state_holiday",
    "school_holiday",
    "open",

    # Store information
    "store_type",
    "assortment",
    "competition_distance",
    "promo2"
]


X = store[features]

y = store["Sales"]


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(store) * 0.80
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print()
print("==========================================")
print("TRAIN / TEST SPLIT")
print("==========================================")

print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)

print(
    "Training period:",
    store["Date"].iloc[0].date(),
    "to",
    store["Date"].iloc[split_index - 1].date()
)

print(
    "Testing period:",
    store["Date"].iloc[split_index].date(),
    "to",
    store["Date"].iloc[-1].date()
)


# ============================================================
# BASELINE
# ============================================================

print()
print("==========================================")
print("BASELINE MODEL")
print("==========================================")

baseline_predictions = (
    store["lag_7"]
    .iloc[split_index:]
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)


print(
    "Baseline MAE :",
    round(baseline_mae, 2)
)

print(
    "Baseline RMSE:",
    round(baseline_rmse, 2)
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print()
print("==========================================")
print("TRAINING XGBOOST")
print("==========================================")

model = XGBRegressor(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42
)


model.fit(
    X_train,
    y_train
)


print("XGBoost training completed.")


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# EVALUATION
# ============================================================

xgb_mae = mean_absolute_error(
    y_test,
    predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


# ============================================================
# MODEL RESULTS
# ============================================================

print()
print("==========================================")
print("MODEL EVALUATION")
print("==========================================")

print()
print("BASELINE")

print(
    "MAE :",
    round(baseline_mae, 2)
)

print(
    "RMSE:",
    round(baseline_rmse, 2)
)


print()
print("XGBOOST")

print(
    "MAE :",
    round(xgb_mae, 2)
)

print(
    "RMSE:",
    round(xgb_rmse, 2)
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print()
print("==========================================")
print("MODEL COMPARISON")
print("==========================================")


if xgb_mae < baseline_mae:

    improvement = (
        (baseline_mae - xgb_mae)
        / baseline_mae
    ) * 100

    print("🏆 XGBoost performs better.")

    print(
        f"Improvement: {improvement:.2f}%"
    )

else:

    difference = (
        (xgb_mae - baseline_mae)
        / baseline_mae
    ) * 100

    print("📌 Baseline performs better.")

    print(
        f"XGBoost is {difference:.2f}% worse."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature": features,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print()
print("==========================================")
print("FEATURE IMPORTANCE")
print("==========================================")

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# TRAIN FINAL PRODUCTION MODEL
# ============================================================

print()
print("==========================================")
print("TRAINING FINAL PRODUCTION MODEL")
print("==========================================")

# Now that we have evaluated the model,
# train a final model using ALL available
# historical records.

final_model = XGBRegressor(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42
)


final_model.fit(
    X,
    y
)


print(
    "Final model trained on all records."
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

MODEL_PATH = "models/rossmann_xgb.json"

final_model.save_model(
    MODEL_PATH
)


print()
print("==========================================")
print("MODEL SAVED")
print("==========================================")

print(
    "Saved to:",
    MODEL_PATH
)


# ============================================================
# COMPLETED
# ============================================================

print()
print("==========================================")
print("FORECASTING MODEL COMPLETED")
print("==========================================")

print(
    "✅ Evaluation completed."
)

print(
    "✅ Final production model created."
)

print(
    "✅ Model saved successfully."
)