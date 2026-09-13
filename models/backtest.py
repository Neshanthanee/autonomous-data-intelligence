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

store["is_state_holiday"] = (
    store["StateHoliday"]
    .astype(str)
    .str.strip()
    .ne("0")
    .astype(int)
)

store["school_holiday"] = (
    store["SchoolHoliday"]
    .astype(int)
)

store["open"] = (
    store["Open"]
    .astype(int)
)


# ============================================================
# STORE FEATURES
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
# REMOVE INITIAL HISTORY
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
# ROLLING BACKTEST SETTINGS
# ============================================================

folds = [
    (500, 100),
    (600, 100),
    (700, 100),
    (800, 100)
]


results = []


# ============================================================
# ROLLING BACKTEST
# ============================================================

print()
print("==========================================")
print("OPEN-DAY DEMAND BACKTEST")
print("==========================================")

for fold_number, (train_end, test_size) in enumerate(
    folds,
    start=1
):

    test_start = train_end
    test_end = test_start + test_size

    if test_end > len(store):
        break


    # ========================================================
    # TRAIN / TEST DATA
    # ========================================================

    train_data = store.iloc[
        :train_end
    ]

    test_data = store.iloc[
        test_start:test_end
    ]


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    X_train = train_data[features]
    y_train = train_data["Sales"]

    X_test = test_data[features]
    y_test = test_data["Sales"]


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


    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = model.predict(
        X_test
    )


    # ========================================================
    # OPEN DAYS ONLY
    # ========================================================

    open_mask = (
        test_data["Open"] == 1
    )


    open_test = test_data.loc[
        open_mask
    ]

    open_actual = y_test.loc[
        open_mask
    ]

    open_predictions = predictions[
        open_mask.to_numpy()
    ]


    # ========================================================
    # EVALUATION
    # ========================================================

    open_mae = mean_absolute_error(
        open_actual,
        open_predictions
    )

    open_rmse = np.sqrt(
        mean_squared_error(
            open_actual,
            open_predictions
        )
    )


    # ========================================================
    # BASELINE
    # ========================================================

    baseline_predictions = (
        open_test["lag_7"]
    )


    baseline_mae = mean_absolute_error(
        open_actual,
        baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            open_actual,
            baseline_predictions
        )
    )


    # ========================================================
    # IMPROVEMENT
    # ========================================================

    improvement = (
        (baseline_mae - open_mae)
        / baseline_mae
    ) * 100


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("------------------------------------------")
    print("FOLD", fold_number)
    print("------------------------------------------")

    print(
        "Training period:",
        train_data["Date"].min().date(),
        "to",
        train_data["Date"].max().date()
    )

    print(
        "Testing period:",
        test_data["Date"].min().date(),
        "to",
        test_data["Date"].max().date()
    )

    print(
        "Open test days:",
        len(open_test)
    )

    print()

    print(
        "Baseline MAE :",
        round(baseline_mae, 2)
    )

    print(
        "Baseline RMSE:",
        round(baseline_rmse, 2)
    )

    print()

    print(
        "XGBoost MAE :",
        round(open_mae, 2)
    )

    print(
        "XGBoost RMSE:",
        round(open_rmse, 2)
    )

    print()

    print(
        "Improvement:",
        round(improvement, 2),
        "%"
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "fold": fold_number,

        "baseline_mae":
            baseline_mae,

        "baseline_rmse":
            baseline_rmse,

        "xgb_mae":
            open_mae,

        "xgb_rmse":
            open_rmse,

        "improvement":
            improvement,

        "open_days":
            len(open_test)

    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# AVERAGE PERFORMANCE
# ============================================================

print()
print("==========================================")
print("AVERAGE OPEN-DAY PERFORMANCE")
print("==========================================")

avg_baseline_mae = (
    results_df["baseline_mae"].mean()
)

avg_baseline_rmse = (
    results_df["baseline_rmse"].mean()
)

avg_xgb_mae = (
    results_df["xgb_mae"].mean()
)

avg_xgb_rmse = (
    results_df["xgb_rmse"].mean()
)

avg_improvement = (
    results_df["improvement"].mean()
)


print()
print(
    "Baseline average MAE :",
    round(avg_baseline_mae, 2)
)

print(
    "Baseline average RMSE:",
    round(avg_baseline_rmse, 2)
)

print()
print(
    "XGBoost average MAE :",
    round(avg_xgb_mae, 2)
)

print(
    "XGBoost average RMSE:",
    round(avg_xgb_rmse, 2)
)

print()
print(
    "Average improvement:",
    round(avg_improvement, 2),
    "%"
)


# ============================================================
# FINAL DECISION
# ============================================================

print()
print("==========================================")
print("FINAL MODEL DECISION")
print("==========================================")

if avg_xgb_mae < avg_baseline_mae:

    print(
        "🏆 XGBoost is better for open-store demand."
    )

else:

    print(
        "📌 Baseline is better for open-store demand."
    )


print()
print("✅ Open-day demand backtesting completed.")