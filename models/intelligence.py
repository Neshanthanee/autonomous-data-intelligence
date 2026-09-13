import os
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# FILE PATHS
# ============================================================

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

FORECAST_PATH = os.path.join(
    PROCESSED_DIR,
    "forecast.csv"
)

SHAP_PATH = os.path.join(
    PROCESSED_DIR,
    "shap_explanation.csv"
)

SALES_PATH = os.getenv(
    "DATA_PATH",
    os.path.join(
        PROCESSED_DIR,
        "rossmann_merged.csv"
    )
)

if not os.path.isabs(SALES_PATH):

    SALES_PATH = os.path.join(
        BASE_DIR,
        SALES_PATH
    )

OUTPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "intelligence.csv"
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("==========================================")
print("LOADING DATA")
print("==========================================")

print(
    f"Forecast path: {FORECAST_PATH}"
)

print(
    f"SHAP path: {SHAP_PATH}"
)

print(
    f"Sales path: {SALES_PATH}"
)

forecast = pd.read_csv(
    FORECAST_PATH
)

shap_data = pd.read_csv(
    SHAP_PATH
)

sales = pd.read_csv(
    SALES_PATH,
    dtype={"StateHoliday": str},
    low_memory=False
)


# ============================================================
# DATE CONVERSION
# ============================================================

def parse_dates(series):

    values = series.astype(str).str.strip()

    dates = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    iso_mask = values.str.match(
        r"^\d{4}-\d{2}-\d{2}$"
    )

    dates.loc[iso_mask] = pd.to_datetime(
        values.loc[iso_mask],
        format="%Y-%m-%d"
    )

    dates.loc[~iso_mask] = pd.to_datetime(
        values.loc[~iso_mask],
        format="%d-%m-%Y"
    )

    return dates


forecast["Date"] = parse_dates(
    forecast["Date"]
)

sales["Date"] = parse_dates(
    sales["Date"]
)

print(
    f"Forecast records: {len(forecast)}"
)

print(
    f"Sales records: {len(sales)}"
)


# ============================================================
# SELECT STORE
# ============================================================

STORE_ID = 1

store_sales = sales[
    sales["Store"] == STORE_ID
].copy()


print(
    f"Store: {STORE_ID}"
)

print(
    f"Historical store records: "
    f"{len(store_sales)}"
)


# ============================================================
# HISTORICAL SALES ANALYSIS
# ============================================================

open_sales = store_sales[
    store_sales["Open"] == 1
]["Sales"].dropna()


average_sales = open_sales.mean()

std_sales = open_sales.std()


high_threshold = (
    average_sales + std_sales
)

low_threshold = max(
    average_sales - std_sales,
    0
)


print()
print("==========================================")
print("HISTORICAL SALES ANALYSIS")
print("==========================================")

print(
    f"Average open-day sales: "
    f"{average_sales:.2f}"
)

print(
    f"Sales standard deviation: "
    f"{std_sales:.2f}"
)

print(
    f"High-demand threshold: "
    f"{high_threshold:.2f}"
)

print(
    f"Low-demand threshold: "
    f"{low_threshold:.2f}"
)


# ============================================================
# FORECAST ANALYSIS
# ============================================================

forecast["DemandStatus"] = "Normal"


forecast.loc[
    forecast["PredictedSales"] >= high_threshold,
    "DemandStatus"
] = "High Demand"


forecast.loc[
    forecast["PredictedSales"] <= low_threshold,
    "DemandStatus"
] = "Low Demand"


forecast.loc[
    forecast["Open"] == 0,
    "DemandStatus"
] = "Closed"


# ============================================================
# FIND TOP SHAP CONTRIBUTORS
# ============================================================

shap_data["abs_shap"] = (
    shap_data["shap_value"]
    .abs()
)


shap_data = (
    shap_data
    .sort_values(
        "abs_shap",
        ascending=False
    )
)


top_positive = shap_data[
    shap_data["shap_value"] > 0
].head(3)


top_negative = shap_data[
    shap_data["shap_value"] < 0
].head(3)


# ============================================================
# BUILD EXPLANATION
# ============================================================

positive_reasons = []


for _, row in top_positive.iterrows():

    positive_reasons.append(
        f"{row['feature']} "
        f"(+{row['shap_value']:.2f})"
    )


negative_reasons = []


for _, row in top_negative.iterrows():

    negative_reasons.append(
        f"{row['feature']} "
        f"({row['shap_value']:.2f})"
    )


positive_text = ", ".join(
    positive_reasons
)

negative_text = ", ".join(
    negative_reasons
)


# ============================================================
# INTELLIGENCE GENERATION
# ============================================================

intelligence = []


for _, row in forecast.iterrows():

    date = row["Date"].strftime(
        "%Y-%m-%d"
    )

    predicted = row["PredictedSales"]

    status = row["DemandStatus"]


    # --------------------------------------------------------
    # CLOSED
    # --------------------------------------------------------

    if status == "Closed":

        alert = "Store Closed"

        recommendation = (
            "Store is closed. "
            "No sales preparation is required."
        )

        explanation = (
            "The forecast indicates zero sales "
            "because the store is expected to be closed."
        )


    # --------------------------------------------------------
    # HIGH DEMAND
    # --------------------------------------------------------

    elif status == "High Demand":

        alert = "High Demand"

        recommendation = (
            "Increase inventory and ensure "
            "sufficient staff availability."
        )

        explanation = (
            "Forecasted demand is significantly "
            "above the historical average."
        )


    # --------------------------------------------------------
    # LOW DEMAND
    # --------------------------------------------------------

    elif status == "Low Demand":

        alert = "Low Demand"

        recommendation = (
            "Consider reducing inventory levels "
            "and staffing requirements."
        )

        explanation = (
            "Forecasted demand is significantly "
            "below the historical average."
        )


    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    else:

        alert = "No Alert"

        recommendation = (
            "Maintain normal inventory "
            "and staffing levels."
        )

        explanation = (
            "Forecasted demand is within the normal "
            "historical range."
        )


    intelligence.append({

        "Date": date,

        "PredictedSales": round(
            predicted,
            2
        ),

        "DemandStatus": status,

        "Alert": alert,

        "Recommendation": recommendation,

        "Explanation": explanation,

        "TopPositiveFactors": positive_text,

        "TopNegativeFactors": negative_text

    })


# ============================================================
# CREATE INTELLIGENCE DATAFRAME
# ============================================================

intelligence_df = pd.DataFrame(
    intelligence
)


# ============================================================
# DISPLAY INTELLIGENCE
# ============================================================

print()
print("==========================================")
print("FORECAST INTELLIGENCE")
print("==========================================")

print(
    intelligence_df[
        [
            "Date",
            "PredictedSales",
            "DemandStatus"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# DISPLAY ALERTS
# ============================================================

print()
print("==========================================")
print("AUTOMATED ALERTS")
print("==========================================")


alerts = intelligence_df[
    intelligence_df["Alert"] != "No Alert"
]


if alerts.empty:

    print(
        "No alerts detected."
    )

else:

    for _, row in alerts.iterrows():

        print(
            f"{row['Date']} | "
            f"{row['Alert']} | "
            f"{row['Recommendation']}"
        )


# ============================================================
# DISPLAY INTELLIGENT RECOMMENDATIONS
# ============================================================

print()
print("==========================================")
print("BUSINESS RECOMMENDATIONS")
print("==========================================")


for _, row in intelligence_df.iterrows():

    print(
        f"{row['Date']} | "
        f"{row['Recommendation']}"
    )


# ============================================================
# DISPLAY XAI INFORMATION
# ============================================================

print()
print("==========================================")
print("XAI INSIGHTS")
print("==========================================")

print(
    "Top factors increasing prediction:"
)


for _, row in top_positive.iterrows():

    print(
        f"+ {row['feature']} "
        f"({row['shap_value']:.2f})"
    )


print()
print(
    "Top factors decreasing prediction:"
)


for _, row in top_negative.iterrows():

    print(
        f"- {row['feature']} "
        f"({row['shap_value']:.2f})"
    )


# ============================================================
# SUMMARY
# ============================================================

high_demand_days = (
    intelligence_df["DemandStatus"]
    == "High Demand"
).sum()


low_demand_days = (
    intelligence_df["DemandStatus"]
    == "Low Demand"
).sum()


closed_days = (
    intelligence_df["DemandStatus"]
    == "Closed"
).sum()


print()
print("==========================================")
print("INTELLIGENCE SUMMARY")
print("==========================================")

print(
    f"High-demand days : "
    f"{high_demand_days}"
)

print(
    f"Low-demand days  : "
    f"{low_demand_days}"
)

print(
    f"Closed days      : "
    f"{closed_days}"
)


# ============================================================
# SAVE OUTPUT
# ============================================================

intelligence_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("==========================================")
print("INTELLIGENCE OUTPUT SAVED")
print("==========================================")

print(
    f"Saved to: {OUTPUT_PATH}"
)


print()
print("==========================================")
print("AUTONOMOUS INTELLIGENCE COMPLETED")
print("==========================================")