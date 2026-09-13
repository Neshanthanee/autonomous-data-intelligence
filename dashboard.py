import os

import streamlit as st
import requests
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Autonomous Data Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Autonomous Data Intelligence"
)

st.caption(
    "Sales Forecasting • Explainable AI • Business Intelligence"
)


# ============================================================
# LOAD API DATA
# ============================================================

try:

    forecast_response = requests.get(
        f"{API_URL}/forecast",
        timeout=10
    )

    intelligence_response = requests.get(
        f"{API_URL}/intelligence",
        timeout=10
    )

    explanation_response = requests.get(
        f"{API_URL}/explanation",
        timeout=10
    )


    forecast_response.raise_for_status()
    intelligence_response.raise_for_status()
    explanation_response.raise_for_status()


    forecast = pd.DataFrame(
        forecast_response.json()
    )

    intelligence = pd.DataFrame(
        intelligence_response.json()
    )

    explanation = pd.DataFrame(
        explanation_response.json()
    )


except requests.exceptions.RequestException as error:

    st.error(
        "❌ Cannot connect to FastAPI."
    )

    st.info(
        "Make sure the FastAPI server is running "
        "and the API URL is correct."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_forecast = forecast[
    "PredictedSales"
].sum()


low_demand_days = (
    intelligence["DemandStatus"]
    == "Low Demand"
).sum()


high_demand_days = (
    intelligence["DemandStatus"]
    == "High Demand"
).sum()


closed_days = (
    intelligence["DemandStatus"]
    == "Closed"
).sum()


# ============================================================
# KPI CARDS
# ============================================================

st.subheader(
    "📌 Forecast Overview"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "7-Day Forecast",
        f"{total_forecast:,.0f}"
    )


with col2:

    st.metric(
        "Low Demand Days",
        low_demand_days
    )


with col3:

    st.metric(
        "High Demand Days",
        high_demand_days
    )


with col4:

    st.metric(
        "Closed Days",
        closed_days
    )


# ============================================================
# FORECAST CHART
# ============================================================

st.subheader(
    "7-Day Sales Forecast"
)


chart_data = forecast[
    [
        "Date",
        "PredictedSales"
    ]
].copy()


chart_data = chart_data.set_index(
    "Date"
)


st.line_chart(
    chart_data
)


# ============================================================
# FORECAST TABLE
# ============================================================

st.subheader(
    "Forecast Details"
)


forecast_display = intelligence[
    [
        "Date",
        "PredictedSales",
        "DemandStatus"
    ]
].copy()


forecast_display["PredictedSales"] = (
    forecast_display["PredictedSales"]
    .round(2)
)


st.dataframe(
    forecast_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ALERTS
# ============================================================

st.subheader(
    "Automated Alerts"
)


alerts = intelligence[
    intelligence["Alert"] != "No Alert"
]


if alerts.empty:

    st.success(
        "No alerts detected."
    )


else:

    for _, row in alerts.iterrows():

        if row["Alert"] == "Low Demand":

            st.warning(
                f"**{row['Date']} — Low Demand**  \n"
                f"{row['Recommendation']}"
            )


        elif row["Alert"] == "High Demand":

            st.error(
                f"**{row['Date']} — High Demand**  \n"
                f"{row['Recommendation']}"
            )


        elif row["Alert"] == "Store Closed":

            st.info(
                f"**{row['Date']} — Store Closed**  \n"
                f"{row['Recommendation']}"
            )


# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

st.subheader(
    "Business Recommendations"
)


for _, row in intelligence.iterrows():

    st.write(
        f"**{row['Date']}** — "
        f"{row['Recommendation']}"
    )


# ============================================================
# XAI SECTION
# ============================================================

st.subheader(
    " Explainable AI"
)


st.write(
    "Top factors influencing the latest model prediction:"
)


xai_display = explanation[
    [
        "feature",
        "value",
        "shap_value"
    ]
].copy()


xai_display["shap_value"] = (
    xai_display["shap_value"]
    .round(2)
)


st.dataframe(
    xai_display,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Positive SHAP values increase the prediction; "
    "negative SHAP values decrease it."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Autonomous Data Intelligence System • "
    "XGBoost + SHAP + FastAPI"
)