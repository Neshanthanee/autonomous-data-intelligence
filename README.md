# Autonomous Data Intelligence System

An end-to-end machine learning system for retail sales forecasting, explainable AI, and automated business intelligence.

The system processes historical retail sales data, generates demand forecasts using XGBoost, validates model performance through rolling backtesting, explains predictions using SHAP, and converts model outputs into business alerts and recommendations.

## Overview

Retail businesses need accurate demand forecasts to plan inventory, staffing, and daily operations.

This project builds an automated pipeline that connects data preparation, machine learning, explainability, business intelligence, API services, and a dashboard into a single system.

The workflow is:

Historical Sales Data
        |
        v
Data Preparation
        |
        v
Feature Engineering
        |
        v
Baseline + XGBoost
        |
        v
Rolling Backtesting
        |
        v
Best Model
        |
        v
SHAP Explainability
        |
        v
Demand Intelligence
        |
        v
FastAPI
        |
        v
Streamlit Dashboard


## Key Features

* Real-world retail sales dataset
* Automated data preparation and feature engineering
* Seasonal-naive baseline comparison
* XGBoost sales forecasting
* Chronological train/test evaluation
* Expanding-window rolling backtesting
* SHAP-based explainable AI
* Demand classification
* Automated alerts
* Business recommendations
* FastAPI backend
* Streamlit dashboard
* Deployment-ready pipeline

## Dataset

The project uses the Rossmann Store Sales dataset.

The dataset contains historical sales information from 1,115 stores, together with store characteristics and business-related variables.

The main training data contains:

* Store
* Date
* Sales
* Customers
* Open
* Promo
* StateHoliday
* SchoolHoliday

Store information includes:

* StoreType
* Assortment
* CompetitionDistance
* CompetitionOpenSinceMonth
* CompetitionOpenSinceYear
* Promo2
* Promo2SinceWeek
* Promo2SinceYear
* PromoInterval

The dataset covers:

2013-01-01 to 2015-07-31

The raw dataset is not included in this repository.

## Data Pipeline

The ETL process combines sales records with store-level information and creates calendar features.

The resulting dataset contains:

* Year
* Month
* Day
* Day of week
* Weekend indicator
* Store information
* Business indicators
* Historical sales information

The main ETL script is:


etl/prepare.py

A smaller deployment dataset is generated using:


etl/prepare_deployment.py


## Forecasting Model

The forecasting system uses historical sales, calendar information, business indicators, and store characteristics.

The model features include:


day_of_week
day_of_month
month
year
is_weekend
lag_1
lag_7
rolling_7_mean
rolling_14_mean
promo
is_state_holiday
school_holiday
open
store_type
assortment
competition_distance
promo2

Historical sales features are generated using lagged and rolling values.

The system uses recursive forecasting for future dates, where previously generated predictions are used as historical inputs for subsequent forecast steps.

## Baseline Model

A seasonal-naive baseline is used for comparison.

The baseline predicts sales using the sales value from the corresponding previous weekly observation.

This provides a simple benchmark for evaluating whether the machine learning model provides meaningful improvement.

## XGBoost Performance

The initial chronological evaluation was performed using Store 1.

Results:

| Model          |     MAE |    RMSE |
| -------------- | ------: | ------: |
| Seasonal Naive | 1209.49 | 1696.92 |
| XGBoost        |  297.99 |  405.70 |

XGBoost achieved a:


75.36% MAE improvement


over the baseline on the chronological holdout period.

## Rolling Backtesting

A single train/test split can give an incomplete view of time-series model performance.

To improve evaluation, the project uses expanding-window rolling backtesting.

Four historical folds were evaluated.

| Fold | Baseline MAE | XGBoost MAE | Improvement |
| ---- | -----------: | ----------: | ----------: |
| 1    |      1151.82 |      367.78 |      68.07% |
| 2    |      1120.55 |      400.50 |      64.26% |
| 3    |      1212.30 |      461.87 |      61.90% |
| 4    |      1513.29 |      351.60 |      76.77% |

Average results:

| Model          | Average MAE | Average RMSE |
| -------------- | ----------: | -----------: |
| Seasonal Naive |     1249.49 |      1575.44 |
| XGBoost        |      395.44 |       547.45 |

Average MAE improvement:


67.75%

XGBoost consistently outperformed the seasonal-naive baseline across all four evaluation folds for Store 1 open-day demand.

## Explainable AI

SHAP is used to explain individual model predictions.

The system identifies which features contributed positively or negatively to a prediction.

Positive SHAP contribution
        |
        v
Pushes predicted sales higher

Negative SHAP contribution
        |
        v
Pushes predicted sales lower


The latest historical prediction for Store 1 produced the following major contributions:

| Feature        | SHAP Contribution |
| -------------- | ----------------: |
| open           |           +666.43 |
| promo          |           +434.39 |
| day_of_month   |           +325.77 |
| day_of_week    |           +142.59 |
| lag_1          |           -110.77 |
| year           |            -63.40 |
| school_holiday |            -56.17 |

The SHAP pipeline is implemented in:

models/xai.py


## Business Intelligence

The forecasting output is converted into business-level intelligence.

The system compares predicted sales against historical open-day demand statistics and classifies future days as:

* High Demand
* Low Demand
* Normal
* Closed

The system also generates:

* Alerts
* Recommendations
* Explanations

Example:


Low Demand

Recommendation:
Consider reducing inventory levels and staffing requirements.

This logic is implemented in:


models/intelligence.py

## API

FastAPI provides the backend service for the system.

Available endpoints:

GET /
GET /health
GET /status
GET /forecast
GET /intelligence
GET /explanation

The API automatically executes the forecasting, explainability, and intelligence pipeline during application startup.

The backend is implemented in:

api.py

FastAPI documentation is available locally at:

http://127.0.0.1:8000/docs

## Dashboard

The Streamlit dashboard provides a user-facing interface for the intelligence system.

It displays:

* 7-day sales forecast
* Forecast overview
* Demand status
* Automated alerts
* Business recommendations
* SHAP explanations

The dashboard is implemented in:


dashboard.py


## Project Structure

autonomous_data_intelligence/
│
├── data/
│   └── deployment/
│       └── store1_history.csv
│
├── etl/
│   ├── prepare.py
│   └── prepare_deployment.py
│
├── models/
│   ├── backtest.py
│   ├── forecast.py
│   ├── intelligence.py
│   ├── predict.py
│   ├── rossmann_xgb.json
│   └── xai.py
│
├── api.py
├── dashboard.py
├── requirements.txt
├── .gitignore
└── README.md

## Installation

Clone the repository:

git clone <repository-url>
cd autonomous_data_intelligence

Create a virtual environment:

python -m venv venv

Activate the environment.

Windows:

venv\Scripts\activate

Install dependencies:


pip install -r requirements.txt


## Running the System

Start the FastAPI backend:


uvicorn api:app --reload


The API will be available at:

http://127.0.0.1:8000

Start the Streamlit dashboard in another terminal:

streamlit run dashboard.py

The dashboard will be available at:

http://localhost:8501

The FastAPI application executes the forecasting pipeline automatically during startup.

## Deployment

The application is structured for cloud deployment using separate backend and frontend services.

The deployment architecture is:

Streamlit Dashboard
        |
        v
FastAPI Backend
        |
        v
Forecasting Pipeline
        |
        +---- XGBoost
        |
        +---- SHAP
        |
        +---- Business Intelligence

A small Store 1 historical dataset is included under:


data/deployment/

The full raw and processed datasets are excluded from the repository.

## Important Deployment Note

The underlying Rossmann dataset contains historical data ending in 2015.

Therefore, the deployed application demonstrates the complete forecasting and intelligence workflow, but it should not be interpreted as a real-time retail sales monitoring system.

Future business inputs such as promotions, holidays, and store operating status are currently provided as forecast inputs for demonstration purposes.

A production implementation would connect these inputs to live business systems or databases.

## Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Explainable AI

* SHAP

### Backend

* FastAPI
* Uvicorn

### Dashboard

* Streamlit

## Future Improvements

Potential extensions include:

* Multi-store forecasting
* Automated database ingestion
* Live business input integration
* Automated model retraining
* Forecast confidence intervals
* Model monitoring
* Data drift detection
* Production authentication
* Cloud database integration
* Real-time operational data

## Project Objective

The goal of this project is not only to predict sales, but to demonstrate how machine learning can be integrated into an end-to-end decision-support system.

The system connects:

Data
→ Prediction
→ Validation
→ Explanation
→ Intelligence
→ Decision Support


This project demonstrates practical machine learning engineering, time-series forecasting, explainable AI, API development, and dashboard development.