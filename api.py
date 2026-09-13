import os
import sys
import subprocess

from contextlib import asynccontextmanager

import pandas as pd

from fastapi import FastAPI, HTTPException


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# DATA AND MODEL PATHS
# ============================================================

DATA_PATH = os.getenv(
    "DATA_PATH",
    os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "rossmann_merged.csv"
    )
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(
        BASE_DIR,
        "models",
        "rossmann_xgb.json"
    )
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILE PATHS
# ============================================================

FORECAST_PATH = os.path.join(
    PROCESSED_DIR,
    "forecast.csv"
)

INTELLIGENCE_PATH = os.path.join(
    PROCESSED_DIR,
    "intelligence.csv"
)

SHAP_PATH = os.path.join(
    PROCESSED_DIR,
    "shap_explanation.csv"
)


# ============================================================
# RUN PIPELINE SCRIPT
# ============================================================

def run_pipeline_script(script_name, pipeline_env):

    script_path = os.path.join(
        BASE_DIR,
        "models",
        script_name
    )

    result = subprocess.run(
        [
            sys.executable,
            script_path
        ],
        env=pipeline_env,
        cwd=BASE_DIR,
        text=True,
        capture_output=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:

        raise RuntimeError(
            f"{script_name} failed with exit code "
            f"{result.returncode}"
        )


# ============================================================
# RUN MACHINE LEARNING PIPELINE
# ============================================================

def run_pipeline():

    print()
    print("==========================================")
    print("STARTING AUTONOMOUS PIPELINE")
    print("==========================================")

    print(
        "Data source:",
        DATA_PATH
    )

    print(
        "Model:",
        MODEL_PATH
    )


    pipeline_env = os.environ.copy()

    pipeline_env["DATA_PATH"] = DATA_PATH
    pipeline_env["MODEL_PATH"] = MODEL_PATH


    # ========================================================
    # FORECAST
    # ========================================================

    print()
    print("Running forecast...")

    run_pipeline_script(
        "forecast.py",
        pipeline_env
    )


    # ========================================================
    # XAI
    # ========================================================

    print()
    print("Running XAI...")

    run_pipeline_script(
        "xai.py",
        pipeline_env
    )


    # ========================================================
    # INTELLIGENCE
    # ========================================================

    print()
    print("Running intelligence...")

    run_pipeline_script(
        "intelligence.py",
        pipeline_env
    )


    print()
    print("==========================================")
    print("AUTONOMOUS PIPELINE COMPLETED")
    print("==========================================")


# ============================================================
# APPLICATION STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print()
    print("==========================================")
    print("APPLICATION STARTUP")
    print("==========================================")

    try:

        run_pipeline()

    except Exception as error:

        print()
        print("==========================================")
        print("PIPELINE ERROR")
        print("==========================================")

        print(error)

    yield


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Autonomous Data Intelligence API",

    description=(
        "Sales forecasting, explainable AI "
        "and automated business intelligence API"
    ),

    version="1.0.0",

    lifespan=lifespan
)


# ============================================================
# HOME / ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "system":
            "Autonomous Data Intelligence",

        "status":
            "running",

        "version":
            "1.0.0"

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy"

    }


# ============================================================
# FORECAST ENDPOINT
# ============================================================

@app.get("/forecast")
def get_forecast():

    if not os.path.exists(
        FORECAST_PATH
    ):

        raise HTTPException(

            status_code=404,

            detail=
                "Forecast file not found."

        )


    forecast = pd.read_csv(
        FORECAST_PATH
    )


    return forecast.to_dict(
        orient="records"
    )


# ============================================================
# INTELLIGENCE ENDPOINT
# ============================================================

@app.get("/intelligence")
def get_intelligence():

    if not os.path.exists(
        INTELLIGENCE_PATH
    ):

        raise HTTPException(

            status_code=404,

            detail=
                "Intelligence file not found."

        )


    intelligence = pd.read_csv(
        INTELLIGENCE_PATH
    )


    return intelligence.to_dict(
        orient="records"
    )


# ============================================================
# XAI / SHAP ENDPOINT
# ============================================================

@app.get("/explanation")
def get_explanation():

    if not os.path.exists(
        SHAP_PATH
    ):

        raise HTTPException(

            status_code=404,

            detail=
                "SHAP explanation file not found."

        )


    shap_data = pd.read_csv(
        SHAP_PATH
    )


    return shap_data.to_dict(
        orient="records"
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

@app.get("/status")
def get_status():

    return {

        "forecast_available":
            os.path.exists(
                FORECAST_PATH
            ),

        "intelligence_available":
            os.path.exists(
                INTELLIGENCE_PATH
            ),

        "xai_available":
            os.path.exists(
                SHAP_PATH
            ),

        "data_source":
            DATA_PATH

    }