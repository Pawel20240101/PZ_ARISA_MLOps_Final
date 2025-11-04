"""Config file for module."""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATASET = "kamilpytlak/personal-key-indicators-of-heart-disease"
DATASET_TEST = None  # brak osobnego test setu

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_NAME = "heart-disease-classifier"

categorical = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

target = "HeartDisease"

# MLflow Configuration
# Priorytet: AWS RDS > MLFLOW_TRACKING_URI z env > localhost
if os.getenv("MLFLOWDBENDPOINT"):
    # AWS RDS PostgreSQL backend
    MLFLOW_TRACKING_URI = (
        f"postgresql://{os.getenv('MLFLOWDBUSERNAME')}:"
        f"{os.getenv('MLFLOWDBPASS')}@"
        f"{os.getenv('MLFLOWDBENDPOINT')}:"
        f"{os.getenv('MLFLOWDBPORT')}/"
        f"{os.getenv('MLFLOWDB')}"
    )
    logger.info("Using AWS RDS as MLflow backend")
else:
    # Lokalny tracking URI
    MLFLOW_TRACKING_URI = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )
    logger.info(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")

# S3 Artifact Store (opcjonalne - dla AWS)
ARTIFACT_BUCKET = os.getenv("ARTIFACT_BUCKET", None)