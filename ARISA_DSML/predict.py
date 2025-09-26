"""Run prediction on test data."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import nannyml as nml
import pandas as pd
import shap
from catboost import CatBoostClassifier
from loguru import logger
from mlflow.client import MlflowClient

from ARISA_DSML.config import FIGURES_DIR, MODEL_NAME, MODELS_DIR, PROCESSED_DATA_DIR, target
from ARISA_DSML.resolve import get_model_by_alias


def plot_shap(model: CatBoostClassifier, df_plot: pd.DataFrame) -> None:
    """Plot model shapley overview plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_plot)

    shap.summary_plot(shap_values, df_plot, show=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "test_shap_overall.png")
    plt.close()


def predict(model: CatBoostClassifier, df_pred: pd.DataFrame, params: dict, probs=False) -> Path:
    """Do predictions on test data."""

    feature_columns = params.get("feature_columns")
    if feature_columns is None:
        logger.warning("No feature_columns in params, using all columns except target")
        feature_columns = [col for col in df_pred.columns if col != target]

    # Walidacja czy wszystkie kolumny istnieją
    missing_cols = [col for col in feature_columns if col not in df_pred.columns]
    if missing_cols:
        logger.error(f"Missing columns in prediction data: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in df_pred.columns]

    preds = model.predict(df_pred[feature_columns])

    result_df = pd.DataFrame()
    result_df[target] = preds

    if probs:
        result_df["predicted_probability"] = [
            p[1] for p in model.predict_proba(df_pred[feature_columns])
        ]

    plot_shap(model, df_pred[feature_columns])

    preds_path = MODELS_DIR / "preds.csv"
    result_df.to_csv(preds_path, index=False)

    return preds_path


if __name__ == "__main__":
    df_test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    # Konwersja targetu jeśli potrzeba
    if target in df_test.columns and df_test[target].dtype == "object":
        df_test[target] = (df_test[target] == "Yes").astype(int)

    client = MlflowClient(mlflow.get_tracking_uri())
    model_info = get_model_by_alias(client, alias="champion")
    if model_info is None:
        logger.info("No champion model, predicting using newest model")
        model_info = client.get_latest_versions(MODEL_NAME)[0]

    run_data_dict = client.get_run(model_info.run_id).data.to_dictionary()
    run = client.get_run(model_info.run_id)
    log_model_meta = json.loads(run.data.tags["mlflow.log-model.history"])

    _, artifact_folder = os.path.split(model_info.source)
    logger.info(f"Artifact folder: {artifact_folder}")
    model_uri = f"runs:/{model_info.run_id}/{artifact_folder}"
    logger.info(f"Model URI: {model_uri}")
    loaded_model = mlflow.catboost.load_model(model_uri)

    # Download artifacts
    client.download_artifacts(model_info.run_id, "udc.pkl", str(MODELS_DIR))
    client.download_artifacts(model_info.run_id, "estimator.pkl", str(MODELS_DIR))

    store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
    udc = store.load(filename="udc.pkl", as_type=nml.UnivariateDriftCalculator)
    estimator = store.load(filename="estimator.pkl", as_type=nml.CBPE)

    params = run_data_dict["params"].copy()
    params["feature_columns"] = [
        inp["name"] for inp in json.loads(log_model_meta[0]["signature"]["inputs"])
    ]

    preds_path = predict(loaded_model, df_test, params, probs=True)

    df_preds = pd.read_csv(preds_path)

    analysis_df = df_test.copy()
    analysis_df["prediction"] = df_preds[target].values
    analysis_df["predicted_probability"] = df_preds["predicted_probability"].values

    from ARISA_DSML.helpers import get_git_commit_hash

    git_hash = get_git_commit_hash()
    mlflow.set_experiment("heart_disease_predictions")

    with mlflow.start_run(tags={"git_sha": git_hash}):
        estimated_performance = estimator.estimate(analysis_df)
        fig1 = estimated_performance.plot()
        mlflow.log_figure(fig1, "estimated_performance.png")
        plt.close()

        drift_df = analysis_df.drop(columns=["prediction", "predicted_probability"], axis=1)
        if target in drift_df.columns:
            drift_df = drift_df.drop(columns=[target])

        univariate_drift = udc.calculate(drift_df)
        plot_col_names = drift_df.columns.tolist()

        for p in plot_col_names:
            try:
                fig2 = univariate_drift.filter(column_names=[p]).plot()
                mlflow.log_figure(fig2, f"univariate_drift_{p}.png")
                plt.close()

                fig3 = univariate_drift.filter(period="analysis", column_names=[p]).plot(
                    kind="distribution"
                )
                mlflow.log_figure(fig3, f"univariate_drift_dist_{p}.png")
                plt.close()
            except Exception as e:
                logger.info(f"Failed to plot univariate drift for {p}: {str(e)}")

        mlflow.log_params({"git_hash": git_hash})
