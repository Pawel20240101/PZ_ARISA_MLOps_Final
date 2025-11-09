"""Model resolution - champion/challenger pattern."""

from loguru import logger
import mlflow
from mlflow.client import MlflowClient

from ARISA_DSML.config import MLFLOW_TRACKING_URI, MODEL_NAME

# MLflow URI configuration
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")


def get_model_by_alias(client, model_name: str = MODEL_NAME, alias: str = "champion"):
    """Get model version by alias from MLflow registry."""
    try:
        alias_mv = client.get_model_version_by_alias(model_name, alias)
        logger.info(f"Found {alias} model: version {alias_mv.version}")
        return alias_mv
    except Exception as e:
        if f"alias {alias} not found" in str(e):
            logger.info(f"No {alias} model found")
            return None
        logger.error(f"Error getting {alias} model: {str(e)}")
        raise e


if __name__ == "__main__":
    logger.info("Starting model resolution process")
    client = MlflowClient(mlflow.get_tracking_uri())

    champ_mv = get_model_by_alias(client)

    if champ_mv is None:
        logger.info("No champion model exists")
        chall_mv = get_model_by_alias(client, alias="challenger")

        if chall_mv is None:
            logger.info("No challenger model exists either")
            try:
                model_info = client.get_latest_versions(MODEL_NAME)[0]
                logger.info(f"Promoting newest model (version {model_info.version}) to champion")
                client.set_registered_model_alias(MODEL_NAME, "champion", model_info.version)
                logger.info("✅ Champion set successfully")
            except IndexError:
                logger.error(f"No models found in registry for {MODEL_NAME}")
                raise Exception(f"No models available to promote for {MODEL_NAME}")
        else:
            logger.info(f"Found challenger (version {chall_mv.version}), " "promoting to champion")
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
            client.set_registered_model_alias(MODEL_NAME, "champion", chall_mv.version)
            logger.info("✅ Challenger promoted to champion")

    # Re-check for challenger after potential promotion
    chall_mv = get_model_by_alias(client, alias="challenger")

    if champ_mv is not None and chall_mv is not None:
        logger.info("Both champion and challenger exist - comparing metrics")

        champ_run = client.get_run(champ_mv.run_id)
        chall_run = client.get_run(chall_mv.run_id)

        # Bezpieczne pobieranie metryk
        try:
            f1_champ = champ_run.data.metrics["f1_cv_mean"]
            f1_chall = chall_run.data.metrics["f1_cv_mean"]
        except KeyError as e:
            logger.error(f"Missing f1_cv_mean metric in model run: {str(e)}")
            logger.error("Make sure train.py logs f1_cv_mean metric!")
            raise

        logger.info(f"Champion F1: {f1_champ:.4f} (version {champ_mv.version})")
        logger.info(f"Challenger F1: {f1_chall:.4f} (version {chall_mv.version})")

        if f1_chall >= f1_champ:
            improvement = ((f1_chall - f1_champ) / f1_champ) * 100
            logger.info(f"✅ Challenger surpassed champion by {improvement:.2f}% - promoting!")
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
            client.set_registered_model_alias(MODEL_NAME, "champion", chall_mv.version)
            logger.info(f"✅ New champion: version {chall_mv.version}")
        else:
            degradation = ((f1_champ - f1_chall) / f1_champ) * 100
            challenge_failed_exc = (
                f"❌ Challenger performs {degradation:.2f}% worse than champion. "
                f"Champion F1: {f1_champ:.4f}, Challenger F1: {f1_chall:.4f}. "
                "Keeping current champion."
            )
            logger.error(challenge_failed_exc)
            raise Exception(challenge_failed_exc)

    elif champ_mv is not None and chall_mv is None:
        logger.info(f"✅ No challenger - continuing with champion (version {champ_mv.version})")
    else:
        logger.warning("Unexpected state in model resolution")

    logger.info("Model resolution completed")
