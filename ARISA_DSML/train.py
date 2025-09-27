"""Functions to train model."""

from pathlib import Path

from catboost import CatBoostClassifier, Pool, cv
import joblib
from loguru import logger
import mlflow
from mlflow.client import MlflowClient
import nannyml as nml
import optuna
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split

from ARISA_DSML.config import (
    FIGURES_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    categorical,
    target,
)
from ARISA_DSML.helpers import get_git_commit_hash


def run_hyperopt(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    categorical_features: list[str],
    test_size: float = 0.25,
    n_trials: int = 20,
    overwrite: bool = False,
) -> Path:
    """Run optuna hyperparameter tuning."""
    best_params_path = MODELS_DIR / "best_params.pkl"

    if not best_params_path.is_file() or overwrite:
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42
        )

        def objective(trial: optuna.trial.Trial) -> float:
            with mlflow.start_run(nested=True):
                params = {
                    "depth": trial.suggest_int("depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3),
                    "iterations": trial.suggest_int("iterations", 50, 300),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 1),
                    "random_strength": trial.suggest_float(
                        "random_strength", 1e-5, 100.0, log=True
                    ),
                }
                model = CatBoostClassifier(
                    **params, verbose=0, cat_features=categorical_features, random_seed=42
                )
                model.fit(
                    X_train_opt,
                    y_train_opt,
                    eval_set=(X_val_opt, y_val_opt),
                    early_stopping_rounds=50,
                )
                mlflow.log_params(params)
                preds = model.predict(X_val_opt)
                probs = model.predict_proba(X_val_opt)

                f1 = f1_score(y_val_opt, preds)
                logloss = log_loss(y_val_opt, probs)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("logloss", logloss)

            return model.get_best_score()["validation"]["Logloss"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(study.best_params, best_params_path)

        params = study.best_params
    else:
        params = joblib.load(best_params_path)

    logger.info(f"Best Parameters: {params}")
    return best_params_path


def train_cv(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    categorical_features: list[str],
    params: dict,
    eval_metric: str = "F1",
    n: int = 5,
) -> Path:
    """Do cross-validated training."""
    cv_params = params.copy()
    cv_params["eval_metric"] = eval_metric
    cv_params["loss_function"] = "Logloss"
    cv_params["cat_features"] = categorical_features
    cv_params["random_seed"] = 42

    data = Pool(X_train, y_train, cat_features=categorical_features)

    cv_results = cv(
        params=cv_params,
        pool=data,
        fold_count=n,
        partition_random_seed=42,
        shuffle=True,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cv_output_path = MODELS_DIR / "cv_results.csv"
    cv_results.to_csv(cv_output_path, index=False)

    logger.info(f"CV results saved to {cv_output_path}")
    return cv_output_path


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    categorical_features: list[str],
    params: dict,
    cv_results: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Train final model."""
    log_params = params.copy()
    log_params["eval_metric"] = "F1"
    log_params["loss_function"] = "Logloss"
    log_params["cat_features"] = categorical_features
    log_params["feature_columns"] = X_train.columns.tolist()
    log_params["random_seed"] = 42

    #model = CatBoostClassifier(**log_params, verbose=True)
    params_for_model = {k: v for k, v in log_params.items() if k != "feature_columns"}
    model = CatBoostClassifier(**params_for_model, verbose=True)



    with mlflow.start_run() as run:
        model.fit(
            X_train,
            y_train,
            verbose_eval=50,
            early_stopping_rounds=50,
            use_best_model=False,
            plot=False,
        )

        #MODELS_DIR.mkdir(parents=True, exist_ok=True)
        #model_path = MODELS_DIR / "catboost_model.cbm"
        #model.save_model(model_path)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "catboost_model.cbm"
        model.save_model(model_path)

        # Dodatkowy eksport uproszczony (do łatwego wczytania w predict.py)
        local_model_path = MODELS_DIR / "model.cb"
        model.save_model(local_model_path)
        logger.info(f"Saved local model copy to {local_model_path}")




        mlflow.log_params(log_params)
        mlflow.catboost.log_model(model, "model")

        client = MlflowClient()
        model_info = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=MODEL_NAME,
        )
        client.set_model_version_tag(
            name=model_info.name,
            version=model_info.version,
            key="git_sha",
            value=get_git_commit_hash(),
        )

        model_params_path = MODELS_DIR / "model_params.pkl"
        save_params = log_params.copy()
        joblib.dump(save_params, model_params_path)

        if cv_results is not None:
            fig1 = plot_error_scatter(
                df_plot=cv_results,
                name="Mean F1 Score",
                title="Cross-Validation (N=5) Mean F1 score with Error Bands",
                xtitle="Training Steps",
                ytitle="Performance Score",
                yaxis_range=[0.5, 1.0],
            )
            mlflow.log_figure(fig1, "test-F1-mean_vs_iterations.png")

            fig2 = plot_error_scatter(
                cv_results,
                x="iterations",
                y="test-Logloss-mean",
                err="test-Logloss-std",
                name="Mean logloss",
                title="Cross-Validation (N=5) Mean Logloss with Error Bands",
                xtitle="Training Steps",
                ytitle="Logloss",
            )
            mlflow.log_figure(fig2, "test-logloss-mean_vs_iterations.png")

        # NannyML monitoring
        reference_df = X_train.copy()
        reference_df["prediction"] = model.predict(X_train)
        reference_df["predicted_probability"] = [p[1] for p in model.predict_proba(X_train)]
        reference_df[target] = y_train.values if hasattr(y_train, "values") else y_train
        chunk_size = 5000

        udc = nml.UnivariateDriftCalculator(
            column_names=X_train.columns.tolist(),
            chunk_size=chunk_size,
        )
        udc.fit(reference_df.drop(columns=["prediction", target, "predicted_probability"]))

        estimator = nml.CBPE(
            problem_type="classification_binary",
            y_pred_proba="predicted_probability",
            y_pred="prediction",
            y_true=target,
            metrics=["roc_auc"],
            chunk_size=chunk_size,
        )
        estimator = estimator.fit(reference_df)

        store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
        store.store(udc, filename="udc.pkl")
        store.store(estimator, filename="estimator.pkl")

        mlflow.log_artifact(MODELS_DIR / "udc.pkl")
        mlflow.log_artifact(MODELS_DIR / "estimator.pkl")

    return (model_path, model_params_path)


def plot_error_scatter(
    df_plot: pd.DataFrame,
    x: str = "iterations",
    y: str = "test-F1-mean",
    err: str = "test-F1-std",
    name: str = "",
    title: str = "",
    xtitle: str = "",
    ytitle: str = "",
    yaxis_range: list[float] | None = None,
) -> go.Figure:
    """Plot plotly scatter plots with error areas."""
    fig = go.Figure()

    if not len(name):
        name = y

    fig.add_trace(
        go.Scatter(
            x=df_plot[x],
            y=df_plot[y],
            mode="lines",
            name=name,
            line={"color": "blue"},
        ),
    )

    upper = df_plot[y] + df_plot[err]
    lower = df_plot[y] - df_plot[err]

    fig.add_trace(
        go.Scatter(
            x=pd.concat([df_plot[x], df_plot[x][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line={"color": "rgba(255, 255, 255, 0)"},
            showlegend=False,
        ),
    )

    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template="plotly_white",
    )

    if yaxis_range is not None:
        fig.update_layout(yaxis={"range": yaxis_range})

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(FIGURES_DIR / f"{y}_vs_{x}.png")
    return fig


def get_or_create_experiment(experiment_name: str):
    """Retrieve the ID of an existing MLflow experiment or create a new one."""
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


if __name__ == "__main__":
    df_train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    # Konwersja targetu na int jeśli potrzeba
    if df_train[target].dtype == "object":
        df_train[target] = (df_train[target] == "Yes").astype(int)

    y_train = df_train.pop(target)
    X_train = df_train

    # Walidacja kolumn kategorycznych
    categorical_features = [col for col in categorical if col in X_train.columns]
    logger.info(f"Categorical features: {categorical_features}")

    experiment_id = get_or_create_experiment("heart_disease_hyperparam_tuning")
    mlflow.set_experiment(experiment_id=experiment_id)
    best_params_path = run_hyperopt(X_train, y_train, categorical_features)
    params = joblib.load(best_params_path)

    cv_output_path = train_cv(X_train, y_train, categorical_features, params)
    cv_results = pd.read_csv(cv_output_path)

    experiment_id = get_or_create_experiment("heart_disease_full_training")
    mlflow.set_experiment(experiment_id=experiment_id)
    model_path, model_params_path = train(
        X_train, y_train, categorical_features, params, cv_results=cv_results
    )
