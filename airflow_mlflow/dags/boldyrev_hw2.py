import os
import logging
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MY_NAME = "Dmitrii"
MY_SURNAME = "Boldyrev"
EXPERIMENT_NAME = f"{MY_NAME}_{MY_SURNAME}"
print("MLFLOW_TRACKING_URI =", os.getenv("MLFLOW_TRACKING_URI"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def prepare_data():
    ### Ваш код здесь.
    # Загрузим датасет
    data = load_wine(as_frame=True)
    df = data.frame.copy()

    features = data.feature_names
    target = "target"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_raw = X_train.iloc[:2, :].copy()  # Сохраняем исходные данные для логирования scaler

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
    logging.info(f"Dataset loaded and prepared. Shape: train {X_train_scaled.shape}, test {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_log(name, model, X_train, y_train, X_test, y_test):
    ### Ваш код здесь.
    with mlflow.start_run(run_name=name, nested=True) as child_run:
        # Обучаем
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Логируем метрики
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })
        # Логируем параметры
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())
        else:
            mlflow.log_param("model_type", str(type(model)))

        # Логируем модель
        signature = infer_signature(X_test, model.predict(X_test))
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:2, :],
        )
        model_type = type(model).__name__
        mlflow.set_tag("model_type", model_type)
        run_id = child_run.info.run_id
        logging.info(f"Child run: {child_run.info.run_id} with model {name} finished:")

    return run_id, acc


def main():
    ### Ваш код здесь.
    client = MlflowClient()
  
    # Эксперимент (если существует, то не создаем)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        # Устанавливаем эксперимент по имени (создаст, если не существует)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logging.info(f"Created and set experiment {EXPERIMENT_NAME}")
    else:
        logging.info(f"Got existing experiment {EXPERIMENT_NAME}")

    X_train, X_test, y_train, y_test = prepare_data()

    # Родительский ран
    with mlflow.start_run(run_name="dimentel") as parent_run:
        mlflow.set_tag("RUN_LEVEL", "parent_run")

        models = [
            ("LogReg", LogisticRegression(solver="liblinear", random_state=42, max_iter=1000)),
            ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42)),
            ("GBDT", GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ]
        run_results = []

        for name, model in models:
            run_id, acc = train_and_log(
                name, model, X_train, y_train, X_test, y_test
            )
            run_results.append({"name": name, "run_id": run_id, "accuracy": acc})

    logging.info(f"Parent run: {parent_run.info.run_id} finished:")

    # Выбираем лучшую модель по accuracy
    best = max(run_results, key=lambda x: x["accuracy"])
    best_run_id = best["run_id"]
    best_acc = best["accuracy"]
    best_model_name = f"{best['name']}_{MY_SURNAME}"

    # Регистрируем в Model Registry (только лучшую)
    model_source = f"runs:/{best_run_id}/model"
    logging.info(f"Registering the best model: {best_model_name} with accuracy={best_acc:.4f} (run_id={best_run_id})")

    # Зарегистрируйте модель (но только если еще не зарегистрирована)
    try:
        client.get_registered_model(best_model_name)
        logging.info(f"Model {best_model_name} is already exist. Skipped")
    except mlflow.exceptions.RestException:
        client.create_registered_model(best_model_name)
        logging.info(f"Model {best_model_name} is registered")

    mv = client.create_model_version(
        name=best_model_name,
        source=model_source,
        run_id=best_run_id,
        description=f"{best['name']} from run {best_run_id}, {MY_NAME} {MY_SURNAME}"
    )
    logging.info(f"Model version {mv.version} created.")

    # Устанавливаем алиас staging
    client.set_registered_model_alias(best_model_name, "staging", mv.version)
    logging.info(f"Best model is {best_model_name} with accuracy={best_acc:.4f}. Version {mv.version} set as `staging`")


if __name__ == "__main__":
    main()
