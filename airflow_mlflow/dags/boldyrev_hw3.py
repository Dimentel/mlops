import io
import os
import logging
from datetime import datetime

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from airflow.providers.amazon.aws.hooks.s3 import S3Hook


# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "S3_CONNECTION"
S3_BUCKET_ID = "S3_BUCKET"
MY_NAME = "dmitrii"
MY_SURNAME = "boldyrev"
MLFLOW_EXPERIMENT_NAME = f"{MY_SURNAME}_{MY_NAME}_Final"

S3_BUCKET = Variable.get("S3_BUCKET")
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
AWS_ACCESS_KEY_ID = Variable.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = Variable.get("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = Variable.get("AWS_ENDPOINT_URL")
AWS_DEFAULT_REGION = Variable.get("AWS_DEFAULT_REGION")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# -----------------
# Утилиты: работа с S3 через BytesIO
# -----------------
def s3_read_csv(hook: S3Hook, bucket: str, key: str) -> pd.DataFrame:
    file = hook.download_file(key=key, bucket_name=bucket)
    
    return pd.read_csv(file)

def s3_write_csv(hook: S3Hook, df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    hook.load_file_obj(file_obj=buf, key=key, bucket_name=bucket, replace=True)

# -----------------
# Таски
# -----------------
def init_pipeline(**context):
    start_ts = datetime.utcnow().isoformat()
    logging.info(f"pipeline_start={start_ts}")
    context["ti"].xcom_push(key="pipeline_start", value=start_ts)

def collect_data(**context):
    ### Ваш код здесь.
    import io
    import pandas as pd
    from sklearn import datasets

    # Получим датасет wines
    wines_raw_data = datasets.load_wine(as_frame=True)
    features = wines_raw_data["feature_names"]
    target = "target"
    context["ti"].xcom_push(key="feature_names", value=features)
    context["ti"].xcom_push(key="target", value=target)

    # Объединим фичи и таргет в один np.array
    data = pd.concat([wines_raw_data["data"], pd.DataFrame(wines_raw_data["target"])], axis=1)

    # Сохраним сырые данные в S3
    BUCKET = Variable.get(S3_BUCKET_ID)
    s3_hook = S3Hook(AWS_CONN_ID)
    s3_write_csv(hook=s3_hook, df=data, bucket=BUCKET, key=f"{MY_SURNAME}/wines_dataset.csv")
    
    logging.info("Raw data downloaded.")


def split_and_preprocess(**context):
    ### Ваш код здесь.
    import io
    from sklearn import datasets
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Используем созданный ранее S3 connection
    BUCKET = Variable.get(S3_BUCKET_ID)
    s3_hook = S3Hook(AWS_CONN_ID)    
    data = s3_read_csv(hook=s3_hook, bucket=BUCKET, key=f"{MY_SURNAME}/wines_dataset.csv")
    features = context["ti"].xcom_pull(key="feature_names")
    target = context["ti"].xcom_pull(key="target")

    # Сделаем препроцессинг
    # Разделим на фичи и таргет
    X, y = data[features], data[target]

    # Разделим данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучим скалер на train
    scaler = StandardScaler()
    X_train_fitted = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns)
    X_test_fitted = pd.DataFrame(scaler.transform(X_test),
                                 columns=X_test.columns)

    data_to_save = [X_train_fitted, X_test_fitted, y_train, y_test]
    names_of_datapeaces = ["X_train_fitted", "X_test_fitted", "y_train", "y_test"]

    for name, data_peace in zip(names_of_datapeaces, data_to_save):
        # Сохраним данные в S3
        s3_write_csv(hook=s3_hook, df=data_peace, bucket=BUCKET, key=f"{MY_SURNAME}/{name}.csv")

    logging.info("Preprocess was finished. Preprocessed data saved.")


def train_and_log_mlflow(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    AWS_ENDPOINT_URL,
    **context,
):
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_ENDPOINT_URL"] = AWS_ENDPOINT_URL
    os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

    ### Ваш код здесь.
    # Используем созданный ранее S3 connection
    # Загрузим выборки для обучения
    s3_hook = S3Hook(AWS_CONN_ID)
    BUCKET = Variable.get(S3_BUCKET_ID)
    X_train_fitted = s3_read_csv(hook=s3_hook, bucket=BUCKET, key=f"{MY_SURNAME}/X_train_fitted.csv")
    y_train = s3_read_csv(hook=s3_hook, bucket=BUCKET, key=f"{MY_SURNAME}/y_train.csv")
    X_test_fitted = s3_read_csv(hook=s3_hook, bucket=BUCKET, key=f"{MY_SURNAME}/X_test_fitted.csv")
    y_test = s3_read_csv(hook=s3_hook, bucket=BUCKET, key=f"{MY_SURNAME}/y_test.csv")

    logging.info("Training data is downloaded. Starting training models and selecting the best one.")

    def train_and_log(name, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(run_name=name, nested=True) as child_run:
            # Обучаем
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            # Логируем метрики
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                "f1": f1_score(y_test, preds, average="weighted", zero_division=0)
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
            logging.info(f"Child run: {child_run.info.run_id} with model {name} finished.")
    
        return run_id, acc
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
  
    # Эксперимент (если существует, то не создаем)
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        # Устанавливаем эксперимент по имени (создаст, если не существует)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logging.info(f"Created and set experiment {MLFLOW_EXPERIMENT_NAME}")
    else:
        logging.info(f"Got existing experiment {MLFLOW_EXPERIMENT_NAME}")

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
                name, model, X_train_fitted, y_train, X_test_fitted, y_test
            )
            run_results.append({"name": name, "run_id": run_id, "accuracy": acc})

    logging.info(f"Parent run: {parent_run.info.run_id} finished.")

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
    logging.info(f"Model {best_model_name} version {mv.version} created.")

    # Устанавливаем алиас staging
    client.set_registered_model_alias(best_model_name, "staging", mv.version)
    logging.info(f"Best model is {best_model_name} with accuracy={best_acc:.4f}. Version {mv.version} set as `staging`")

    return best_run_id


def serve_model(**context):
    ### Ваш код здесь.
    import subprocess
    import time
    import requests
    import json
    from mlflow import MlflowClient
    import mlflow.pyfunc
    
    best_run_id = context["ti"].xcom_pull(task_ids="train_and_log_mlflow")
    
    if not best_run_id:
        raise ValueError("No best_run_id found from previous task")
    
    model_uri = f"runs:/{best_run_id}/model"
    logging.info(f"Starting serving for model: {model_uri}")
    
    # Получаем input_example через MLflow Client
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    input_example = None
    # Создадим пример вручную
    manual_example = {"dataframe_split":
                      {"columns": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", 
                                      "total_phenols", "flavanoids", "nonflavanoid_phenols",
                                        "proanthocyanins", "color_intensity", "hue",
                                        "od280/od315_of_diluted_wines", "proline"],
                           "data": [
                                      [
                                        0.808733375074741,
                                        0.6373187412153815,
                                        0.7157857907062145,
                                        -1.241280355132674,
                                        1.0655672175876454,
                                        0.6466366889064012,
                                        1.0272423668544342,
                                        -1.5493209362378155,
                                        0.0893605295377359,
                                        0.0182522279697858,
                                        0.0155169482461234,
                                        1.0661342138680747,
                                        0.3654871511124797
                                      ],
                                      [
                                        1.506217438777986,
                                        1.4619533435557177,
                                        0.2844919479327384,
                                        -0.1665132183714564,
                                        0.7230806967695248,
                                        0.8826840149339304,
                                        0.6474808012510822,
                                        -0.5322347170717389,
                                        -0.6155947590377376,
                                        0.0785270273118702,
                                        -0.3702935548384197,
                                        1.0244440013558849,
                                        1.145551508811453
                                      ]
                               ]
                            }
                        }

    # Пробуем несколько способов получить input_example
    try:
        # Из артефактов модели
        input_example_path = client.download_artifacts(best_run_id, "model/input_example.json")
        with open(input_example_path, 'r') as f:
            input_example = json.load(f)
        logging.info(f"\u2705 Loaded input_example from model artifacts")
        
    except Exception as e1:
        logging.warning(f"Could not load input_example.json: {e1}")
        try:
            # Пробуем альтернативное расположение - serving_input_payload.json
            serving_input_path = client.download_artifacts(best_run_id, "model/serving_input_payload.json")
            with open(serving_input_path, 'r') as f:
                input_example = json.load(f)
            logging.info("\u2705 Loaded input_example from serving_input_payload.json")
            
        except Exception as e2:
            logging.warning(f"Could not load serving_input_payload.json: {e2}")
            try:
                # Загружаем модель и получаем input_example из метаданных
                model = mlflow.pyfunc.load_model(model_uri)
                if hasattr(model, 'metadata') and model.metadata.get_input_example():
                    input_example = model.metadata.get_input_example()
                    logging.info("\u2705 Loaded input_example from model metadata")
                else:
                    raise Exception("No input_example in model metadata")
                    
            except Exception as e3:
                logging.warning(f"Could not load input_example from model: {e3}")
                # Используем данные из нашего пайплайна
                try:
                    s3_hook = S3Hook(AWS_CONN_ID)
                    BUCKET = Variable.get(S3_BUCKET_ID)
                    X_test = s3_read_csv(s3_hook, BUCKET, f"{MY_SURNAME}/X_test_fitted.csv")
                    
                    # Берем первую строку из тестовых данных
                    input_example = {
                        "dataframe_split": {
                            "columns": X_test.columns.tolist(),
                            "data": [X_test.iloc[:2].values.tolist()]
                        }
                    }
                    logging.info("\u2705 Created input_example from test data")
                    
                except Exception as e4:
                    logging.error(f"All methods failed: {e4}")
                    # Берём ранее созданный пример
                    input_example = manual_example
                    logging.info("\u26a0 Using manual input_example")
    
    # Форматируем input_example
    normalized_input = None
    if isinstance(input_example, dict):
        if 'dataframe_split' in input_example:
            # Уже в правильном формате
            normalized_input = input_example
            logging.info("Input example already in format of serving_input_payload.json)")
        elif 'columns' in input_example:
            # Конвертируем из формата 'inputs' в 'dataframe_split'
            normalized_input = {
                "dataframe_split": {
                    "columns": input_example.get('columns', []),
                    "data": input_example['data']
                }
            }
            logging.info("Converted input example from format of input_example.json to format of serving_input_payload.json")
        else:
            # Неизвестный формат
            logging.warning(f"Unknown input_example format: {list(input_example.keys())}")
            normalized_input = manual_example
    else:
        # Если input_example не dict (например, numpy array или список)
        logging.warning(f"Input example is not a dict: {type(input_example)}")
        normalized_input = manual_example
    
    # Логируем
    if 'dataframe_split' in normalized_input and 'columns' in normalized_input['dataframe_split']:
        num_features = len(normalized_input['dataframe_split']['columns'])
        logging.info(f"Using input_example with {num_features} features")
    else:
        logging.warning("Could not determine number of features in input_example")
        num_features = "unknown"
    
    logging.info(f"Final input_example format: {list(normalized_input.keys())}")
    
    # Запускаем MLflow serving
    server_process = subprocess.Popen([
        "mlflow", "models", "serve",
        "--model-uri", model_uri,
        "--host", "0.0.0.0", 
        "--port", "5001",
        "--no-conda"
    ], env=dict(os.environ, **{
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID, 
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "AWS_DEFAULT_REGION": AWS_DEFAULT_REGION,
        "AWS_ENDPOINT_URL": AWS_ENDPOINT_URL,
    }))
    
    try:
        logging.info("Waiting for server to start...")
        time.sleep(15)
        
        logging.info("Sending prediction request...")
        response = requests.post(
            "http://127.0.0.1:5001/invocations",
            json=normalized_input,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        response.raise_for_status()
        prediction = response.json()
        
        logging.info(f"\u2705 SUCCESS: Prediction received: {prediction}")
        logging.info(f"Model: {model_uri}")
        logging.info(f"Input features: {normalized_input['dataframe_split']['columns']}")
        
        return {
            "status": "success", 
            "prediction": prediction,
            "model_uri": model_uri,
            "input_example_source": "dynamic"
        }
        
    except Exception as e:
        logging.error(f"\u274c FAILED: Prediction test failed: {e}")
        raise
    finally:
        server_process.terminate()
        server_process.wait()
        logging.info("MLflow server stopped")

def cleanup(**context):
    ### Ваш код здесь.
    import io
    import json

    
    # Удалим временные данные
    s3_hook = S3Hook(AWS_CONN_ID)
    BUCKET = Variable.get(S3_BUCKET_ID)

    logging.info("Deleting temporary files.")
    
    # Список файлов для удаления
    files_to_delete = [
        "X_train_fitted.csv",
        "X_test_fitted.csv",
        "y_train.csv",
        "y_test.csv",
        "wines_dataset.csv"
    ]

    # Удаляем каждый файл
    for file_name in files_to_delete:
        s3_key = f"{MY_SURNAME}/{file_name}"
        try:
            # Проверяем существует ли файл перед удалением
            if s3_hook.check_for_key(s3_key, bucket_name=BUCKET):
                s3_hook.delete_objects(bucket=BUCKET, keys=[s3_key])
                logging.info(f"Deleted: {s3_key}")
            else:
                logging.info(f"File is not exist: {s3_key}")
        except Exception as e:
            logging.warning(f"Error during deleting {s3_key}: {e}")

    logging.info("Temporary files are deleted.")

default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}

with DAG(
    dag_id="hw33",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    init_pipeline = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)
    collect_data = PythonOperator(task_id="collect_data", python_callable=collect_data)
    split_and_preprocess = PythonOperator(task_id="split_and_preprocess", python_callable=split_and_preprocess)
    train_and_log_mlflow = PythonOperator(
        task_id="train_and_log_mlflow",
        python_callable=train_and_log_mlflow,
        op_kwargs={
            "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            "AWS_DEFAULT_REGION": AWS_DEFAULT_REGION,
            "AWS_ENDPOINT_URL": AWS_ENDPOINT_URL,
        },
    )
    serve_model = PythonOperator(task_id="serve_model", python_callable=serve_model) # Можете заменить на любой другой оператор!

    cleanup = PythonOperator(task_id="cleanup", python_callable=cleanup)
    
    init_pipeline >> collect_data >> split_and_preprocess >> train_and_log_mlflow >> serve_model >> cleanup
