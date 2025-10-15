from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd


def log_with_evaluate(AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    AWS_ENDPOINT_URL,
    **context):

    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_ENDPOINT_URL"] = AWS_ENDPOINT_URL
    os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

    mlflow.set_tracking_uri("http://mlflow-service:5000")
    mlflow.set_experiment("iris_evaluate_example")
    
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    evaluate_df = pd.concat([X_test, y_test], axis=1)

    with mlflow.start_run() as run:
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
    
        # логируем модель
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name="iris_rf_model"
        )
    
        # evaluate → метрики и артефакты
        result = mlflow.evaluate(
            model=model_info.model_uri,
            data=evaluate_df,
            targets="target",
            model_type="classifier",
            evaluators=["default"]
        )
    
        model_uri = model_info.model_uri
        return model_uri



with DAG(
    dag_id="w4_example",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlflow", "evaluate"],
) as dag:

    log_experiment = PythonOperator(
        task_id="log_with_evaluate",
        python_callable=log_with_evaluate,
        provide_context=True,
        op_kwargs={
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY"),
            "AWS_DEFAULT_REGION": "ru-central1",
            "AWS_ENDPOINT_URL": "https://storage.yandexcloud.net",
        },
    )

    serve_model = BashOperator(
        task_id="serve_model",
        bash_command=(
            "export PATH=$PATH:/home/airflow/.local/bin\n"
            "MODEL_URI={{ ti.xcom_pull(task_ids='log_with_evaluate') | trim | replace(\"'\", \"\") }}\n"
            "echo Using MODEL_URI=$MODEL_URI\n"
            "mlflow models serve "
            "--model-uri $MODEL_URI "
            "--host 0.0.0.0 --port 5001 --no-conda &\n"
            "SERVER_PID=$!\n"
            "sleep 10\n"
            "curl --fail -X POST http://127.0.0.1:5001/invocations "
            "-H 'Content-Type: application/json' "
            "-d '{\"inputs\": [[5.1, 3.5, 1.4, 0.2]]}'\n"
            "STATUS=$?\n"
            "kill $SERVER_PID || true\n"
            "exit $STATUS\n"
        ),
        env={
            "MLFLOW_TRACKING_URI": Variable.get("MLFLOW_TRACKING_URI"),
            "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY"),
            "AWS_DEFAULT_REGION": "ru-central1",
            "AWS_ENDPOINT_URL": "https://storage.yandexcloud.net",
            
            
        },
    )

    log_experiment >> serve_model