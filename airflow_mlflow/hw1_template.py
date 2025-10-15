from datetime import datetime
import io
import logging
import os
import tempfile
import requests
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable

# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "S3_CONNECTION"
S3_BUCKET = Variable.get("S3_BUCKET")
MY_NAME = ""
MY_SURNAME = ""

S3_KEY_MODEL_METRICS = f"{MY_SURNAME}/model_metrics.json"
S3_KEY_PIPELINE_METRICS = f"{MY_SURNAME}/pipeline_metrics.json"


# -----------------
# Утилиты: работа с S3 через BytesIO
# -----------------
def s3_read_csv(hook: S3Hook, bucket: str, key: str) -> pd.DataFrame:
    buf = io.BytesIO()
    hook.get_conn().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pd.read_csv(buf)


def s3_write_csv(hook: S3Hook, df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    hook.get_conn().upload_fileobj(buf, bucket, key)


# -----------------
# Таски
# -----------------


def init_pipeline(**context):
    start_ts = datetime.utcnow().isoformat()
    logging.info(f"Запуск пайплайна: {start_ts}")
    context["ti"].xcom_push(key="pipeline_start", value=start_ts)


def collect_data(**context):
    ### Ваш код здесь.

def split_and_preprocess(**context):
    ### Ваш код здесь.


def train_model(**context):
    ### Ваш код здесь.


def collect_metrics_model(**context):
    ### Ваш код здесь.


def collect_metrics_pipeline(**context):
    ### Ваш код здесь.


default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}

with DAG(
    dag_id="hw1",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    t1 = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)
    t2 = PythonOperator(task_id="collect_data", python_callable=collect_data)
    t3 = PythonOperator(task_id="split_and_preprocess", python_callable=split_and_preprocess)
    t4 = PythonOperator(task_id="train_model", python_callable=train_model)
    t5 = PythonOperator(task_id="collect_metrics_model", python_callable=collect_metrics_model)
    t6 = PythonOperator(task_id="collect_metrics_pipeline", python_callable=collect_metrics_pipeline)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
