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
S3_BUCKET_ID = "S3_BUCKET"
MY_NAME = "dmitrii"
MY_SURNAME = "boldyrev"

S3_KEY_MODEL_METRICS = f"{MY_SURNAME}/model_metrics.json"
S3_KEY_PIPELINE_METRICS = f"{MY_SURNAME}/pipeline_metrics.json"

logging.basicConfig(filename=f"{MY_SURNAME}_hw1_dag.log", level=logging.INFO)
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


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

    # Сохраним сырые данные в буффер
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    # Сохраним файл в формате pkl на S3
    BUCKET = Variable.get(S3_BUCKET_ID)
    s3_hook = S3Hook(AWS_CONN_ID)
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"{BUCKET}/{MY_SURNAME}/wines_dataset.pkl",
        bucket_name=BUCKET,
        replace=True,
    )
    _LOG.info("Raw data downloaded.")


def split_and_preprocess(**context):
    ### Ваш код здесь.
    import io
    from sklearn import datasets
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Используем созданный ранее S3 connection
    s3_hook = S3Hook(AWS_CONN_ID)
    BUCKET = Variable.get(S3_BUCKET_ID)
    file = s3_hook.download_file(key=f"{BUCKET}/{MY_SURNAME}/wines_dataset.pkl", bucket_name=BUCKET)
    data = pd.read_pickle(file)
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
        # Сохраним данные в буффер
        filebuffer = io.BytesIO()
        data_peace.to_pickle(filebuffer)
        filebuffer.seek(0)

        # Сохраним файл в формате pkl на S3
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"{BUCKET}/{MY_SURNAME}/{name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )
    _LOG.info("Preprocess was finished. Preprocessed data saved.")


def train_model(**context):
    ### Ваш код здесь.
    import io
    import pickle
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    # Используем созданный ранее S3 connection
    # Загрузим выборки для обучения
    s3_hook = S3Hook(AWS_CONN_ID)
    BUCKET = Variable.get(S3_BUCKET_ID)
    X_train_fitted = pd.read_pickle(
        s3_hook.download_file(
            key=f"{BUCKET}/{MY_SURNAME}/X_train_fitted.pkl", bucket_name=BUCKET
        )
    )

    y_train = pd.read_pickle(
        s3_hook.download_file(
            key=f"{BUCKET}/{MY_SURNAME}/y_train.pkl", bucket_name=BUCKET
        )
    )

    # Обучим модель
    model = RandomForestClassifier()

    start_ts = datetime.utcnow().isoformat()
    logging.info(f"Запуск обучения: {start_ts}")

    model.fit(X_train_fitted, y_train)

    finish_ts = datetime.utcnow().isoformat()
    logging.info(f"Обучение завершено: {finish_ts}")
    context["ti"].xcom_push(key="train_duration", value=start_ts)

    # Сохраним данные в буффер
    filebuffer = io.BytesIO()

    # Используем pickle.dump()
    pickle.dump(model, filebuffer)
    filebuffer.seek(0)

    # Сохраним модель в формате pkl на S3
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"{BUCKET}/{MY_SURNAME}/rf_model.pkl",
        bucket_name=BUCKET,
        replace=True,
    )
    _LOG.info(f"Model {model} is fitted and saved")


def collect_metrics_model(**context):
    ### Ваш код здесь.
    import io
    import json
    import pandas as pd
    from sklearn.metrics import f1_score, accuracy_score

    # Используем созданный ранее S3 connection
    # Загрузим выборки для инференса
    s3_hook = S3Hook(AWS_CONN_ID)
    BUCKET = Variable.get(S3_BUCKET_ID)
    X_test_fitted = pd.read_pickle(
        s3_hook.download_file(
            key=f"{BUCKET}/{MY_SURNAME}/X_test_fitted.pkl", bucket_name=BUCKET
        )
    )

    y_test = pd.read_pickle(
        s3_hook.download_file(
            key=f"{BUCKET}/{MY_SURNAME}/y_test.pkl", bucket_name=BUCKET
        )
    )
    # Загрузим модель
    model = pd.read_pickle(
        s3_hook.download_file(
            key=f"{BUCKET}/{MY_SURNAME}/rf_model.pkl", bucket_name=BUCKET
        )
    )
    # Рассчитаем метрики модели
    y_pred = model.predict(X_test_fitted)

    metrics = {}
    metrics["f1"] = f1_score(y_test, y_pred, average='macro')
    metrics["accuracy"] = accuracy_score(y_test, y_pred)

    # Сохраним метрики модели в json
    filebuffer = io.BytesIO()
    filebuffer.write(json.dumps(metrics).encode())
    filebuffer.seek(0)

    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"{BUCKET}/{MY_SURNAME}/model_metrics.json",
        bucket_name=BUCKET,
        replace=True,
    )
    finish_ts = datetime.utcnow().isoformat()
    logging.info(f"Завершение пайплайна: {finish_ts}")
    context["ti"].xcom_push(key="pipeline_finished", value=finish_ts)


def collect_metrics_pipeline(**context):
    ### Ваш код здесь.
    import io
    import json
    pipeline_metrics = {
        "pipeline_start": context["ti"].xcom_pull(key="pipeline_start"),
        "pipeline_finish": context["ti"].xcom_pull(key="pipeline_finish"),
        "train_duration": context["ti"].xcom_pull(key="train_duration")
    }
    # Сохраним метрики пайплайна в json
    s3_hook = S3Hook(AWS_CONN_ID)
    BUCKET = Variable.get(S3_BUCKET_ID)

    filebuffer = io.BytesIO()
    filebuffer.write(json.dumps(pipeline_metrics).encode())
    filebuffer.seek(0)

    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"{BUCKET}/{MY_SURNAME}/pipeline_metrics.json",
        bucket_name=BUCKET,
        replace=True,
    )


def cleanup(**context):
    ### Ваш код здесь.
    import io
    import json

    # Удалим временные данные
    s3_hook = S3Hook(AWS_CONN_ID)
    BUCKET = Variable.get(S3_BUCKET_ID)

    # Список файлов для удаления
    files_to_delete = [
        "X_train_fitted.pkl",
        "X_test_fitted.pkl",
        "y_train.pkl",
        "y_test.pkl",
        "wines_dataset.pkl"
    ]

    # Удаляем каждый файл
    for file_name in files_to_delete:
        s3_key = f"{BUCKET}/{MY_SURNAME}/{file_name}"
        try:
            # Проверяем существует ли файл перед удалением
            if s3_hook.check_for_key(s3_key, bucket_name=BUCKET):
                s3_hook.delete_objects(bucket=BUCKET, keys=[s3_key])
                print(f"Удалён: {s3_key}")
            else:
                print(f"Файл не существует: {s3_key}")
        except Exception as e:
            print(f"Ошибка удаления {s3_key}: {e}")

    _LOG.info("Temporary files are deleted.")


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
    t7 = PythonOperator(task_id="cleanup", python_callable=cleanup)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
