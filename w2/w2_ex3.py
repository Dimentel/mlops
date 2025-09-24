from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

def init_pipeline():
    start_ts = datetime.utcnow().isoformat()
    return start_ts

def train_model(start_ts, bash_value):
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    import logging

    logging.info(f"pipeline_start={start_ts}")
    logging.info(f"bash_value={bash_value}")

    iris = load_iris(as_frame=True)
    df = iris.frame

    X, y = df.drop(columns=["target"]), df["target"]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    score = model.score(X, y)
    logging.info(f"Accuracy на тренировке: {score:.3f}")
    return score


default_args = {
    "owner": "liza",
    "retries": 1,
}


with DAG(
    dag_id="w2_ex3",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
) as dag:

    init = PythonOperator(
        task_id="init_pipeline",
        python_callable=init_pipeline,
    )

    # BashOperator, который читает XCom из init и просто печатает
    bash_print = BashOperator(
        task_id="print_init_ts",
        bash_command="echo 'Timestamp из XCom: {{ ti.xcom_pull(task_ids=\"init_pipeline\") }}'",
    )

    # BashOperator, который пушит новое значение в XCom
    bash_push = BashOperator(
        task_id="bash_push_value",
        bash_command="echo 123",  # stdout попадёт в XCom
        do_xcom_push=True,
    )

    # train_model теперь принимает оба значения через op_args
    train = PythonVirtualenvOperator(
        task_id="train_model",
        python_callable=train_model,
        requirements=["scikit-learn", "pandas"],
        op_args=[
            "{{ ti.xcom_pull(task_ids='init_pipeline') }}",
            "{{ ti.xcom_pull(task_ids='bash_push_value') }}",
        ],
    )

    init >> bash_print >> bash_push >> train
