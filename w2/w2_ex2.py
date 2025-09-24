from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator

from datetime import datetime

def init_pipeline(**context):
    ts = datetime.utcnow().isoformat()
    return ts
    

def train_model(ts):
    from sklearn.datasets import load_iris
    import logging
    from sklearn.linear_model import LogisticRegression
    logging.info(f"pipeline_start={ts}")
    
    data = load_iris(as_frame=True)
    df = data.frame
    X, y = df.drop(columns=["target"]), df["target"]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    score = model.score(X, y)
    logging.info(f"Accuracy на тренировке: {score:.3f}")


default_args = {
    "owner": "liza",
    "retries": 1,
}



with DAG(
    dag_id="w2_ex2",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args
) as dag:

    init = PythonOperator(task_id="init", python_callable=init_pipeline)
    train = PythonVirtualenvOperator(
        task_id="train_model",
        python_callable=train_model,
        requirements=["scikit-learn", "pandas"],
    )

    init >> train