from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator

with DAG(
    dag_id="w2_ex4_dockere",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
) as dag:

    docker_task = DockerOperator(
        task_id="docker_task",
        image="python:3.9-slim",
        command="python -c 'print(100)'",
        do_xcom_push=True,
    )
