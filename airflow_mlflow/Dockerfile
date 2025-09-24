FROM apache/airflow:2.9.3-python3.10

# Копируем requirements в домашнюю директорию airflow
COPY requirements.txt /home/airflow/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /home/airflow/requirements.txt && \
    which airflow && airflow version && \
    rm -f /home/airflow/requirements.txt

WORKDIR /opt/airflow