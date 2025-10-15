import os
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MY_NAME = ""
MY_SURNAME = ""
EXPERIMENT_NAME = f"{YOUR_NAME}_{MY_SURNAME[0[}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")



def prepare_data():
    ### Ваш код здесь.


def train_and_log(name, model, X_train, y_train, X_test, y_test):
    ### Ваш код здесь.


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    ### Ваш код здесь.


if __name__ == "__main__":
    main()
