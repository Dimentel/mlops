import os
import io
import joblib
import boto3
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ENV config.
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://storage.yandexcloud.net")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "ru-central1")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY_MODEL = os.getenv("S3_KEY_MODEL")


# Load model from S3.
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=AWS_ENDPOINT_URL,
    region_name=AWS_DEFAULT_REGION,
)


buf = io.BytesIO()
s3.download_fileobj(Bucket=S3_BUCKET, Key=S3_KEY_MODEL, Fileobj=buf)
buf.seek(0)
model = joblib.load(buf)

# API.
app = FastAPI()

class Features(BaseModel):
    inputs: list[list[float]]

@app.post("/predict")
def predict(data: Features):
    X = pd.DataFrame(data.inputs)
    preds = model.predict(X).tolist()
    return {"predictions": preds}
