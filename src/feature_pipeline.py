import pandas as pd
import mlflow
import boto3
from io import BytesIO
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def load_data_from_s3(bucket, key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    data = pd.read_csv(BytesIO(response['Body'].read()))
    return data

def save_data_to_s3(data, bucket, key):
    s3 = boto3.client('s3')
    csv_buffer = BytesIO()
    data.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

def main():
    # Load raw data
    raw_data = load_data_from_s3("oilprice-features", "raw/oil_data.csv")
    
    # Feature engineering: add lag feature
    raw_data['lag_1'] = raw_data['price'].shift(1)
    raw_data = raw_data.dropna()
    
    # Save processed data
    save_data_to_s3(raw_data, "oilprice-features", "features/processed.csv")
    
    # Log metrics to MLflow
    with mlflow.start_run():
        mlflow.log_metric("feature_count", len(raw_data))
        mlflow.log_param("feature_columns", raw_data.columns.tolist())
        # Save processed data locally for MLflow artifact
        processed_file = "processed.csv"
        raw_data.to_csv(processed_file, index=False)
        mlflow.log_artifact(processed_file)

if __name__ == "__main__":
    main()