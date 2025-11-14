# start_training.py
import os
import io
import time
import boto3
import pandas as pd
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris, estimator

# ---------- CONFIG ----------
ROLE_ARN = os.getenv("SM_ROLE_ARN")      # GitHub secret
BUCKET   = "prathvi-raw"                 # your bucket
PROCESSED_PREFIX = "processed/"          # where ETL outputs CSVs (may contain date subfolders)
TMP_KEY = "processed_for_xgb/train.csv"  # temp merged CSV used by XGBoost
OUTPUT_S3 = f"s3://{BUCKET}/models/"
INSTANCE_TYPE = "ml.m5.large"            # you can change to smaller/larger (cost tradeoff)
FRAMEWORK_VERSION = "1.5-1"              # xgboost container version
# ----------------------------

s3 = boto3.client("s3")
sess = sagemaker.Session()
region = sess.boto_region_name

def list_processed_csv_keys(bucket, prefix):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].lower().endswith(".csv")]
    return keys

def download_and_merge_csvs(bucket, keys):
    dfs = []
    for k in keys:
        obj = s3.get_object(Bucket=bucket, Key=k)
        content = obj["Body"].read()
        df = pd.read_csv(io.BytesIO(content))
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No processed CSVs found under prefix.")
    merged = pd.concat(dfs, ignore_index=True)
    return merged

def prepare_xgb_csv(df):
    # Ensure numeric-only columns (XGBoost works on numeric features)
    df_num = df.select_dtypes(include=["number"]).copy()
    if df_num.shape[1] < 2:
        raise RuntimeError("Not enough numeric columns for training after filtering.")
    # Determine label: prefer 'Outcome' else use last numeric column
    label_col = "Outcome" if "Outcome" in df_num.columns else df_num.columns[-1]
    # Move label to first column
    cols = [label_col] + [c for c in df_num.columns if c != label_col]
    df_out = df_num[cols]
    # XGBoost built-in expects CSV without header and label in first col
    return df_out

def upload_df_to_s3_csv(df, bucket, key):
    csv_bytes = df.to_csv(index=False, header=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
    return f"s3://{bucket}/{key}"

def main():
    print("[INFO] Listing processed CSVs...")
    keys = list_processed_csv_keys(BUCKET, PROCESSED_PREFIX)
    print(f"[INFO] Found {len(keys)} processed CSV files.")
    merged = download_and_merge_csvs(BUCKET, keys)
    print(f"[INFO] Merged shape: {merged.shape}")

    df_xgb = prepare_xgb_csv(merged)
    print(f"[INFO] Prepared data shape for XGBoost: {df_xgb.shape}")

    print("[INFO] Uploading merged CSV to S3 for XGBoost...")
    train_s3_uri = upload_df_to_s3_csv(df_xgb, BUCKET, TMP_KEY)
    print(f"[INFO] Uploaded to: {train_s3_uri}")

    # get XGBoost container image URI
    image_uri = image_uris.retrieve(framework="xgboost", region=region, version=FRAMEWORK_VERSION, py_version="py3", instance_type=INSTANCE_TYPE)

    xgb_estimator = estimator.Estimator(
        image_uri=image_uri,
        role=ROLE_ARN,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=OUTPUT_S3,
        sagemaker_session=sess,
        base_job_name="xgb-etl-ml"
    )

    # common hyperparameters for the container
    xgb_estimator.set_hyperparameters(
        objective="binary:logistic",  # change if regression / multi-class
        num_round=50,                 # number of boosting rounds
        eta=0.2,
        max_depth=5,
        subsample=0.8,
        verbosity=1                   # replace deprecated 'silent'
    )


    train_input = TrainingInput(s3_data=train_s3_uri, content_type="text/csv")
    print("[INFO] Starting XGBoost training job...")
    xgb_estimator.fit({"train": train_input}, wait=True, logs=True)
    print("[INFO] Training completed.")
    print(f"[INFO] Model artifact: {xgb_estimator.model_data}")

    # optional: clean up temporary merged csv from S3 to avoid storage clutter
    # s3.delete_object(Bucket=BUCKET, Key=TMP_KEY)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
