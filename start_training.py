# start_training.py
import os
import io
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb                       # Local XGBoost for accuracy calculation
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris, estimator

# ---------- CONFIG ----------
ROLE_ARN = os.getenv("SM_ROLE_ARN")
BUCKET   = "prathvi-raw"
PROCESSED_PREFIX = "processed/"
TMP_KEY = "processed_for_xgb/train.csv"
OUTPUT_S3 = f"s3://{BUCKET}/models/"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.5-1"
# ----------------------------

s3 = boto3.client("s3")
sess = sagemaker.Session()
region = sess.boto_region_name


def list_processed_csv_keys(bucket, prefix):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".csv")]


def download_and_merge_csvs(bucket, keys):
    dfs = []
    for k in keys:
        obj = s3.get_object(Bucket=bucket, Key=k)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def prepare_for_xgb(df):
    df_num = df.select_dtypes(include=["number"]).copy()
    label_col = "Outcome" if "Outcome" in df_num.columns else df_num.columns[-1]
    cols = [label_col] + [c for c in df_num.columns if c != label_col]
    return df_num[cols], label_col


def upload_train_csv(df, bucket, key):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=df.to_csv(index=False, header=False).encode("utf-8")
    )
    return f"s3://{bucket}/{key}"


def main():
    print("[INFO] Finding processed CSV files...")
    keys = list_processed_csv_keys(BUCKET, PROCESSED_PREFIX)
    print(f"[INFO] Found: {len(keys)} files")

    df_merged = download_and_merge_csvs(BUCKET, keys)
    df_prepared, label_col = prepare_for_xgb(df_merged)

    print("[INFO] Performing local train/test split for accuracy…")
    y = df_prepared.iloc[:, 0]
    X = df_prepared.iloc[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- LOCAL XGBoost training (for accuracy) ----
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "eta": 0.2,
        "subsample": 0.8
    }

    print("[INFO] Training local XGBoost model...")
    bst = xgb.train(params, dtrain, num_boost_round=50)

    preds = (bst.predict(dtest) > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)

    print(f"\n==============================")
    print(f" Final Accuracy: {acc:.4f}")
    print(f"==============================\n")

    # ---- Upload only the TRAIN data to SageMaker ----
    df_train_only = pd.concat([y_train, X_train], axis=1)
    train_s3_uri = upload_train_csv(df_train_only, BUCKET, TMP_KEY)
    print(f"[INFO] Uploaded TRAIN CSV to: {train_s3_uri}")

    # ---- Start SageMaker XGBoost Training ----
    print("[INFO] Starting SageMaker XGBoost job...")
    image_uri = image_uris.retrieve(
        "xgboost", region=region, version=FRAMEWORK_VERSION
    )

    xgb_estimator = estimator.Estimator(
        image_uri=image_uri,
        role=ROLE_ARN,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=OUTPUT_S3,
        sagemaker_session=sess,
    )

    xgb_estimator.set_hyperparameters(
        objective="binary:logistic",
        num_round=50,
        eta=0.2,
        max_depth=5,
        subsample=0.8,
        verbosity=1
    )

    train_input = TrainingInput(s3_data=train_s3_uri, content_type="text/csv")

    xgb_estimator.fit({"train": train_input}, wait=True, logs=True)
    print("[INFO] SageMaker training completed.")
    print(f"[INFO] Model saved to: {xgb_estimator.model_data}")


if __name__ == "__main__":
    main()
