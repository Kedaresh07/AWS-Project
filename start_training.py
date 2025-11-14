# start_training.py
import os
import io
import time
import boto3
import pandas as pd
import xgboost as xgb
import sagemaker
from sagemaker import image_uris, estimator
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- CONFIG ----------------
ROLE_ARN = os.getenv("SM_ROLE_ARN")            # must be set as GitHub secret
BUCKET = "prathvi-raw"                         # your bucket
PROCESSED_PREFIX = "processed/"                # where ETL writes CSVs
TMP_KEY = "processed_for_xgb/train.csv"        # temporary merged train CSV for SageMaker
OUTPUT_S3 = f"s3://{BUCKET}/models/"           # SageMaker output path
INSTANCE_TYPE = "ml.m5.large"                  # change for costs
FRAMEWORK_VERSION = "1.5-1"                    # xgboost container version
NUM_ROUND = 200
EARLY_STOPPING_ROUNDS = 20
# ----------------------------------------

s3 = boto3.client("s3")
sess = sagemaker.Session()
region = sess.boto_region_name


def list_processed_csv_keys(bucket, prefix):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = resp.get("Contents", [])
    keys = [o["Key"] for o in contents if o["Key"].lower().endswith(".csv")]
    return keys


def download_and_merge_csvs(bucket, keys):
    dfs = []
    for k in keys:
        obj = s3.get_object(Bucket=bucket, Key=k)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No processed CSVs found under s3://{bucket}/{PROCESSED_PREFIX}")
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def impute_zero_median(df, cols):
    # common for Pima diabetes dataset: zeros indicate missingness for these columns
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(0, pd.NA)
            med = df[c].median()
            df[c] = df[c].fillna(med)
    return df


def prepare_numeric_label_df(df):
    # keep numeric columns only (XGBoost works best with numeric features)
    df_num = df.select_dtypes(include=["number"]).copy()
    if df_num.shape[1] < 2:
        raise RuntimeError("Not enough numeric columns after filtering.")
    label_col = "Outcome" if "Outcome" in df_num.columns else df_num.columns[-1]
    # move label to first column
    cols = [label_col] + [c for c in df_num.columns if c != label_col]
    df_out = df_num[cols]
    return df_out, label_col


def upload_df_to_s3_csv_no_header(df, bucket, key):
    # upload CSV without header (XGBoost built-in expects no header and label as first column)
    csv_bytes = df.to_csv(index=False, header=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
    return f"s3://{bucket}/{key}"


def train_local_xgb(df_train, df_val, df_test):
    # df_* have label in first column
    y_train = df_train.iloc[:, 0].values
    X_train = df_train.iloc[:, 1:].values
    y_val = df_val.iloc[:, 0].values
    X_val = df_val.iloc[:, 1:].values
    y_test = df_test.iloc[:, 0].values
    X_test = df_test.iloc[:, 1:].values

    # compute scale_pos_weight if class imbalance present
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "error",
        "verbosity": 1
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print("[INFO] Running local XGBoost training with early stopping...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_ROUND,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=10
    )

    preds_prob = bst.predict(dtest, ntree_limit=bst.best_ntree_limit if hasattr(bst, "best_ntree_limit") else NUM_ROUND)
    preds = (preds_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"\n==============================")
    print(f" Final Test Accuracy: {acc:.4f}")
    print(f" Best iteration: {getattr(bst, 'best_iteration', 'N/A')}")
    print(f"==============================\n")
    return acc


def launch_sagemaker_xgb(train_s3_uri):
    print("[INFO] Launching SageMaker built-in XGBoost training job...")
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

    xgb_estimator.set_hyperparameters(
        objective="binary:logistic",
        num_round=NUM_ROUND,
        eta=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=1
    )

    train_input = TrainingInput(s3_data=train_s3_uri, content_type="text/csv")
    xgb_estimator.fit({"train": train_input}, wait=True, logs=True)
    print("[INFO] SageMaker training completed.")
    print(f"[INFO] Model artifact: {xgb_estimator.model_data}")
    return xgb_estimator.model_data


def main():
    print("[INFO] Listing processed CSVs in S3...")
    keys = list_processed_csv_keys(BUCKET, PROCESSED_PREFIX)
    print(f"[INFO] Found {len(keys)} files under s3://{BUCKET}/{PROCESSED_PREFIX}")
    merged = download_and_merge_csvs(BUCKET, keys)
    print(f"[INFO] Merged dataframe shape: {merged.shape}")

    # Impute zeros for known columns (Pima diabetes typical)
    zero_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    merged = impute_zero_median(merged, zero_missing_cols)

    df_num, label_col = prepare_numeric_label_df(merged)
    print(f"[INFO] Numeric dataframe shape (label first): {df_num.shape}; label='{label_col}'")

    # Split: 60% train, 20% val, 20% test
    df_temp, df_test = train_test_split(df_num, test_size=0.20, random_state=42, stratify=df_num.iloc[:, 0])
    df_train, df_val = train_test_split(df_temp, test_size=0.25, random_state=42, stratify=df_temp.iloc[:, 0])  # 0.25*0.8 = 0.2

    print(f"[INFO] Train shape: {df_train.shape}, Val shape: {df_val.shape}, Test shape: {df_test.shape}")

    # Local training & accuracy
    acc = train_local_xgb(df_train, df_val, df_test)

    # Upload TRAIN CSV (label first, no header) to S3 for SageMaker built-in XGBoost
    print("[INFO] Uploading TRAIN CSV to S3 for SageMaker...")
    train_s3_uri = upload_df_to_s3_csv_no_header(df_train, BUCKET, TMP_KEY)
    print(f"[INFO] Uploaded train CSV to {train_s3_uri}")

    # Launch SageMaker built-in XGBoost training (production run)
    model_artifact = launch_sagemaker_xgb(train_s3_uri)

    # Optionally clean up the temporary merged CSV (commented out)
    # s3.delete_object(Bucket=BUCKET, Key=TMP_KEY)

    print("[INFO] Done. Final Test Accuracy (local estimate):", acc)
    print(f"[INFO] Model artifact path: {model_artifact}")


if __name__ == "__main__":
    main()
