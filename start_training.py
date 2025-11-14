# start_training.py
import os
import io
import time
import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
import sagemaker
from sagemaker import image_uris, estimator
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG - edit if needed ----------------
ROLE_ARN = os.getenv("SM_ROLE_ARN")            # GitHub secret must provide this
BUCKET = "prathvi-raw"                         # your bucket
PROCESSED_PREFIX = "processed/"                # where ETL writes CSVs
TMP_KEY = "processed_for_xgb/train.csv"        # temporary merged train CSV for SageMaker
OUTPUT_S3 = f"s3://{BUCKET}/models/"           # SageMaker output path
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.5-1"
NUM_ROUND = 200
EARLY_STOPPING_ROUNDS = 20
# -------------------------------------------------------

s3 = boto3.client("s3")
sess = sagemaker.Session()
region = sess.boto_region_name


def list_processed_csv_keys(bucket, prefix):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = resp.get("Contents", []) if resp else []
    keys = [o["Key"] for o in contents if o["Key"].lower().endswith(".csv")]
    return keys


def download_and_merge_csvs(bucket, keys):
    dfs = []
    for k in keys:
        obj = s3.get_object(Bucket=bucket, Key=k)
        content = obj["Body"].read()
        df = pd.read_csv(io.BytesIO(content))
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No processed CSVs found under s3://{bucket}/{PROCESSED_PREFIX}")
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def coerce_and_clean_label_firstcol(df):
    """
    Ensure the first column is label, coerce to numeric, drop rows where label is NaN.
    Returns cleaned df with label as first column.
    """
    # ensure at least one column
    if df.shape[1] < 1:
        raise RuntimeError("Dataframe has no columns")

    # Move label to first column if not already (we assume ETL puts label either first or named 'diagnosis'/'Outcome')
    # If a label column exists by name, prefer it and move to front
    for candidate in ("diagnosis", "Outcome", "outcome", "target", "label"):
        if candidate in df.columns:
            if df.columns[0] != candidate:
                cols = [candidate] + [c for c in df.columns if c != candidate]
                df = df[cols]
            break

    # Coerce first column to numeric (safe), convert common string labels to numeric before coerce
    lbl = df.columns[0]
    # mapping for common string values to numeric
    mapping = {"m":1, "malignant":1, "malignant.":1, "malign":1,
               "b":0, "benign":0, "benign.":0, "benignlike":0}
    df[lbl] = df[lbl].astype(str).str.strip().str.lower().map(mapping).combine_first(df[lbl])

    # now coerce to numeric
    df[lbl] = pd.to_numeric(df[lbl], errors='coerce')

    before = df.shape[0]
    df = df.dropna(subset=[lbl])            # drop rows where label became NaN
    df = df.reset_index(drop=True)
    after = df.shape[0]
    print(f"[INFO] Dropped {before-after} rows with invalid/missing labels.")
    return df


def impute_numeric_medians(df):
    # For each numeric column, fill missing with median (compute medians on existing numeric values)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    medians = {}
    for c in num_cols:
        med = df[c].median()
        if not pd.isna(med):
            medians[c] = med
            df[c] = df[c].fillna(med)
    return df, medians


def upload_df_to_s3_csv_no_header(df, bucket, key):
    csv_bytes = df.to_csv(index=False, header=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
    return f"s3://{bucket}/{key}"


def train_local_xgb(X_train, y_train, X_val, y_val, X_test, y_test):
    # DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # compute scale_pos_weight
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

    print("[INFO] Local XGBoost training (early stopping)...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_ROUND,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=10
    )

    # Predict using best_iteration (modern xgboost uses iteration_range)
    if hasattr(bst, "best_iteration") and bst.best_iteration is not None:
        preds_prob = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    else:
        preds_prob = bst.predict(dtest)
    preds = (preds_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"[RESULT] Final Test Accuracy: {acc:.4f}")
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
    print("[INFO] Listing processed CSVs...")
    keys = list_processed_csv_keys(BUCKET, PROCESSED_PREFIX)
    print(f"[INFO] Found {len(keys)} processed CSV files.")
    merged = download_and_merge_csvs(BUCKET, keys)
    print(f"[INFO] Merged shape: {merged.shape}")

    # ensure header/columns consistent
    # coerce first column as label and drop NaNs
    merged = coerce_and_clean_label_firstcol(merged)

    # keep numeric columns only (label might already be numeric)
    # but keep label (first col) even if non-numeric (we coerced it)
    label_col = merged.columns[0]
    # convert remaining columns to numeric where possible
    for c in merged.columns[1:]:
        merged[c] = pd.to_numeric(merged[c], errors='coerce')

    # impute numeric medians
    merged, medians = impute_numeric_medians(merged)
    print(f"[INFO] After imputation shape: {merged.shape}")

    # Split dataset: 60% train, 20% val, 20% test
    df_temp, df_test = train_test_split(merged, test_size=0.20, random_state=42, stratify=merged[label_col])
    df_train, df_val = train_test_split(df_temp, test_size=0.25, random_state=42, stratify=df_temp[label_col])

    print(f"[INFO] Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")

    # prepare arrays for local training
    y_train = df_train.iloc[:, 0].astype(int).values
    X_train = df_train.iloc[:, 1:].values.astype(float)
    y_val = df_val.iloc[:, 0].astype(int).values
    X_val = df_val.iloc[:, 1:].values.astype(float)
    y_test = df_test.iloc[:, 0].astype(int).values
    X_test = df_test.iloc[:, 1:].astype(float)

    # scale features (optional but often helps)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # local training & accuracy
    acc = train_local_xgb(X_train, y_train, X_val, y_val, X_test, y_test)

    # reconstruct TRAIN CSV (label first) from df_train to upload (without header)
    print("[INFO] Uploading TRAIN CSV to S3 for SageMaker...")
    train_s3_uri = upload_df_to_s3_csv_no_header(df_train, BUCKET, TMP_KEY)
    print(f"[INFO] Uploaded train CSV to: {train_s3_uri}")

    # launch SageMaker built-in XGBoost
    model_artifact = launch_sagemaker_xgb(train_s3_uri)

    print("[INFO] Done. Local Test Accuracy:", acc)
    print("[INFO] Model artifact path:", model_artifact)


if __name__ == "__main__":
    main()
