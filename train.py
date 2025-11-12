# train.py
import argparse
import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_all_csvs(root: str) -> pd.DataFrame:
    """Recursively load all CSV files under root into a single DataFrame."""
    csv_paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_paths.append(os.path.join(dirpath, f))
    if not csv_paths:
        raise RuntimeError(
            f"No CSV files found under {root}. "
            "Make sure your S3 prefix (processed/) has CSVs (e.g., processed/<date>/file.csv)."
        )
    print(f"[INFO] Found {len(csv_paths)} CSV file(s).")
    df_iter = (pd.read_csv(p) for p in csv_paths)
    df = pd.concat(df_iter, ignore_index=True)
    print(f"[INFO] Combined shape: {df.shape}")
    return df


def select_features_and_label(df: pd.DataFrame):
    """Use only numeric columns. Label is 'Outcome' if present; else the last column."""
    # keep numeric columns only (diabetes dataset is numeric already, but safe)
    df_num = df.select_dtypes(include=["number"])
    if df_num.shape[1] < 2:
        raise RuntimeError("Not enough numeric columns after filtering.")

    # choose label
    label_col = "Outcome" if "Outcome" in df_num.columns else df_num.columns[-1]
    X = df_num.drop(columns=[label_col])
    y = df_num[label_col]
    print(f"[INFO] Label column: {label_col}")
    print(f"[INFO] X shape: {X.shape}, y length: {len(y)}")
    return X, y


def main(args):
    # SageMaker’s default envs if args not provided
    train_dir = args.train or os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = args.model_dir or os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    print(f"[INFO] Training data dir: {train_dir}")
    print(f"[INFO] Model output dir:  {model_dir}")

    # 1) Load data
    df = load_all_csvs(train_dir)

    # 2) Split features/label
    X, y = select_features_and_label(df)

    # 3) Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # 4) Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[RESULT] Accuracy: {acc:.4f}")

    # 5) Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=None, help="Path to training data folder")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to save model artifacts")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
