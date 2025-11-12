import argparse
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # SageMaker passes input/output directories as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    # --- Load processed data from S3 mount ---
    print("Loading training data...")
    all_files = [os.path.join(args.train, f) for f in os.listdir(args.train) if f.endswith(".csv")]
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # --- Split data ---
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train model ---
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # --- Evaluate ---
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.4f}")

    # --- Save model artifact ---
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
