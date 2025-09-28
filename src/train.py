import pandas as pd
import argparse
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(input_path, model_path):
    df = pd.read_csv(input_path)
    X = df.drop("fallo", axis=1)
    y = df["fallo"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"[INFO] Validation Accuracy: {acc:.4f}")

    joblib.dump(model, model_path)
    print(f"[INFO] Model saved to: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    train_model(args.input, args.model)