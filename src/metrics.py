import argparse
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model_path, input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model = joblib.load(model_path)
    df = pd.read_csv(input_path)

    if "fallo" not in df.columns:
        raise ValueError("The dataset must contain the target column 'fallo'")

    X = df.drop("fallo", axis=1)
    y_true = df["fallo"]

    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    print(f"[INFO] Metrics: {metrics}")
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, zero_division=0))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    print("[INFO] Metrics and confusion matrix saved at:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--input", required=True, help="Path to validation/test dataset with target")
    parser.add_argument("--output", required=True, help="Directory to save metrics and plots")
    args = parser.parse_args()

    evaluate_model(args.model, args.input, args.output)
