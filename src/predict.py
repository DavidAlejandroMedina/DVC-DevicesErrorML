import pandas as pd
import argparse
import joblib

def predict_model(model_path, input_path, output_path):
    model = joblib.load(model_path)
    df = pd.read_csv(input_path)

    predictions = model.predict(df)
    pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predict_model(args.model, args.input, args.output)