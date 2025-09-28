import pandas as pd
import argparse
import os

def preprocess_test(input_path, output_path):
    df = pd.read_csv(input_path)

    # Misma transformación que train
    df = df.dropna()
    df = df.drop(['ID','uid','serial','model','download_usage','upload_usage'], axis=1)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessed test saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    preprocess_test(args.input, args.output)
