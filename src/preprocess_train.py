import pandas as pd
import argparse
import os
from sklearn.preprocessing import OneHotEncoder


def preprocess_train(input_path, output_path):
    df = pd.read_csv(input_path)

    encoder = OneHotEncoder(sparse_output=False, dtype=int)  # para evitar booleanos
    encoded_model = encoder.fit_transform(df[["model"]])
    encoded_cols = encoder.get_feature_names_out(["model"])

    df_encoded = pd.DataFrame(encoded_model, columns=encoded_cols)
    df = pd.concat([df.drop(columns=["model"]), df_encoded], axis=1)
    
    df["fallo"] = (df["performance"] < 0.98).astype(int)
    df = df.drop(['ID','serial','model_A','model_B','uid','download_usage','upload_usage','model_C','performance'], axis=1)
    
    df.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessed train saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    preprocess_train(args.input, args.output)
