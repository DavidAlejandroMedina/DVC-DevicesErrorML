import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns


def exploratory_analysis(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading dataset...")
    df = pd.read_csv(input_path)

    plt.figure(figsize=(20, 20))
    df.hist(figsize=(20, 20), bins=30, edgecolor='black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histograms.png"))
    plt.close()
    print("[INFO] Histograms saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to train.csv")
    parser.add_argument("--output", required=True, help="Directory to save plots")
    args = parser.parse_args()

    exploratory_analysis(args.input, args.output)